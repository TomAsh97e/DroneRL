import gymnasium as gym
import numpy as np
import pybullet as p
import pytest
from gymnasium.utils.env_checker import check_env as gym_check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from dronerl.agents import ppo
from dronerl.agents.scripted import PController
from dronerl.envs.goal_reach import GoalReachAviary
from dronerl.eval.goal_reach_eval import evaluate_policy, run_episode
from dronerl.eval.robustness import PerturbedGoalReach
from dronerl.train import sb3_baseline


def place(env, pos, rpy=(0, 0, 0)):
    p.resetBasePositionAndOrientation(
        env.DRONE_IDS[0], pos, p.getQuaternionFromEuler(rpy), physicsClientId=env.CLIENT
    )


@pytest.mark.parametrize("act", ["rpm", "vel"])
def test_env_checkers(act):
    env = GoalReachAviary(act=act)
    gym_check_env(env, skip_render_check=True)
    sb3_check_env(env)
    assert env.observation_space.shape == (17 if act == "rpm" else 16,)
    env.close()


def test_gym_make():
    env = gym.make("dronerl.envs:GoalReach-v0", act="rpm", reward="sparse")
    assert env.unwrapped.ACT == "rpm" and env.unwrapped.REWARD == "sparse"
    env.close()


def test_start_and_goal_ranges():
    env = GoalReachAviary()
    for seed in range(50):
        obs, _ = env.reset(seed=seed)
        start, goal = env.pos[0], env.goal
        assert np.all(np.abs(start[:2]) <= 1) and 0.5 <= start[2] <= 1.5
        assert 0.5 <= np.linalg.norm(goal - start) <= 2.0
        assert np.all(np.abs(goal[:2]) <= 2) and 0.5 <= goal[2] <= 2.0
        np.testing.assert_allclose(obs[:3], goal - start, atol=1e-6)
    env.close()


@pytest.mark.parametrize("act", ["rpm", "vel"])
def test_determinism(act):
    actions = np.random.default_rng(0).uniform(-1, 1, (100, 4 if act == "rpm" else 3))
    runs = []
    for _ in range(2):
        env = GoalReachAviary(act=act)
        env.reset(seed=5)
        env.reset(seed=7)  # an earlier episode must not leak into the next one
        obs = [env.reset(seed=3)[0]]
        for a in actions:
            o, _, terminated, truncated, _ = env.step(a)
            obs.append(o)
            if terminated or truncated:
                break
        runs.append(np.array(obs))
        env.close()
    np.testing.assert_array_equal(runs[0], runs[1])


def test_dense_reward_sign():
    env = GoalReachAviary(act="vel", reward="dense")
    for sign in (1, -1):
        env.reset(seed=1)
        for step in range(24):
            direction = sign * (env.goal - env.pos[0]) / np.linalg.norm(env.goal - env.pos[0])
            _, reward, terminated, _, _ = env.step(direction)
            # the first steps only tilt the drone, judge once it is moving
            if step >= 8:
                assert reward * sign > 0
            if terminated:
                break
    env.close()


def test_success_and_crash():
    env = GoalReachAviary(act="vel")
    env.reset(seed=0)
    place(env, env.goal + [0.05, 0, 0])
    _, _, terminated, _, info = env.step(np.zeros(3))
    assert terminated and info["is_success"] and not info["crash"]

    env.reset(seed=0)
    place(env, [0, 0, 0.04])
    _, reward, terminated, _, info = env.step(np.zeros(3))
    assert terminated and info["crash"] and not info["is_success"] and reward < -9

    env.reset(seed=0)
    place(env, [2.1, 0, 1])
    _, _, terminated, _, info = env.step(np.zeros(3))
    assert terminated and info["out_of_bounds"]
    env.close()


def test_truncation_after_episode_sec():
    env = GoalReachAviary(act="vel", episode_sec=1.0)
    env.reset(seed=0)
    env.goal = np.array([10.0, 0, 1])  # unreachable, the episode can only time out
    for step in range(48):
        _, _, terminated, truncated, _ = env.step(np.zeros(3))
        assert not terminated and truncated == (step == 47)
    env.close()


@pytest.mark.parametrize("controller", ["p", "hover"])
def test_potential_shaping_telescopes(controller):
    gamma = 0.99
    env = GoalReachAviary(act="vel", reward="potential", gamma=gamma)
    env.reset(seed=2)
    phi0 = -10 * np.linalg.norm(env.goal - env.pos[0])
    shaped, t = 0.0, 0
    while True:
        a = np.clip(2 * (env.goal - env.pos[0]), -1, 1) if controller == "p" else np.zeros(3)
        _, reward, terminated, truncated, info = env.step(a)
        sparse = 10 * info["is_success"] - 10 * (info["crash"] or info["out_of_bounds"])
        shaped += gamma**t * (reward - sparse)
        t += 1
        if terminated or truncated:
            break
    phi_end = 0.0 if terminated else -10 * info["distance"]
    assert terminated == (controller == "p")
    np.testing.assert_allclose(shaped, gamma**t * phi_end - phi0, rtol=1e-9)
    env.close()


def test_eval_protocol_p_controller():
    result = evaluate_policy("p:5", {}, episodes=3)
    m = result["metrics"]
    assert m["success_rate"] == 1.0 and m["crash_rate"] == 0.0
    assert 0.8 < m["path_efficiency"] <= 1.0


def test_train_and_evaluate_on_goal_reach(tmp_path):
    env = {"env_id": "dronerl.envs:GoalReach-v0", "env_kwargs": {"act": "vel"}}
    common = {"log_dir": str(tmp_path), "eval_every": 512, "eval_episodes": 1}
    sb3_cfg = sb3_baseline.Config(
        **env,
        **common,
        total_steps=512,
        n_envs=2,
        normalize=True,
        final_eval_episodes=1,
        params={"n_steps": 256, "batch_size": 128, "n_epochs": 1},
    )
    ppo_cfg = ppo.Config(
        **env,
        **common,
        total_steps=512,
        num_envs=2,
        num_steps=256,
        num_minibatches=2,
        update_epochs=1,
        hidden=32,
        final_eval_episodes=1,
    )
    for run_dir in (sb3_baseline.train(sb3_cfg), ppo.train(ppo_cfg)):
        result = evaluate_policy(str(run_dir), {}, episodes=2)
        assert result["env_kwargs"] == {"act": "vel"}
        assert 0.0 <= result["metrics"]["success_rate"] <= 1.0


def test_perturbed_env_nominal_matches_and_mass_scales():
    act = PController(5)
    nominal, perturbed = GoalReachAviary(), PerturbedGoalReach()
    for seed in (10_000, 10_001):
        a, b = run_episode(nominal, act, seed), run_episode(perturbed, act, seed)
        np.testing.assert_array_equal(a["pos"], b["pos"])
    heavy = PerturbedGoalReach(mass_scale=1.2)
    heavy.reset(seed=0)
    mass, _, inertia = p.getDynamicsInfo(heavy.DRONE_IDS[0], -1, physicsClientId=heavy.CLIENT)[:3]
    assert mass == pytest.approx(1.2 * heavy.M)
    np.testing.assert_allclose(inertia, 1.2 * np.diag(heavy.J))
    for env in (nominal, perturbed, heavy):
        env.close()
