"""Policies evaluated without retraining in conditions they were not trained on (M1.9)."""

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import pybullet as p
import torch
from gym_pybullet_drones.utils.enums import Physics

from dronerl.envs.goal_reach import GoalReachAviary
from dronerl.eval.evaluate import EVAL_SEED
from dronerl.eval.goal_reach_eval import TEST_EPISODES, load_policy, metrics, run_episode

CONDITIONS = {
    "nominalne": {},
    "masa +20%": {"env": {"mass_scale": 1.2}},
    "PYB_GND_DRAG_DW": {"env": {"physics": Physics.PYB_GND_DRAG_DW}},
    "szum obserwacji": {"noise": True},
    "opóźnienie 1": {"delay": 1},
    "opóźnienie 2": {"delay": 2},
    "wiatr 10% ciężaru": {"env": {"wind": 0.1}},
}


class PerturbedGoalReach(GoalReachAviary):
    """GoalReach with a heavier drone and a constant horizontal force along x."""

    def __init__(self, mass_scale=1.0, wind=0.0, **kwargs):
        self.mass_scale = mass_scale
        self.wind = wind
        super().__init__(**kwargs)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # the controller and HOVER_RPM keep the nominal mass, only the simulated body changes.
        # Without an explicit inertia pybullet recomputes it from the collision shape
        p.changeDynamics(
            self.DRONE_IDS[0],
            -1,
            mass=self.mass_scale * self.M,
            localInertiaDiagonal=self.mass_scale * np.diag(self.J),
            physicsClientId=self.CLIENT,
        )
        return obs, info

    def _physics(self, rpm, nth_drone):
        super()._physics(rpm, nth_drone)
        if self.wind:
            drone = self.DRONE_IDS[nth_drone]
            pos = p.getBasePositionAndOrientation(drone, physicsClientId=self.CLIENT)[0]
            force = [self.wind * self.GRAVITY, 0, 0]
            p.applyExternalForce(drone, -1, force, pos, p.WORLD_FRAME, physicsClientId=self.CLIENT)


def with_noise(act, rng):
    def noisy(obs):
        obs = obs.copy()
        pos_noise = rng.normal(0, 0.02, 3)
        obs[0:3] -= pos_noise  # goal - pos
        obs[12] += pos_noise[2]  # height
        obs[3:6] += rng.normal(0, 0.05, 3)
        return act(obs)

    return noisy


def with_delay(act, steps, act_dim):
    queue = deque([np.zeros(act_dim, dtype=np.float32)] * steps)

    def delayed(obs):
        queue.append(act(obs))
        return queue.popleft()

    return delayed


def evaluate_condition(act, kwargs, condition, episodes):
    env = PerturbedGoalReach(**kwargs, **condition.get("env", {}))
    act_dim = env.action_space.shape[0]
    runs = []
    for seed in range(EVAL_SEED, EVAL_SEED + episodes):
        policy = act
        if condition.get("noise"):
            policy = with_noise(policy, np.random.default_rng(seed))
        if condition.get("delay"):
            policy = with_delay(policy, condition["delay"], act_dim)
        runs.append(run_episode(env, policy, seed))
    result = metrics(runs, env.CTRL_FREQ, env.SUCCESS_RADIUS)
    env.close()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policies", nargs="+", help='run directories, checkpoints or "p[:k_p]"')
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=TEST_EPISODES)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    results = {}
    for spec in args.policies:
        _, name, act, kwargs = load_policy(spec)
        results[name] = {}
        for label, condition in CONDITIONS.items():
            results[name][label] = evaluate_condition(act, kwargs, condition, args.episodes)
            sr = results[name][label]["success_rate"]
            print(f"{name}, {label}: SR {100 * sr:.0f}%", flush=True)
    (args.out / "robustness.json").write_text(json.dumps(results, indent=2))
    head = ["polityka"] + list(CONDITIONS)
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, by_condition in results.items():
        cells = [f"{100 * m['success_rate']:.0f}" for m in by_condition.values()]
        lines.append("| " + " | ".join([name] + cells) + " |")
    text = "SR [%] na zbiorze testowym\n\n" + "\n".join(lines)
    (args.out / "robustness.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
