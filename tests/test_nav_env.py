import math

import numpy as np
import pytest
import torch

from dronerl import contract
from dronerl.envs import maps as mapgen
from dronerl.envs import raycast
from dronerl.envs.nav_gpu import NOMINAL, SUB_DT, SUBSTEPS, NavEnv


def scene(circles=(), boxes=(), start=(2, 5), goal=(18, 5), yaw0=0.0, size=(20, 10), walls=True):
    boxes = list(boxes) + (mapgen.outer_walls(*size) if walls else [])
    return {
        "kind": 0,
        "level": 0,
        "size": np.array(size, float),
        "circles": np.array(circles, float).reshape(-1, 3),
        "boxes": np.array(boxes, float).reshape(-1, 5),
        "start": np.array(start, float),
        "goal": np.array(goal, float),
        "yaw0": yaw0,
        "height": 3.0,
        "path": np.array([start, goal], float),
        "path_len": math.dist(start, goal),
        "attempts": 0,
    }


def env_for(*scenes, **kwargs):
    return NavEnv(mapgen.collate(list(scenes)), device="cpu", autoreset=False, **kwargs)


def fly(env, action, steps):
    a = torch.as_tensor(action, dtype=torch.float32).expand(env.num_envs, 3)
    for _ in range(steps):
        out = env.step(a)
    return out


def reference_depth(m, pos, yaw, height):
    """Per-pixel sphere tracing of the extruded map in numpy, independent of the column method."""
    v, u = np.meshgrid(
        np.arange(contract.IMG_H) + 0.5, np.arange(contract.IMG_W) + 0.5, indexing="ij"
    )
    left = (contract.IMG_W / 2 - u) / contract.FOCAL
    up = (contract.IMG_H / 2 - v) / contract.FOCAL
    fwd, side = np.array([math.cos(yaw), math.sin(yaw)]), np.array([-math.sin(yaw), math.cos(yaw)])
    d = np.concatenate([fwd + left.reshape(-1, 1) * side, up.reshape(-1, 1)], 1)
    norm = np.linalg.norm(d, axis=1)
    d /= norm[:, None]
    origin = np.array([*pos, height])
    t, active = np.zeros(len(d)), np.ones(len(d), bool)
    for _ in range(3000):
        p = origin + t[:, None] * d
        w = np.stack([mapgen.sdf(p[:, :2], m["circles"], m["boxes"]), p[:, 2] - m["height"]], 1)
        prism = np.minimum(w.max(1), 0) + np.linalg.norm(np.maximum(w, 0), axis=1)
        s = np.minimum(prism, p[:, 2])
        active &= (s > 1e-7) & (t < 30)
        if not active.any():
            break
        t = np.where(active, t + s, t)
    depth = np.where(t < 30, t / norm, np.inf)  # forward component of the unit ray is 1/norm
    return contract.encode_depth(torch.tensor(depth.reshape(contract.IMG_H, contract.IMG_W)))


def test_depth_of_a_cylinder_ahead():
    env = env_for(scene(circles=[(5, 5, 0.5)], walls=False))
    depth = contract.decode_depth(env.reset()["depth"][0, 0]).double()
    alpha = raycast.COLUMN_ANGLES.double()
    hit = 3 * alpha.cos() - (0.25 - 9 * alpha.sin() ** 2).clamp(min=0).sqrt()
    z_obs = torch.where(9 * alpha.sin() ** 2 <= 0.25, hit * alpha.cos(), math.inf)
    slopes = raycast.ROW_SLOPES.double()[:, None]
    z_obs = torch.where(1.2 + z_obs * slopes <= 3.0, z_obs, math.inf)
    z_floor = torch.where(slopes < 0, 1.2 / -slopes, math.inf)
    expected = contract.decode_depth(contract.encode_depth(torch.minimum(z_obs, z_floor)))
    torch.testing.assert_close(depth, expected, atol=1e-4, rtol=0)
    # the middle columns see the cylinder at about d - r, the top row passes over it
    assert depth[20, 31] == pytest.approx(2.5, abs=2e-3)
    assert contract.encode_depth(depth[0, 31]) == 0


def test_empty_scene_shows_only_the_floor():
    env = env_for(scene(walls=False))
    depth = env.reset()["depth"][0, 0]
    assert (depth[: contract.IMG_H // 2] == 0).all()
    slope = raycast.ROW_SLOPES[-1].item()
    assert contract.decode_depth(depth[-1, 0]).item() == pytest.approx(1.2 / -slope, rel=1e-5)


def test_rotated_box_and_lidar():
    env = env_for(scene(boxes=[(5, 5, 2, 0.1, math.pi / 4)], walls=False))
    lidar = contract.decode_depth(env.reset()["lidar"][0])
    assert lidar[0].item() == pytest.approx(3 - 0.1 * math.sqrt(2), abs=1e-4)
    assert (env.reset()["lidar"][0, 18:54] == 0).all()  # nothing behind the drone


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_column_depth_matches_per_pixel_reference(seed):
    pool = mapgen.load("val", 5)
    rng = np.random.default_rng(seed)
    i = int(rng.integers(len(pool["start"])))
    m = mapgen.get_map(pool, i)
    # a point along the A* path, closer to obstacles than the start
    k = int(rng.integers(len(m["path"]) - 1))
    pos = m["path"][k] + rng.uniform(0.2, 0.8) * (m["path"][k + 1] - m["path"][k])
    yaw = rng.uniform(-math.pi, math.pi)
    env = env_for(dict(m, start=pos, yaw0=yaw))
    depth = env.reset()["depth"][0, 0].double()
    expected = reference_depth(m, pos, yaw, NOMINAL["height"])
    torch.testing.assert_close(depth, expected, atol=1e-4, rtol=0)


def test_rays_and_distances_match_numpy():
    pool = mapgen.load("val", 5)
    m = mapgen.get_map(pool, 3)
    env = env_for(m)
    env.reset()
    rng = np.random.default_rng(0)
    points = rng.uniform([0, 0], m["size"], (256, 2))
    obstacles = tuple(x.expand(len(points), *x.shape[1:]) for x in env.obstacles())
    d = raycast.distance(torch.tensor(points, dtype=torch.float32), obstacles)
    np.testing.assert_allclose(d.numpy(), mapgen.sdf(points, m["circles"], m["boxes"]), atol=1e-4)

    outside = points[mapgen.sdf(points, m["circles"], m["boxes"]) > 0][:32]
    angles = torch.tensor(rng.uniform(-math.pi, math.pi, (len(outside), 8)), dtype=torch.float32)
    sub = tuple(x[: len(outside)] for x in obstacles)
    t = raycast.cast(torch.tensor(outside, dtype=torch.float32), angles, sub).double().numpy()
    for o, ray_angles, hits in zip(outside, angles.double().numpy(), t):
        for angle, hit in zip(ray_angles, hits):
            direction = np.array([math.cos(angle), math.sin(angle)])
            # nothing before the hit when marching in 1 mm steps, the surface at the hit
            before = o + np.arange(0, min(hit, 40), 1e-3)[:, None] * direction
            assert mapgen.sdf(before, m["circles"], m["boxes"]).min() > -1e-4
            if np.isfinite(hit):
                at = mapgen.sdf((o + hit * direction)[None], m["circles"], m["boxes"])
                assert abs(at[0]) < 1e-3


def test_velocity_step_response_with_delay():
    env = env_for(scene(walls=False, size=(100, 100), start=(10, 50), goal=(90, 50)))
    env.reset()
    env.accel[:] = 100  # first order only
    alpha = 1 - math.exp(-SUB_DT / NOMINAL["tau_v"])
    delay = round(NOMINAL["delay"] / SUB_DT)
    for n in range(1, 8):
        fly(env, [1, 0, 0], 1)
        expected = 1 - (1 - alpha) ** max(0, SUBSTEPS * n - delay)
        assert env.vel[0, 0].item() == pytest.approx(expected, abs=1e-5)
        assert env.vel[0, 1].item() == pytest.approx(0, abs=1e-6)


def test_acceleration_limit_and_heading_frame():
    env = env_for(scene(walls=False, size=(100, 100), start=(50, 50), yaw0=math.pi / 2))
    env.reset()
    env.accel[:] = 1.0
    fly(env, [1, 0, 0], 5)
    # facing +y, so forward is +y; 0.5 s with 40 ms of delay at 1 m/s² gives 0.46 m/s
    assert env.vel[0, 1].item() == pytest.approx(0.46, abs=1e-4)
    assert env.vel[0, 0].item() == pytest.approx(0, abs=1e-5)
    fly(env, [0, 0, 1], 30)
    assert env.yaw_rate[0].item() == pytest.approx(contract.YAW_RATE_MAX, rel=1e-4)


def test_terminal_flags_and_rewards():
    scenes = [
        scene(circles=[(3.0, 5, 0.3)]),  # straight into a pillar
        scene(goal=(2.6, 5)),  # goal just ahead
        scene(walls=False, size=(4, 10), start=(3, 5), goal=(3, 9)),  # leaves the area
        scene(),  # hovers until the time limit
    ]
    env = env_for(*scenes)
    env.reset()
    a = torch.tensor([[1.0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0]])
    flags = {k: torch.zeros(4, dtype=torch.bool) for k in ("collision", "success", "out_of_area")}
    rewards = torch.zeros(4)
    for step in range(contract.EPISODE_STEPS):
        _, reward, terminated, truncated, info = env.step(a)
        rewards += reward
        for k in flags:
            flags[k] |= info[k]
        assert truncated[3] == (step == contract.EPISODE_STEPS - 1)
    assert flags["collision"].tolist() == [True, False, False, False]
    assert flags["success"].tolist() == [False, True, False, False]
    assert flags["out_of_area"].tolist() == [False, False, True, False]
    assert rewards[0] < -9 and rewards[1] > 9
    assert not env.alive.any()
    # a frozen env neither moves nor collects reward
    pos = env.pos.clone()
    _, reward, terminated, truncated, _ = env.step(a)
    assert (reward == 0).all() and not terminated.any() and not truncated.any()
    torch.testing.assert_close(env.pos, pos)


def test_autoreset_returns_the_new_episode():
    pool = mapgen.collate([mapgen.map_from_seed((1, 0, i)) for i in range(4)])
    env = NavEnv(pool, num_envs=3, device="cpu", seed=0)
    obs = env.reset()
    env.steps[1] = contract.EPISODE_STEPS - 1
    obs, _, terminated, truncated, info = env.step(torch.zeros(3, 3))
    assert truncated.tolist() == [False, True, False] and not terminated.any()
    assert info["final_idx"].tolist() == [1] and info["length"][1] == contract.EPISODE_STEPS
    assert env.steps[1] == 0
    torch.testing.assert_close(env.pos[1], env.starts[env.map_idx[1]])
    torch.testing.assert_close(obs["state"][1], env.observe()["state"][1])
    assert not torch.equal(info["final_obs"]["state"][0], obs["state"][1])


def test_determinism():
    pool = mapgen.load("val", 4)
    actions = torch.rand(40, 16, 3, generator=torch.Generator().manual_seed(0)) * 2 - 1
    runs = []
    for _ in range(2):
        env = NavEnv(pool, num_envs=16, device="cpu", seed=3, level=4, mix=0.0, randomize=True)
        obs = [env.reset()["depth"]]
        for a in actions:
            obs.append(env.step(a)[0]["depth"])
        runs.append(torch.stack(obs))
    torch.testing.assert_close(runs[0], runs[1], atol=0, rtol=0)


def test_curriculum_promotion_and_level_mix():
    pool = mapgen.collate([mapgen.map_from_seed((1, level, 0)) for level in range(6)])
    env = NavEnv(pool, num_envs=4, device="cpu", level=0, curriculum=True)
    env.track(torch.zeros(999, dtype=torch.long), torch.ones(999, dtype=torch.bool))
    assert env.level == 0
    env.track(torch.tensor([0, 1]), torch.tensor([True, True]))  # level 1 episodes do not count
    assert env.level == 1 and len(env.window) == 0
    env.level = 3
    levels = env.sample_levels(20_000)
    share = torch.bincount(levels, minlength=6).float() / len(levels)
    torch.testing.assert_close(share[:4], torch.tensor([0.2 / 3] * 3 + [0.8]), atol=0.01, rtol=0)
    assert share[4:].sum() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cuda_smoke():
    env = NavEnv(mapgen.load("val", 5), num_envs=64, device="cuda", level=5, mix=0)
    obs = env.reset()
    for _ in range(10):
        obs, reward, *_ = env.step(torch.rand(64, 3, device="cuda") * 2 - 1)
    assert obs["depth"].is_cuda and torch.isfinite(reward).all()
    assert all(torch.isfinite(v).all() for v in obs.values())
