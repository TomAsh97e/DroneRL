import math

import numpy as np
import pytest
import torch

from dronerl import contract
from dronerl.agents.student import Student
from dronerl.envs import maps as mapgen
from dronerl.envs.nav_gpu import CONDITIONS, NavEnv
from dronerl.runtime import frames, safety
from dronerl.runtime.observation import Observation, student_state


def test_frames_on_known_cases():
    np.testing.assert_allclose(frames.enu_to_ned([1.0, 0.0, 0.0]), [0, 1, 0])  # east
    np.testing.assert_allclose(frames.ned_to_enu([1.0, 0.0, 0.0]), [0, 1, 0])  # north
    np.testing.assert_allclose(frames.ned_to_enu([0.0, 0.0, -2.0]), [0, 0, 2])  # up
    np.testing.assert_allclose(frames.flu_to_frd([1.0, 1.0, 1.0]), [1, -1, -1])
    assert frames.yaw_enu_to_ned(0.0) == pytest.approx(math.pi / 2)  # facing east
    assert frames.yaw_ned_to_enu(0.0) == pytest.approx(math.pi / 2)  # facing north
    assert frames.yaw_ned_to_enu(math.pi) == pytest.approx(-math.pi / 2)  # facing south
    # facing north: forward is north, left is west, turning left is a negative NED yawspeed
    vel, yawspeed = frames.command_to_ned([1.0, 0.0, 0.5], 0.0)
    np.testing.assert_allclose(vel, [1, 0], atol=1e-12)
    assert yawspeed == -0.5
    np.testing.assert_allclose(frames.command_to_ned([0.0, 1.0, 0.0], 0.0)[0], [0, -1], atol=1e-12)
    # facing east (NED yaw pi/2): left is north
    np.testing.assert_allclose(
        frames.command_to_ned([0.0, 1.0, 0.0], math.pi / 2)[0], [1, 0], atol=1e-12
    )


def test_frames_round_trips():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(20, 3))
    np.testing.assert_allclose(frames.ned_to_enu(frames.enu_to_ned(v)), v)
    np.testing.assert_allclose(frames.frd_to_flu(frames.flu_to_frd(v)), v)
    yaw = rng.uniform(-math.pi, math.pi, 20)
    np.testing.assert_allclose(
        frames.wrap(frames.yaw_ned_to_enu(frames.yaw_enu_to_ned(yaw)) - yaw), 0, atol=1e-12
    )
    start, origin = np.array([2.0, 5.0]), np.array([0.3, -0.2, -1.0])
    p = rng.uniform(0, 10, (20, 2))
    np.testing.assert_allclose(
        frames.ned_to_map(frames.map_to_ned(p, start, origin), start, origin), p
    )
    np.testing.assert_allclose(frames.map_to_ned(start, start, origin), origin[:2])
    for cmd, psi in zip(rng.normal(size=(20, 3)), yaw):
        vel, yawspeed = frames.command_to_ned(cmd, psi)
        np.testing.assert_allclose(frames.ned_to_command(vel, yawspeed, psi), cmd, atol=1e-12)


def test_command_matches_the_bridge_of_navenv():
    # NavEnv turns a heading-frame command into an ENU velocity with the yaw at its arrival
    rng = np.random.default_rng(1)
    for cmd, yaw in zip(rng.normal(size=(20, 3)), rng.uniform(-math.pi, math.pi, 20)):
        c, s = math.cos(yaw), math.sin(yaw)
        world = [c * cmd[0] - s * cmd[1], s * cmd[0] + c * cmd[1]]
        vel, _ = frames.command_to_ned(cmd, frames.yaw_enu_to_ned(yaw))
        np.testing.assert_allclose(frames.ned_to_enu(vel), world, atol=1e-12)


def scene():
    size = (20.0, 10.0)
    return {
        "kind": 0,
        "level": 0,
        "size": np.array(size),
        "circles": np.array([[10.0, 4.0, 0.5]]),
        "boxes": np.array(mapgen.outer_walls(*size)),
        "start": np.array([2.0, 5.0]),
        "goal": np.array([18.0, 3.0]),
        "yaw0": 0.4,
        "height": 3.0,
        "path": np.array([[2.0, 5.0], [18.0, 3.0]]),
        "path_len": 16.1,
        "attempts": 0,
    }


def test_state_matches_navenv_on_a_trajectory():
    m = scene()
    env = NavEnv(mapgen.collate([m]), device="cpu", autoreset=False, student=CONDITIONS["clean"])
    env.reset()
    origin = np.array([0.4, -0.3, -1.1])  # EKF origin is not the start in general
    goal = frames.map_to_ned(m["goal"], m["start"], origin)
    gen = torch.Generator().manual_seed(0)
    for _ in range(30):
        pos = frames.map_to_ned(env.pos[0].numpy(), m["start"], origin)
        vel = frames.enu_to_ned(env.vel[0].numpy())
        yaw = frames.yaw_enu_to_ned(env.yaw[0].item())
        state = student_state(
            pos, vel, yaw, -env.yaw_rate[0].item(), env.height[0].item(), goal, env.prev_action[0]
        )
        torch.testing.assert_close(state, env.student_state(env.rows)[0], atol=1e-5, rtol=1e-5)
        env.step(torch.rand(1, 3, generator=gen) * 2 - 1)


def test_frame_stack_matches_the_student():
    student = Student("stack3")
    images = torch.rand(7, 2, contract.IMG_H, contract.IMG_W)
    flights, obs, memory = [4, 3], None, None
    starts = [k == 0 for n in flights for k in range(n)]
    for image, start in zip(images, starts):
        if start:
            obs = Observation(goal=np.zeros(2))
        memory = student.stack(image[None], memory, torch.tensor([start]))
        torch.testing.assert_close(obs.image(image), memory[0])


def test_observation_keeps_the_last_action():
    obs = Observation(goal=np.array([5.0, 0.0]))
    torch.testing.assert_close(obs.prev, torch.zeros(3))
    obs.acted(torch.tensor([1.5, -0.2, 0.3]))
    state = obs.state(np.zeros(3), np.zeros(3), 0.0, 0.0, 1.2)
    torch.testing.assert_close(state[-3:], torch.tensor([1.0, -0.2, 0.3]))
    # goal 5 m north and facing north: straight ahead
    torch.testing.assert_close(state[4:7], torch.tensor([1.0, 0.0, math.log1p(5.0)]))


def test_clip_to_the_contract():
    np.testing.assert_allclose(safety.clip([2.0, -1.0, 3.0]), [1.0, -0.5, contract.YAW_RATE_MAX])
    np.testing.assert_allclose(safety.clip([-1.0, 0.2, -3.0]), [-0.3, 0.2, -contract.YAW_RATE_MAX])


def test_fence_cuts_only_outward_velocity_near_the_edges():
    box = (0.0, 0.0, 10.0, 8.0)
    np.testing.assert_allclose(safety.fence([1.0, 1.0], np.array([5.0, 4.0]), box), [1, 1])
    np.testing.assert_allclose(safety.fence([1.0, 1.0], np.array([9.8, 4.0]), box), [0, 1])
    np.testing.assert_allclose(safety.fence([-1.0, -1.0], np.array([9.8, 0.1]), box), [-1, 0])
    np.testing.assert_allclose(safety.fence([-1.0, 1.0], np.array([0.2, 7.9]), box), [0, 0])


def test_fence_limits_the_speed_towards_an_edge():
    box, pos = (0.0, 0.0, 10.0, 8.0), np.array([9.9 - safety.FENCE_MARGIN, 4.0])
    limit = 0.1 / safety.BRAKE
    np.testing.assert_allclose(safety.fence([0.1, 0.0], pos, box), [0.1, 0])
    # a fast command towards a goal near the edge slows down instead of stopping
    np.testing.assert_allclose(safety.fence([1.0, 0.0], pos, box), [limit, 0])
    # a slow command of a drone still flying out fast is cut
    np.testing.assert_allclose(safety.fence([0.1, 0.0], pos, box, now=(1.0, 0.0)), [0, 0])
    np.testing.assert_allclose(safety.fence([0.1, 0.0], pos, box, now=(-1.0, 0.0)), [0.1, 0])


def test_watchdog_and_height_band():
    f = safety.SafetyFilter()
    np.testing.assert_allclose(f(0.0, np.zeros(3), 0.0, 1.2)[0], 0)  # nothing received yet
    f.command([0.8, 0.0, 0.1], 10.0)
    cmd, reason = f(10.29, np.zeros(3), 0.0, 1.2)
    assert reason == "ok"
    np.testing.assert_allclose(cmd, [0.8, 0, 0.1])
    assert f(10.31, np.zeros(3), 0.0, 1.2)[1] == "watchdog"
    f.command([0.8, 0.0, 0.1], 10.4)
    assert f(10.5, np.zeros(3), 0.0, 1.2)[1] == "ok"
    for height in (0.3, 2.5):
        cmd, reason = f(10.5, np.zeros(3), 0.0, height)
        assert reason == "height" and not cmd.any()


def test_filter_fences_in_the_heading_frame():
    # facing east at the east edge of the arena: forward is cut, left (north) and turning are kept
    f = safety.SafetyFilter(box=(0.0, 0.0, 8.0, 10.0))
    f.command([1.0, 0.5, 0.2], 0.0)
    cmd, reason = f(0.1, np.array([4.0, 9.9, -1.2]), math.pi / 2, 1.2)
    assert reason == "fence"
    np.testing.assert_allclose(cmd, [0.0, 0.5, 0.2], atol=1e-12)
