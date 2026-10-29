import math

import pytest
import torch

from dronerl import contract as c


def test_action_scaling():
    a = torch.tensor([[0.0, 0, 0], [1, 1, 1], [-1, -1, -1], [0.5, -0.5, 0.25], [3, -3, 0]])
    cmd = c.action_to_command(a)
    expected = [
        [0, 0, 0],
        [1.0, 0.5, math.radians(60)],
        [-0.3, -0.5, -math.radians(60)],
        [0.5, -0.25, math.radians(15)],
        [1.0, -0.5, 0],  # clipped
    ]
    torch.testing.assert_close(cmd, torch.tensor(expected))
    fwd = c.action_to_command(torch.linspace(-1, 1, 101)[:, None].expand(-1, 3))[:, 0]
    assert torch.all(fwd.diff() > 0)
    torch.testing.assert_close(c.command_to_action(cmd[:4]), a[:4])


def test_depth_encoding():
    d = torch.tensor([0.1, 0.2, 1.0, 8.0, 20.0, math.inf])
    v = c.encode_depth(d)
    assert v[0] == 1 and v[1] == 1 and v[3] == 0 and v[4] == 0 and v[5] == 0
    assert 0 < v[2] < 1
    torch.testing.assert_close(c.decode_depth(v[1:4]), d[1:4])


def test_student_depth_encoding():
    d = torch.tensor([[0.1, 1.0, 20.0, math.inf, math.nan]])
    img = c.encode_student_depth(d)
    assert img.shape == (c.STUDENT_CHANNELS, 1, 5)
    torch.testing.assert_close(img[0], c.encode_depth(d.nan_to_num(math.inf)))
    # far is a valid reading, missing data is 0 in both channels
    assert img[1].tolist() == [[1, 1, 1, 1, 0]] and img[0, 0, 4] == 0
    batch = c.encode_student_depth(torch.ones(4, c.IMG_H, c.IMG_W))
    assert batch.shape == (4, c.STUDENT_CHANNELS, c.IMG_H, c.IMG_W)


def test_camera_intrinsics():
    assert c.FOCAL == pytest.approx(32)
    assert math.degrees(2 * math.atan(c.IMG_H / 2 / c.FOCAL)) == pytest.approx(73.74, abs=0.01)


def test_state_vector_in_heading_frame():
    # facing +y and flying forward, the goal 3 m away along -x is on the left
    vel, yaw, goal = torch.tensor([0.0, 0.8]), torch.tensor(math.pi / 2), torch.tensor([-3.0, 0])
    prev = torch.tensor([0.5, 0, -1])
    state = c.state_vector(vel, yaw, torch.tensor(0.1), torch.tensor(1.2), goal, prev)
    expected = torch.tensor([0.8, 0, 0.1, 1.2, 0, 1, math.log(4), 0.5, 0, -1])
    assert state.shape == (c.STATE_DIM,)
    torch.testing.assert_close(state, expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize(
    "terms, change",
    [
        ({}, 0),
        ({"progress": 0.1}, 0.1),
        ({"success": 1.0}, c.GOAL_REWARD),
        ({"collision": 1.0}, -c.COLLISION_PENALTY),
        ({"action_change": 2.0}, -2 * c.ACTION_RATE_W),
        ({"clearance": 0.25}, -c.PROXIMITY_W / 2),
        # the proximity term saturates at the collision distance
        ({"clearance": 0.0}, -c.PROXIMITY_W),
        ({"clearance": -0.2}, -c.PROXIMITY_W),
    ],
)
def test_reward_terms(terms, change):
    args = {"progress": 0.0, "success": 0.0, "collision": 0.0, "action_change": 0.0}
    args = {k: torch.tensor(v) for k, v in {**args, "clearance": 1.0, **terms}.items()}
    assert c.reward(**args).item() == pytest.approx(change - c.TIME_PENALTY)
