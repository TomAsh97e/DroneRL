import math

import numpy as np
import pytest

from dronerl import contract
from dronerl.envs import maps


@pytest.fixture(scope="module")
def sample():
    return {
        (level, i): maps.map_from_seed((123, level, i)) for level in (0, 2, 5) for i in range(6)
    }


def test_same_seed_same_map():
    a, b = maps.map_from_seed((7, 4, 3)), maps.map_from_seed((7, 4, 3))
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    assert not np.array_equal(a["start"], maps.map_from_seed((7, 4, 4))["start"])


def test_map_layout(sample):
    for (level, i), m in sample.items():
        assert m["kind"] == i % 2 and len(m["boxes"]) >= 4
        assert 10 <= np.linalg.norm(m["goal"] - m["start"]) <= 18
        ends = maps.sdf(np.stack([m["start"], m["goal"]]), m["circles"], m["boxes"])
        assert ends.min() >= maps.START_CLEARANCE
        obstacles = len(m["circles"]) + len(m["boxes"]) - 4
        assert obstacles == round(maps.DENSITY[level] * np.prod(m["size"]) / 100)
        straight = maps.segment_sdf(m["start"], m["goal"], m["circles"], m["boxes"])
        assert (straight < contract.COLLISION_DIST) == (level > 0)


def test_astar_path_keeps_clearance(sample):
    for m in sample.values():
        path = m["path"]
        np.testing.assert_array_equal(path[0], m["start"])
        np.testing.assert_array_equal(path[-1], m["goal"])
        for a, b in zip(path[:-1], path[1:]):
            # string pulling keeps INFLATE, the grid steps between cell centers almost
            assert maps.segment_sdf(a, b, m["circles"], m["boxes"]) > maps.INFLATE - maps.CELL
        assert m["path_len"] == pytest.approx(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def test_astar_on_small_grid():
    free = np.ones((5, 5), bool)
    free[2, :4] = False  # wall with a gap at the top
    path = maps.astar(free, (0, 0), (4, 0))
    assert path[0] == (0, 0) and path[-1] == (4, 0)
    assert all(free[c] for c in path)
    steps = [math.dist(a, b) for a, b in zip(path[:-1], path[1:])]
    # the gap is entered and left horizontally, diagonals would cut the corners of the wall
    assert sum(steps) == pytest.approx(8 + 2 * math.sqrt(2))
    free[2, 4] = False
    assert maps.astar(free, (0, 0), (4, 0)) is None


def test_sdf_of_box_and_circle():
    circles = np.array([[0.0, 0, 1]])
    boxes = np.array([[5.0, 0, 1, 0.5, np.pi / 2]])  # rotated, long side along y
    points = np.array([[0.0, 0], [2, 0], [5, 0], [5, 2], [6, 0], [5.5, 1.5]])
    expected = [-1, 1, -0.5, 1, 0.5, math.hypot(0, 0.5)]
    np.testing.assert_allclose(maps.sdf(points, circles, boxes), expected, atol=1e-9)


def test_collate_and_concat():
    ms = [maps.map_from_seed((5, level, i)) for level in (0, 5) for i in range(2)]
    a, b = maps.collate(ms[:2]), maps.collate(ms[2:])
    pool = maps.concat([a, b])
    assert len(pool["start"]) == 4 and pool["boxes"].shape[1] == max(len(m["boxes"]) for m in ms)
    for i, m in enumerate(ms):
        back = maps.get_map(pool, i)
        for key in ("circles", "boxes", "path", "start", "goal"):
            np.testing.assert_array_equal(back[key], m[key])
        # paths are padded with the goal, so a follower ends there
        np.testing.assert_array_equal(pool["path"][i, -1], m["goal"])
