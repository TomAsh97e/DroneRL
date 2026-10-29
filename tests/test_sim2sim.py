import numpy as np

from dronerl.envs import nav_gpu
from dronerl.eval import nav_eval, sim2sim


def flight(outcome, steps, start, goal):
    """run.npz of monitor_node for a straight flight from start towards goal."""
    pos = np.linspace(start, goal, steps + 1)
    flags = {k: k == outcome for k in nav_eval.OUTCOMES}
    path = float(np.linalg.norm(np.subtract(goal, start)))
    return {
        "pos": pos,
        "yaw": np.zeros(steps + 1),
        "actions": np.zeros((steps, 3)),
        "length": steps,
        "path_length": path,
        "min_clearance": 0.3,
        "reward": np.ones(steps),
        **flags,
    }


def test_series_names():
    assert sim2sim.series_of("asym_s0_gz_rgb") == ("asym_s0", "gz_rgb")
    assert sim2sim.series_of("dagger_s0_gz_depth") == ("dagger_s0", "gz_depth")
    assert sim2sim.series_of("astar") == ("astar", "gz")


def test_stacked_flights_give_the_metrics_of_nav_eval():
    runs = [flight("success", 30, (1, 1), (4, 1)), flight("collision", 12, (1, 2), (2, 2))]
    run = sim2sim.stack(runs)
    assert run["pos"].shape == (31, 2, 2) and run["alive"].sum(0).tolist() == [30, 12]
    np.testing.assert_allclose(run["pos"][-1, 1], (2, 2))
    np.testing.assert_allclose(run["return"], [30, 12])
    pool = {
        "start": np.array([[1.0, 1.0], [1.0, 2.0]]),
        "goal": np.array([[4.0, 1.0], [4.0, 2.0]]),
        "path_len": np.array([3.0, 3.0]),
    }
    m = nav_eval.metrics(run, pool)
    assert m["success_rate"] == 0.5 and m["collision_rate"] == 0.5
    np.testing.assert_allclose(m["path_efficiency"], 2.5 / 3)


def test_paired_skips_pairs_without_a_flight():
    gz = np.array([1.0, 0.0, np.nan, 1.0])
    p = np.array([1.0, 0.9, 0.0, 0.1])
    r = sim2sim.paired(gz, p)
    assert r["pairs"] == 3 and (r["lost"], r["gained"]) == (1, 1)
    np.testing.assert_allclose([r["gz_sr"], r["navenv_sr"]], [2 / 3, 2 / 3])


def test_delay_range_is_restored():
    before = nav_gpu.RANGES["delay"]
    with sim2sim.delay_range(sim2sim.M3_DELAY):
        assert nav_gpu.RANGES["delay"] == sim2sim.M3_DELAY
    assert nav_gpu.RANGES["delay"] == before
