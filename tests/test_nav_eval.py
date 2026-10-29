import pytest
import yaml
from rsl_rl.models import CNNModel

from dronerl.agents import ppo_gpu
from dronerl.envs import maps as mapgen
from dronerl.envs.nav_gpu import NavEnv
from dronerl.eval import nav_eval, student_compare
from dronerl.train import rsl_rl_baseline


def test_baselines_on_validation_maps(tmp_path):
    levels = {"hard": mapgen.TEST_LEVELS["hard"]}
    astar = nav_eval.evaluate("astar", "val", levels, "cpu", tmp_path, plots=True)
    straight = nav_eval.evaluate("straight", "val", levels, "cpu", tmp_path, video=2, failures=True)
    assert astar["levels"]["hard"]["success_rate"] == 1.0
    # per-step metrics need copies of the state, not views of the live CPU tensors
    assert astar["levels"]["hard"]["mean_action_change"] > 0
    assert astar["levels"]["hard"]["off_fov"] is not None
    # the straight line is blocked on every map
    assert straight["levels"]["hard"]["success_rate"] < 0.05
    assert len(astar["episodes"]["hard"]) == mapgen.SIZES["val"]
    assert list(tmp_path.glob("*/trajectories_hard.png"))
    assert len(list(tmp_path.glob("prosto*/episode_hard_map*.mp4"))) == 2
    assert list(tmp_path.glob("prosto*/failures_hard.png"))
    assert "| A* lookahead=1 k_yaw=4 | hard | 1 | 100.0 |" in nav_eval.table([astar, straight])


def fake_run(root, variant, save):
    run = root / variant
    (run / "checkpoints").mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump({"variant": variant, "seed": 3}))
    save(run / "checkpoints" / "best.pt")
    return run


def test_trained_policies_from_run_directories(tmp_path):
    obs = NavEnv(mapgen.load("val", 0), 1, "cpu").reset()
    actor = CNNModel(
        obs, {"actor": rsl_rl_baseline.GROUPS}, "actor", 3, **rsl_rl_baseline.model_cfg(0.5)
    )
    runs = [
        fake_run(tmp_path, "own", lambda p: ppo_gpu.Agent().save(p, 0)),
        fake_run(tmp_path, "rsl", lambda p: rsl_rl_baseline.save(actor, p, 0)),
    ]
    levels = {"empty": 0}
    results = [nav_eval.evaluate(str(run), "val", levels, "cpu") for run in runs]
    assert [r["name"] for r in results] == ["own s3", "rsl s3"]
    assert all(r["latency_ms"]["cpu_1_thread"] > 0 for r in results)
    text = nav_eval.table(results)
    assert "| own | empty | 1 |" in text and "| rsl s3 |" in text


def test_student_ratios():
    def result(variant, sr):
        return {"variant": variant, "levels": {"medium": {"success_rate": sr}}}

    results = [result("curriculum_rand", 0.8), result("main", 0.6), result("main", 0.8)]
    assert student_compare.ratios(results)["main"]["medium"] == pytest.approx([0.75, 1.0])
