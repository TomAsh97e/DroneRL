import json

import yaml

from dronerl.train import tune


def test_tiny_study_on_cpu(tmp_path):
    base = {"device": "cpu", "log_dir": str(tmp_path), "num_envs": 16, "eval_levels": {"empty": 0}}
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base))
    out = tmp_path / "optuna"
    args = ["--config", str(tmp_path / "base.yaml"), "--steps", "2048", "--trials", "2"]
    tune.main(args + ["--evals", "2", "--out", str(out)])
    trials = json.loads((out / "trials.json").read_text())
    # trial 0 is the starting configuration
    assert len(trials) == 2 and trials[0]["params"]["learning_rate"] == 3e-4
    assert all(t["state"] == "COMPLETE" and 0 <= t["value"] <= 1 for t in trials)
    assert (out / "history.png").exists() and "ważność" in (out / "summary.md").read_text()
