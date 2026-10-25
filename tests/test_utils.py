import csv
import dataclasses
import json

import pytest
import torch
import yaml

from dronerl.eval import plot
from dronerl.eval.evaluate import score
from dronerl.utils.run import load_config, make_run_dir, set_seed


@dataclasses.dataclass
class Config:
    env_id: str = "CartPole-v1"
    env_kwargs: dict = dataclasses.field(default_factory=dict)
    seed: int = 0
    total_steps: int = 1000
    learning_rate: float = 1e-3
    double: bool = False
    target_kl: float | None = None
    exp: str = "test"
    variant: str = "algo"
    log_dir: str = "runs"


def test_set_seed():
    set_seed(3)
    a = torch.randn(5)
    set_seed(3)
    assert torch.equal(a, torch.randn(5))


def test_load_config(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("total_steps: 1e6\nenv_kwargs: {continuous: true}\n")
    overrides = ["learning_rate=3e-4", "double=true", "target_kl=0.02", "seed=7"]
    cfg = load_config(Config, path, overrides)
    assert cfg.total_steps == 1_000_000 and isinstance(cfg.total_steps, int)
    assert cfg.env_kwargs == {"continuous": True}
    assert cfg.learning_rate == 3e-4 and cfg.double is True
    assert cfg.target_kl == 0.02 and cfg.seed == 7


def test_load_config_rejects_unknown_key():
    with pytest.raises(KeyError):
        load_config(Config, overrides=["nope=1"])


def test_make_run_dir(tmp_path):
    cfg = Config(log_dir=str(tmp_path))
    first, second = make_run_dir(cfg), make_run_dir(cfg)
    assert first != second
    for name in ("config.yaml", "git.txt", "checkpoints"):
        assert (first / name).exists()
    assert yaml.safe_load((first / "config.yaml").read_text())["variant"] == "algo"


def fake_run(path, variant, seed, returns):
    path.mkdir(parents=True)
    (path / "config.yaml").write_text(yaml.safe_dump({"variant": variant, "seed": seed}))
    with (path / "eval.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "time", "mean_return", "std_return"])
        for i, ret in enumerate(returns):
            writer.writerow([(i + 1) * 100, i * 10, ret, 1.0])
    (path / "final.json").write_text(json.dumps({"mean_return": returns[-1]}))


def test_plot(tmp_path):
    for seed in range(2):
        fake_run(tmp_path / f"a_s{seed}", "a", seed, [1.0, 2.0, 3.0 + seed])
        fake_run(tmp_path / f"b_s{seed}", "b", seed, [0.0, 1.0])
    out = tmp_path / "plots" / "curves.png"
    plot.main([str(tmp_path / "*_s*"), "--out", str(out)])
    assert out.exists()
    text = out.with_suffix(".md").read_text()
    assert "| a | 2 |" in text and "| b | 2 |" in text


def test_checkpoint_score_prefers_success_rate():
    stalling = {"mean_return": 37.0, "success_rate": 0.1}
    reaching = {"mean_return": 27.0, "success_rate": 1.0}
    assert score(reaching) > score(stalling)
    assert score({"mean_return": 2.0}) > score({"mean_return": 1.0})
