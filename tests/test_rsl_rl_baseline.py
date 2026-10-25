import json

import torch

from dronerl import contract
from dronerl.envs import maps as mapgen
from dronerl.envs.nav_gpu import NavEnv
from dronerl.train import rsl_rl_baseline


def test_vec_env_contract():
    nav = NavEnv(mapgen.load("val", 2), 8, "cpu", level=2, mix=0)
    env = rsl_rl_baseline.NavVecEnv(nav, 0.99, True)
    obs = env.get_observations()
    assert obs["depth"].shape == (8, 1, contract.IMG_H, contract.IMG_W)
    with torch.inference_mode():
        for _ in range(contract.EPISODE_STEPS + 1):
            obs, reward, dones, extras = env.step(torch.zeros(8, contract.ACT_DIM))
            if dones.any():
                break
    # hovering in place ends only by the time limit
    assert dones.all() and extras["time_outs"].all()
    assert extras["log"]["Nav/timeout"].tolist() == [1.0] * 8
    assert reward.shape == (8,) and obs.batch_size == torch.Size([8])


def test_short_training_on_cpu(tmp_path):
    cfg = rsl_rl_baseline.Config(
        device="cpu",
        log_dir=str(tmp_path),
        total_steps=256,
        num_envs=16,
        num_steps=8,
        num_minibatches=2,
        update_epochs=1,
        eval_every=128,
        eval_levels={"empty": 0},
    )
    run_dir = rsl_rl_baseline.train(cfg)
    final = json.loads((run_dir / "final.json").read_text())
    assert 0 <= final["success_rate"] <= 1 and "sr_empty" in final
    assert len((run_dir / "eval.csv").read_text().splitlines()) == 3
    act = rsl_rl_baseline.load(run_dir / "checkpoints" / "best.pt")
    obs = NavEnv(mapgen.load("val", 2), 4, "cpu", level=2, mix=0).reset()
    assert act(obs).shape == (4, contract.ACT_DIM)
