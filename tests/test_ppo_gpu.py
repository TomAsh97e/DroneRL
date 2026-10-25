import json

import torch

from dronerl import contract
from dronerl.agents import ppo_gpu


def fake_obs(n):
    return {
        "depth": torch.rand(n, 1, contract.IMG_H, contract.IMG_W),
        "state": torch.randn(n, contract.STATE_DIM),
        "lidar": torch.rand(n, contract.LIDAR_BEAMS),
    }


def test_agent_shapes_and_checkpoint(tmp_path):
    agent = ppo_gpu.Agent(init_std=0.5)
    obs = fake_obs(5)
    agent.vec(obs, update=True)
    vec = agent.vec(obs)
    dist = agent.dist(obs["depth"], vec)
    assert dist.sample().shape == (5, contract.ACT_DIM)
    torch.testing.assert_close(dist.stddev[0], torch.full((3,), 0.5))
    assert agent.value(obs["depth"], vec).shape == (5,)
    agent.save(tmp_path / "a.pt", 7)
    loaded = ppo_gpu.Agent.load(tmp_path / "a.pt")
    torch.testing.assert_close(loaded.act(obs), agent.act(obs))
    torch.testing.assert_close(loaded.vec_rms.mean, agent.vec_rms.mean)


def test_short_training_on_cpu(tmp_path):
    cfg = ppo_gpu.Config(
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
    run_dir = ppo_gpu.train(cfg)
    final = json.loads((run_dir / "final.json").read_text())
    assert 0 <= final["success_rate"] <= 1 and "sr_empty" in final
    assert (run_dir / "checkpoints" / "best.pt").exists()
