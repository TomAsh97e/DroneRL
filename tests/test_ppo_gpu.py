import json

import torch

from dronerl import contract
from dronerl.agents import ppo_asym, ppo_gpu
from dronerl.agents.student import Student


def fake_obs(n):
    return {
        "depth": torch.rand(n, 1, contract.IMG_H, contract.IMG_W),
        "state": torch.randn(n, contract.STATE_DIM),
        "lidar": torch.rand(n, contract.LIDAR_BEAMS),
        "student_depth": torch.rand(n, 2, contract.IMG_H, contract.IMG_W).round(),
        "student_state": torch.randn(n, contract.STATE_DIM),
    }


def test_agent_shapes_and_checkpoint(tmp_path):
    agent = ppo_gpu.Agent(init_std=0.5)
    obs = fake_obs(5)
    x = agent.inputs(obs, update=True)
    dist = agent.dist(x)
    assert dist.sample().shape == (5, contract.ACT_DIM)
    torch.testing.assert_close(dist.stddev[0], torch.full((3,), 0.5))
    assert agent.value(x).shape == (5,)
    torch.testing.assert_close(agent.value_of(obs), agent.value(x), atol=1e-2, rtol=0)
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


def test_asymmetric_agent(tmp_path):
    agent = ppo_asym.AsymAgent(init_std=0.5)
    obs = [fake_obs(4) for _ in range(2)]
    agent.inputs(obs[0], update=True, start=torch.ones(4, dtype=torch.bool))
    x = agent.inputs(obs[1], update=True, start=torch.tensor([False, True, False, False]))
    assert x["image"].shape == (4, 3, contract.IMG_H, contract.IMG_W)
    # the frames of the actor: the older ones only in a running episode
    assert torch.equal(x["image"][0, 0], x["image"][0, 1])
    assert torch.equal(x["image"][1, 0], x["image"][1, 2])
    assert agent.dist(x).sample().shape == (4, contract.ACT_DIM)
    torch.testing.assert_close(agent.value_of(obs[1]), agent.value(x), atol=1e-2, rtol=0)
    agent.save(tmp_path / "a.pt", 7)
    # the checkpoint is a student as well
    policy, student = agent.policy(), ppo_asym.AsymAgent.load(tmp_path / "a.pt").policy()
    torch.testing.assert_close(Student.load(tmp_path / "a.pt").step(obs[0])[0], policy(obs[0]))
    torch.testing.assert_close(student(obs[0]), agent.actor.step(obs[0])[0])


def test_short_asymmetric_training_on_cpu(tmp_path):
    cfg = ppo_asym.Config(
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
    run_dir = ppo_asym.train(cfg)
    final = json.loads((run_dir / "final.json").read_text())
    assert 0 <= final["success_rate"] <= 1 and "sr_empty" in final
    assert Student.load(run_dir / "checkpoints" / "best.pt").history == "stack3"


def test_training_starts_from_a_checkpoint(tmp_path):
    agent = ppo_asym.AsymAgent(init_std=0.2)
    agent.save(tmp_path / "init.pt", 0)
    cfg = ppo_asym.Config(
        device="cpu",
        log_dir=str(tmp_path),
        total_steps=128,
        num_envs=16,
        num_steps=8,
        num_minibatches=2,
        update_epochs=1,
        learning_rate=0.0,
        eval_every=128,
        eval_levels={"empty": 0},
        init=str(tmp_path / "init.pt"),
    )
    run_dir = ppo_asym.train(cfg)
    trained = ppo_asym.AsymAgent.load(run_dir / "checkpoints" / "best.pt")
    # without updates the weights stay those of the checkpoint, the std is not the default one;
    # the running statistics of the inputs still follow the rollouts
    torch.testing.assert_close(dict(trained.named_parameters()), dict(agent.named_parameters()))
