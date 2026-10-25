import pytest
import torch

from dronerl.agents.ppo import Agent, Config, RunningMeanStd, gae, train


def rollout_example():
    # Step 1 is cut by the time limit, step 2 ends the episode in a terminal state
    return dict(
        rewards=torch.tensor([[1.0], [2.0], [3.0]]),
        values=torch.tensor([[0.5], [0.4], [0.3]]),
        term=torch.tensor([[0.0], [0.0], [1.0]]),
        trunc=torch.tensor([[0.0], [1.0], [0.0]]),
        final_values=torch.tensor([[0.0], [0.6], [0.0]]),
        next_value=torch.tensor([0.2]),
        gamma=0.9,
        lam=0.8,
    )


def test_gae_with_truncation():
    adv, returns = gae(**rollout_example())
    # t=2: 3 - 0.3; t=1: 2 + 0.9*0.6 - 0.4; t=0: (1 + 0.9*0.4 - 0.5) + 0.9*0.8*A1
    expected = torch.tensor([[0.86 + 0.72 * 2.14], [2.14], [2.7]])
    assert torch.allclose(adv, expected)
    assert torch.allclose(returns, expected + torch.tensor([[0.5], [0.4], [0.3]]))


def test_gae_without_truncation_bootstrap():
    adv, _ = gae(**rollout_example(), bootstrap=False)
    assert torch.allclose(adv, torch.tensor([[0.86 + 0.72 * 1.6], [1.6], [2.7]]))


def test_gae_bootstraps_from_next_value():
    reward, value, zero, next_value = [[1.0]], [[0.5]], [[0.0]], [0.2]
    args = [torch.tensor(x) for x in (reward, value, zero, zero, zero, next_value)]
    adv, _ = gae(*args, gamma=0.9, lam=0.8)
    assert adv.item() == pytest.approx(1 + 0.9 * 0.2 - 0.5)


def test_running_mean_std_and_checkpoint(tmp_path):
    scale, shift = torch.tensor([1.0, 5.0, 0.1]), torch.tensor([2.0, -1.0, 0.5])
    data = torch.randn(1000, 3, dtype=torch.float64) * scale + shift
    rms = RunningMeanStd(3)
    for chunk in data.split(100):
        rms.update(chunk)
    assert torch.allclose(rms.mean, data.mean(0), atol=1e-5)
    assert torch.allclose(rms.var, data.var(0, unbiased=False), rtol=1e-4)

    agent = Agent(3, 2, hidden=32)
    agent.obs_rms.update(data)
    z = agent.normalize(data)
    assert z.mean(0).abs().max() < 0.05 and (z.std(0) - 1).abs().max() < 0.05

    agent.save(tmp_path / "agent.pt", step=5)
    loaded = Agent.load(tmp_path / "agent.pt")
    assert loaded.hidden == 32
    assert torch.equal(loaded.obs_rms.mean, agent.obs_rms.mean)
    assert torch.equal(loaded.obs_rms.var, agent.obs_rms.var)
    obs = data[:4].numpy()
    assert (loaded.act(obs) == agent.act(obs)).all()


def test_smoke(tmp_path):
    # 2 iterations x 128 steps cross Pendulum's 200-step limit, so truncation is exercised
    cfg = Config(
        log_dir=str(tmp_path),
        total_steps=512,
        num_envs=2,
        num_steps=128,
        num_minibatches=4,
        eval_every=256,
        eval_episodes=1,
        final_eval_episodes=1,
    )
    run_dir = train(cfg)
    for name in ("eval.csv", "final.json", "checkpoints/best.pt", "checkpoints/last.pt"):
        assert (run_dir / name).exists()
