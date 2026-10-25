import torch

from dronerl.agents.dqn import Config, td_target, train


def test_td_target():
    rewards = torch.tensor([1.0, 1.0])
    terminated = torch.tensor([0.0, 1.0])
    next_q = torch.tensor([[1.0, 3.0], [2.0, 5.0]])
    next_online = torch.tensor([[2.0, 1.0], [0.0, 1.0]])

    y = td_target(rewards, terminated, next_q, next_online, 0.9, double=False)
    assert torch.allclose(y, torch.tensor([1 + 0.9 * 3.0, 1.0]))

    # Double DQN: the online network picks action 0 in the first transition
    y = td_target(rewards, terminated, next_q, next_online, 0.9, double=True)
    assert torch.allclose(y, torch.tensor([1 + 0.9 * 1.0, 1.0]))


def test_smoke(tmp_path):
    cfg = Config(
        log_dir=str(tmp_path),
        total_steps=2000,
        learning_starts=500,
        eval_every=1000,
        eval_episodes=2,
        final_eval_episodes=2,
    )
    run_dir = train(cfg)
    for name in ("eval.csv", "final.json", "checkpoints/best.pt", "checkpoints/last.pt"):
        assert (run_dir / name).exists()
