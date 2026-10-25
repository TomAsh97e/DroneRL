from dronerl.train.sb3_baseline import Config, train


def check_outputs(run_dir, *names):
    for name in ("eval.csv", "final.json", "checkpoints/best.zip", *names):
        assert (run_dir / name).exists()


def test_smoke_dqn(tmp_path):
    cfg = Config(
        algo="dqn",
        env_id="CartPole-v1",
        log_dir=str(tmp_path),
        total_steps=600,
        params={"learning_starts": 100},
        eval_every=300,
        eval_episodes=1,
        final_eval_episodes=1,
    )
    check_outputs(train(cfg))


def test_smoke_ppo_normalized(tmp_path):
    cfg = Config(
        log_dir=str(tmp_path),
        total_steps=256,
        n_envs=2,
        normalize=True,
        params={"n_steps": 64, "batch_size": 64, "gamma": 0.9},
        eval_every=128,
        eval_episodes=1,
        final_eval_episodes=1,
    )
    check_outputs(train(cfg), "checkpoints/best_vecnormalize.pkl")
