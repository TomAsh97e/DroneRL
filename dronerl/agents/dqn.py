"""DQN with a replay buffer and a target network, single file in the spirit of CleanRL."""

import time
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dronerl.eval.evaluate import VAL_SEED, evaluate, score
from dronerl.utils.logger import Logger
from dronerl.utils.run import make_run_dir, parse_args, set_seed


@dataclass
class Config:
    env_id: str = "CartPole-v1"
    env_kwargs: dict = field(default_factory=dict)
    seed: int = 0
    exp: str = "default"
    variant: str = "dqn"
    log_dir: str = "runs"
    total_steps: int = 500_000
    learning_rate: float = 2.5e-4
    buffer_size: int = 10_000
    gamma: float = 0.99
    target_update: int = 500
    batch_size: int = 128
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_fraction: float = 0.5
    learning_starts: int = 10_000
    train_frequency: int = 10
    target_network: bool = True
    double: bool = False
    eval_every: int = 10_000
    eval_episodes: int = 20
    final_eval_episodes: int = 100


def q_network(obs_dim, n_actions):
    return nn.Sequential(
        nn.Linear(obs_dim, 120), nn.ReLU(), nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, n_actions)
    )


class ReplayBuffer:
    def __init__(self, size, obs_dim):
        self.obs = np.zeros((size, obs_dim), np.float32)
        self.next_obs = np.zeros((size, obs_dim), np.float32)
        self.actions = np.zeros(size, np.int64)
        self.rewards = np.zeros(size, np.float32)
        self.terminated = np.zeros(size, np.float32)
        self.size, self.pos, self.count = size, 0, 0

    def add(self, obs, action, reward, next_obs, terminated):
        i = self.pos
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.terminated[i] = next_obs, terminated
        self.pos = (i + 1) % self.size
        self.count = min(self.count + 1, self.size)

    def sample(self, n, rng):
        idx = rng.integers(0, self.count, n)
        data = (self.obs, self.actions, self.rewards, self.next_obs, self.terminated)
        return tuple(torch.as_tensor(x[idx]) for x in data)


def td_target(rewards, terminated, next_q, next_q_online, gamma, double):
    """Only a terminal state stops the bootstrap; a time-limit cut still bootstraps."""
    if double:
        # The online network picks the action and the target network evaluates it
        best = next_q_online.argmax(1, keepdim=True)
        next_v = next_q.gather(1, best).squeeze(1)
    else:
        next_v = next_q.max(1).values
    return rewards + gamma * (1 - terminated) * next_v


def train(cfg: Config):
    set_seed(cfg.seed)
    run_dir = make_run_dir(cfg)
    log = Logger(run_dir)
    env = gym.make(cfg.env_id, **cfg.env_kwargs)
    eval_env = gym.make(cfg.env_id, **cfg.env_kwargs)
    env.action_space.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    obs_dim, n_actions = env.observation_space.shape[0], env.action_space.n
    q = q_network(obs_dim, n_actions)
    target = q_network(obs_dim, n_actions)
    target.load_state_dict(q.state_dict())
    optimizer = torch.optim.Adam(q.parameters(), lr=cfg.learning_rate)
    buffer = ReplayBuffer(cfg.buffer_size, obs_dim)
    ckpt = run_dir / "checkpoints"

    def act(o):
        with torch.no_grad():
            return int(q(torch.as_tensor(o, dtype=torch.float32)).argmax())

    obs, _ = env.reset(seed=cfg.seed)
    ep_return, ep_len, best, best_step = 0.0, 0, (-np.inf,), 0
    start = time.time()
    for step in range(1, cfg.total_steps + 1):
        frac = min(1.0, step / (cfg.eps_fraction * cfg.total_steps))
        eps = cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)
        action = env.action_space.sample() if rng.random() < eps else act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        buffer.add(obs, action, reward, next_obs, terminated)
        ep_return += float(reward)
        ep_len += 1
        if terminated or truncated:
            log.scalar("train/episode_return", ep_return, step)
            log.scalar("train/episode_length", ep_len, step)
            obs, _ = env.reset()
            ep_return, ep_len = 0.0, 0
        else:
            obs = next_obs

        if step > cfg.learning_starts:
            if step % cfg.train_frequency == 0:
                b_obs, b_act, b_rew, b_next, b_term = buffer.sample(cfg.batch_size, rng)
                with torch.no_grad():
                    next_q = (target if cfg.target_network else q)(b_next)
                    next_online = q(b_next) if cfg.double else None
                    y = td_target(b_rew, b_term, next_q, next_online, cfg.gamma, cfg.double)
                pred = q(b_obs).gather(1, b_act[:, None]).squeeze(1)
                loss = F.mse_loss(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % 100 == 0:
                    log.scalar("train/td_loss", loss.item(), step)
                    log.scalar("train/q_values", pred.mean().item(), step)
                    log.scalar("train/epsilon", eps, step)
                    log.scalar("train/sps", step / (time.time() - start), step)
            if cfg.target_network and step % cfg.target_update == 0:
                target.load_state_dict(q.state_dict())

        if step % cfg.eval_every == 0 or step == cfg.total_steps:
            returns, extra = evaluate(eval_env, act, cfg.eval_episodes, VAL_SEED)
            row = log.eval(step, returns, **extra)
            if score(row) >= best:
                best, best_step = score(row), step
                torch.save({"q": q.state_dict(), "step": step}, ckpt / "best.pt")

    torch.save({"q": q.state_dict(), "step": cfg.total_steps}, ckpt / "last.pt")
    q.load_state_dict(torch.load(ckpt / "best.pt")["q"])
    returns, extra = evaluate(eval_env, act, cfg.final_eval_episodes)
    log.final(best_step, returns, **extra)
    log.close()
    env.close()
    eval_env.close()
    return run_dir


if __name__ == "__main__":
    train(parse_args(Config))
