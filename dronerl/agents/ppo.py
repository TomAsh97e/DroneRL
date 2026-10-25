"""PPO for continuous actions, single file in the spirit of CleanRL."""

import time
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv
from torch.distributions import Normal

from dronerl.eval.evaluate import VAL_SEED, evaluate, score
from dronerl.utils.logger import Logger
from dronerl.utils.run import make_run_dir, parse_args, set_seed


@dataclass
class Config:
    env_id: str = "Pendulum-v1"
    env_kwargs: dict = field(default_factory=dict)
    seed: int = 0
    exp: str = "default"
    variant: str = "ppo"
    log_dir: str = "runs"
    total_steps: int = 1_000_000
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    num_envs: int = 1
    num_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 10
    num_minibatches: int = 32
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    norm_obs: bool = True
    norm_reward: bool = True
    bootstrap_truncated: bool = True
    async_envs: bool = False
    hidden: int = 64
    eval_every: int = 20_000
    eval_episodes: int = 20
    final_eval_episodes: int = 100


class RunningMeanStd(nn.Module):
    """Running mean and variance kept in buffers, so they are saved together with the model."""

    def __init__(self, shape=()):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float64))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float64))
        self.register_buffer("count", torch.tensor(1e-4, dtype=torch.float64))

    def update(self, x):
        x = torch.as_tensor(x, dtype=torch.float64)
        n, mean, var = x.shape[0], x.mean(0), x.var(0, unbiased=False)
        delta, total = mean - self.mean, self.count + n
        m2 = self.var * self.count + var * n + delta**2 * self.count * n / total
        self.mean += delta * n / total
        self.var.copy_(m2 / total)
        self.count += n


def layer(n_in, n_out, std=np.sqrt(2)):
    lin = nn.Linear(n_in, n_out)
    nn.init.orthogonal_(lin.weight, std)
    nn.init.zeros_(lin.bias)
    return lin


class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim, norm_obs=True, hidden=64):
        super().__init__()
        self.obs_dim, self.act_dim, self.norm_obs, self.hidden = obs_dim, act_dim, norm_obs, hidden
        # Normalization is part of the policy, so evaluation and export see the training statistics
        self.obs_rms = RunningMeanStd(obs_dim)
        h = hidden
        self.critic = nn.Sequential(
            layer(obs_dim, h), nn.Tanh(), layer(h, h), nn.Tanh(), layer(h, 1, 1.0)
        )
        self.actor = nn.Sequential(
            layer(obs_dim, h), nn.Tanh(), layer(h, h), nn.Tanh(), layer(h, act_dim, 0.01)
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def normalize(self, obs, update=False):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        if not self.norm_obs:
            return obs
        if update:
            self.obs_rms.update(obs)
        z = (obs - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + 1e-8)
        return z.clamp(-10, 10).float()

    def value(self, x):
        return self.critic(x).squeeze(-1)

    def dist(self, x):
        mean = self.actor(x)
        return Normal(mean, self.log_std.exp().expand_as(mean))

    def act(self, obs):
        """Deterministic action (mean of the distribution) for a raw observation."""
        with torch.no_grad():
            return self.actor(self.normalize(obs)).numpy()

    def save(self, path, step):
        meta = {
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "norm_obs": self.norm_obs,
            "hidden": self.hidden,
        }
        torch.save({"agent": self.state_dict(), "step": step, **meta}, path)

    @classmethod
    def load(cls, path):
        ckpt = torch.load(path)
        # M0 checkpoints predate the hidden size option
        agent = cls(ckpt["obs_dim"], ckpt["act_dim"], ckpt["norm_obs"], ckpt.get("hidden", 64))
        agent.load_state_dict(ckpt["agent"])
        return agent


def gae(rewards, values, term, trunc, final_values, next_value, gamma, lam, bootstrap=True):
    """Advantages and returns for a rollout of shape [T, N].

    After step t the next state is valued by values[t + 1] (next_value after the last step),
    unless the episode ended at t. A truncated episode still has a future, but with autoreset
    its true next state is final_obs, so that value goes into the reward instead.
    """
    if bootstrap:
        rewards = rewards + gamma * final_values * trunc
    nonterminal = 1.0 - torch.maximum(term, trunc)
    next_values = torch.cat([values[1:], next_value[None]])
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(next_value)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * nonterminal[t] - values[t]
        last = delta + gamma * lam * nonterminal[t] * last
        adv[t] = last
    return adv, adv + values


def make_envs(cfg, make_env):
    fns = [lambda: gym.wrappers.ClipAction(make_env()) for _ in range(cfg.num_envs)]
    vec = AsyncVectorEnv if cfg.async_envs else SyncVectorEnv
    # SAME_STEP puts the real last observation in info["final_obs"], needed for truncation
    return vec(fns, autoreset_mode=AutoresetMode.SAME_STEP)


def train(cfg: Config, make_env=None):
    make_env = make_env or (lambda: gym.make(cfg.env_id, **cfg.env_kwargs))
    set_seed(cfg.seed)
    run_dir = make_run_dir(cfg)
    log = Logger(run_dir)
    envs = make_envs(cfg, make_env)
    eval_env = gym.wrappers.ClipAction(make_env())
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]
    agent = Agent(obs_dim, act_dim, cfg.norm_obs, cfg.hidden)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)
    ret_rms = RunningMeanStd()
    ckpt = run_dir / "checkpoints"

    T, N = cfg.num_steps, cfg.num_envs
    batch = T * N
    minibatch = batch // cfg.num_minibatches
    iterations = cfg.total_steps // batch
    obs_buf, act_buf = torch.zeros(T, N, obs_dim), torch.zeros(T, N, act_dim)
    logp_buf, rew_buf, val_buf = torch.zeros(T, N), torch.zeros(T, N), torch.zeros(T, N)
    term_buf, trunc_buf, final_buf = torch.zeros(T, N), torch.zeros(T, N), torch.zeros(T, N)

    raw_obs, _ = envs.reset(seed=cfg.seed)
    disc_return = np.zeros(N)
    ep_return, ep_len = np.zeros(N), np.zeros(N, dtype=int)
    step, next_eval, best, best_step = 0, cfg.eval_every, (-np.inf,), 0
    start = time.time()

    for it in range(iterations):
        if cfg.anneal_lr:
            optimizer.param_groups[0]["lr"] = cfg.learning_rate * (1 - it / iterations)

        for t in range(T):
            obs = agent.normalize(raw_obs, update=True)
            with torch.no_grad():
                dist = agent.dist(obs)
                action = dist.sample()
                logp_buf[t] = dist.log_prob(action).sum(-1)
                val_buf[t] = agent.value(obs)
            obs_buf[t], act_buf[t] = obs, action
            raw_obs, reward, term, trunc, info = envs.step(action.numpy())
            step += N

            done = term | trunc
            ep_return += reward
            ep_len += 1
            for i in np.flatnonzero(done):
                log.scalar("train/episode_return", ep_return[i], step)
                log.scalar("train/episode_length", ep_len[i], step)
            ep_return[done], ep_len[done] = 0, 0

            if cfg.norm_reward:
                # Scale by the std of the discounted return, as gymnasium's NormalizeReward does
                disc_return = disc_return * cfg.gamma + reward
                ret_rms.update(disc_return)
                reward = np.clip(reward / np.sqrt(ret_rms.var.item() + 1e-8), -10, 10)
                disc_return[done] = 0

            only_trunc = trunc & ~term
            final_buf[t] = 0
            if only_trunc.any():
                final = np.stack(info["final_obs"][only_trunc])
                with torch.no_grad():
                    final_buf[t, only_trunc] = agent.value(agent.normalize(final))
            rew_buf[t] = torch.as_tensor(reward, dtype=torch.float32)
            term_buf[t] = torch.as_tensor(term, dtype=torch.float32)
            trunc_buf[t] = torch.as_tensor(only_trunc, dtype=torch.float32)

        with torch.no_grad():
            next_value = agent.value(agent.normalize(raw_obs))
        bufs = (rew_buf, val_buf, term_buf, trunc_buf, final_buf)
        adv, returns = gae(*bufs, next_value, cfg.gamma, cfg.gae_lambda, cfg.bootstrap_truncated)

        b_obs, b_act = obs_buf.reshape(batch, obs_dim), act_buf.reshape(batch, act_dim)
        b_logp, b_adv, b_ret, b_val = (x.reshape(batch) for x in (logp_buf, adv, returns, val_buf))
        clipfracs = []
        for _ in range(cfg.update_epochs):
            perm = torch.randperm(batch)
            for s in range(0, batch, minibatch):
                idx = perm[s : s + minibatch]
                dist = agent.dist(b_obs[idx])
                logratio = dist.log_prob(b_act[idx]).sum(-1) - b_logp[idx]
                ratio = logratio.exp()
                entropy = dist.entropy().sum(-1).mean()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1).abs() > cfg.clip_coef).float().mean().item())

                mb_adv = b_adv[idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                clipped_ratio = ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(-mb_adv * ratio, -mb_adv * clipped_ratio).mean()

                value = agent.value(b_obs[idx])
                v_loss = (value - b_ret[idx]) ** 2
                if cfg.clip_vloss:
                    change = (value - b_val[idx]).clamp(-cfg.clip_coef, cfg.clip_coef)
                    v_loss = torch.max(v_loss, (b_val[idx] + change - b_ret[idx]) ** 2)
                v_loss = 0.5 * v_loss.mean()

                loss = pg_loss - cfg.ent_coef * entropy + cfg.vf_coef * v_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()
            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                break

        var_ret = b_ret.var()
        explained_var = (1 - (b_ret - b_val).var() / var_ret).item() if var_ret > 0 else np.nan
        log.scalar("train/policy_loss", pg_loss.item(), step)
        log.scalar("train/value_loss", v_loss.item(), step)
        log.scalar("train/entropy", entropy.item(), step)
        log.scalar("train/approx_kl", approx_kl.item(), step)
        log.scalar("train/clipfrac", np.mean(clipfracs), step)
        log.scalar("train/explained_variance", explained_var, step)
        log.scalar("train/policy_std", agent.log_std.exp().mean().item(), step)
        log.scalar("train/learning_rate", optimizer.param_groups[0]["lr"], step)
        log.scalar("train/sps", step / (time.time() - start), step)

        if step >= next_eval or it == iterations - 1:
            while next_eval <= step:
                next_eval += cfg.eval_every
            returns, extra = evaluate(eval_env, agent.act, cfg.eval_episodes, VAL_SEED)
            row = log.eval(step, returns, **extra)
            if score(row) >= best:
                best, best_step = score(row), step
                agent.save(ckpt / "best.pt", step)

    agent.save(ckpt / "last.pt", step)
    best_agent = Agent.load(ckpt / "best.pt")
    returns, extra = evaluate(eval_env, best_agent.act, cfg.final_eval_episodes)
    log.final(best_step, returns, **extra)
    log.close()
    envs.close()
    eval_env.close()
    return run_dir


if __name__ == "__main__":
    train(parse_args(Config))
