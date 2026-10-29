"""PPO for the GPU navigation environment: rollout, buffer and update stay on the device (M2.5).

Same algorithm as ppo.py, with a CNN for the depth image and dict observations.
"""

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from dronerl import contract
from dronerl.agents.ppo import RunningMeanStd, gae, layer
from dronerl.envs import maps as mapgen
from dronerl.envs.nav_gpu import CONDITIONS, NavEnv, load_maps
from dronerl.eval.evaluate import score
from dronerl.utils.logger import Logger
from dronerl.utils.run import make_run_dir, parse_args, set_seed

VEC_DIM = contract.STATE_DIM + contract.LIDAR_BEAMS
EVAL_SEED = 0


@dataclass
class Config:
    env_id: str = f"nav-v{contract.VERSION}"
    seed: int = 0
    exp: str = "default"
    variant: str = "ppo_gpu"
    log_dir: str = "runs"
    device: str = "cuda"
    total_steps: int = 30_000_000
    num_envs: int = 2048
    num_steps: int = 32
    learning_rate: float = 3e-4
    anneal_lr: bool = False
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 5
    num_minibatches: int = 4
    clip_coef: float = 0.2
    clip_vloss: bool = False
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    target_kl: float | None = None
    init_std: float = 0.5
    norm_reward: bool = True
    bootstrap_truncated: bool = True
    # environment: start level, promotions, share of lower levels, see NavEnv
    level: int = 0
    curriculum: bool = False
    mix: float = 0.2
    randomize: bool = False
    prox_w: float = contract.PROXIMITY_W
    eval_every: int = 2_000_000
    # label -> map level of the validation and test envs
    eval_levels: dict = field(default_factory=lambda: dict(mapgen.TEST_LEVELS))
    # M3: sensors of the student in training and a nav_gpu.CONDITIONS name for evaluation
    sensors: dict | None = None
    eval_condition: str | None = None
    # M4.12: checkpoint whose agent the training starts from, to fine-tune it
    init: str | None = None


class Encoder(nn.Module):
    """Depth image [N, C, 48, 64] -> 128 features; stride 4 first, measured 2-4x faster in
    training than a stride 2 start (plan/M2.md)."""

    def __init__(self, channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 16, 8, 4, padding=2),
            nn.ELU(),
            nn.Conv2d(16, 32, 4, 2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.ELU(),
            nn.Flatten(),
            layer(32 * 6 * 8, 128),
            nn.ELU(),
        )

    def forward(self, depth):
        return self.net(depth)


class Head(nn.Module):
    """Own depth encoder plus an MLP on the image features, state and lidar."""

    def __init__(self, out_dim, out_std):
        super().__init__()
        self.encoder = Encoder()
        self.mlp = nn.Sequential(
            layer(128 + VEC_DIM, 256),
            nn.ELU(),
            layer(256, 256),
            nn.ELU(),
            layer(256, out_dim, out_std),
        )

    def forward(self, depth, vec):
        return self.mlp(torch.cat([self.encoder(depth), vec], -1))


class Agent(nn.Module):
    def __init__(self, init_std=0.5):
        super().__init__()
        # the depth image is already in [0, 1], only the vector part is normalized
        self.vec_rms = RunningMeanStd(VEC_DIM)
        self.actor = Head(contract.ACT_DIM, 0.01)
        self.critic = Head(1, 1.0)
        self.log_std = nn.Parameter(torch.full((contract.ACT_DIM,), math.log(init_std)))

    def vec(self, obs, update=False):
        x = torch.cat([obs["state"], obs["lidar"]], -1)
        if update:
            self.vec_rms.update(x)
        z = (x - self.vec_rms.mean) / torch.sqrt(self.vec_rms.var + 1e-8)
        return z.clamp(-10, 10).float()

    def inputs(self, obs, update=False, start=None):
        """What the rollout buffer keeps of an observation; start marks new episodes."""
        return {"depth": obs["depth"].half(), "vec": self.vec(obs, update)}

    def value(self, x):
        return self.critic(x["depth"].float(), x["vec"]).squeeze(-1)

    def value_of(self, obs):
        return self.critic(obs["depth"], self.vec(obs)).squeeze(-1)

    def dist(self, x):
        mean = self.actor(x["depth"].float(), x["vec"])
        return Normal(mean, self.log_std.exp().expand_as(mean))

    @torch.no_grad()
    def act(self, obs):
        """Deterministic action (mean of the distribution) for a raw observation."""
        return self.actor(obs["depth"], self.vec(obs))

    def policy(self):
        """act for one batch of evaluation episodes."""
        return self.act

    def save(self, path, step):
        torch.save({"agent": self.state_dict(), "step": step, "version": contract.VERSION}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device)
        agent = cls().to(device)
        agent.load_state_dict(ckpt["agent"])
        return agent.eval()


@torch.no_grad()
def validate(policy, envs):
    """Success, collision rate and returns per level of evaluation envs (one episode per map).

    policy() gives act for one batch of episodes, every validation meets the same random
    conditions.
    """
    result = {}
    for label, env in envs.items():
        env.gen.manual_seed(EVAL_SEED)
        act = policy()
        obs = env.reset()
        success, collision = torch.zeros_like(env.alive), torch.zeros_like(env.alive)
        returns = torch.zeros(env.num_envs, device=env.device)
        while env.alive.any():
            obs, reward, _, _, info = env.step(act(obs))
            success |= info["success"]
            collision |= info["collision"]
            returns += reward
        result[label] = {
            "success_rate": success.float().mean().item(),
            "collision_rate": collision.float().mean().item(),
            "returns": returns.cpu().numpy(),
        }
    return result


def eval_envs(split, levels, device, condition=None):
    """Nominal envs, or with the sensors and dynamics of a nav_gpu.CONDITIONS name."""
    randomize, sensors = condition not in (None, "clean"), CONDITIONS.get(condition)
    return {
        label: NavEnv(
            mapgen.load(split, level),
            device=device,
            autoreset=False,
            randomize=randomize,
            student=sensors,
        )
        for label, level in levels.items()
    }


def eval_row(results):
    """Mean over levels for checkpoint selection, plus the per-level rates."""
    extra = {"success_rate": float(np.mean([r["success_rate"] for r in results.values()]))}
    for label, r in results.items():
        extra[f"sr_{label}"] = r["success_rate"]
        extra[f"cr_{label}"] = r["collision_rate"]
    returns = np.concatenate([r["returns"] for r in results.values()])
    return returns, extra


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train(cfg: Config, on_eval=None, agent=None):
    """`on_eval(step, row)` runs after every validation, e.g. to report to Optuna. `agent` replaces
    the teacher's Agent, e.g. by one with an asymmetric critic (M3.11)."""
    set_seed(cfg.seed)
    dev = torch.device(cfg.device)
    torch.backends.cudnn.benchmark = True
    run_dir = make_run_dir(cfg)
    log = Logger(run_dir)
    ckpt = run_dir / "checkpoints"
    env = NavEnv(
        load_maps("train"),
        cfg.num_envs,
        dev,
        cfg.seed,
        cfg.level,
        cfg.curriculum,
        cfg.mix,
        cfg.randomize,
        cfg.prox_w,
        student=cfg.sensors,
    )
    val = eval_envs("val", cfg.eval_levels, dev, cfg.eval_condition)
    agent = (Agent(cfg.init_std) if agent is None else agent).to(dev)
    if cfg.init:
        agent.load_state_dict(torch.load(cfg.init, map_location=dev)["agent"])
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)
    ret_rms = RunningMeanStd().to(dev)

    T, N = cfg.num_steps, cfg.num_envs
    batch = T * N
    minibatch = batch // cfg.num_minibatches
    iterations = cfg.total_steps // batch
    x_buf = None  # dict like agent.inputs(), allocated at the first step
    act_buf = torch.zeros(T, N, contract.ACT_DIM, device=dev)
    logp_buf, rew_buf, val_buf, term_buf, trunc_buf, final_buf = (
        torch.zeros(T, N, device=dev) for _ in range(6)
    )
    levels = len(mapgen.DENSITY)

    obs = env.reset()
    first = torch.ones(N, dtype=torch.bool, device=dev)
    disc_return = torch.zeros(N, device=dev)
    step, next_eval, best, best_step = 0, cfg.eval_every, (-np.inf,), 0
    start = time.time()

    for it in range(iterations):
        if cfg.anneal_lr:
            optimizer.param_groups[0]["lr"] = cfg.learning_rate * (1 - it / iterations)
        # episodes finished in this iteration, per level: count, successes, collisions, timeouts,
        # length
        stats = torch.zeros(5, levels, device=dev)
        t0 = time.time()
        for t in range(T):
            with torch.no_grad():
                x = agent.inputs(obs, update=True, start=first)
                dist = agent.dist(x)
                action = dist.sample()
                logp_buf[t] = dist.log_prob(action).sum(-1)
                val_buf[t] = agent.value(x)
            if x_buf is None:
                x_buf = {k: v.new_zeros(T, *v.shape) for k, v in x.items()}
            for k, v in x.items():
                x_buf[k][t] = v
            act_buf[t] = action
            obs, reward, term, trunc, info = env.step(action)
            step += N

            done = first = term | trunc
            lvl = info["level"]
            stats[0] += torch.bincount(lvl, done.float(), minlength=levels)
            for row, key in enumerate(["success", "collision", "timeout"], 1):
                stats[row] += torch.bincount(lvl, info[key].float(), minlength=levels)
            stats[4] += torch.bincount(lvl, (info["length"] * done).float(), minlength=levels)

            if cfg.norm_reward:
                # scale by the std of the discounted return, as gymnasium's NormalizeReward does
                disc_return = disc_return * cfg.gamma + reward
                ret_rms.update(disc_return)
                reward = (reward / torch.sqrt(ret_rms.var + 1e-8)).clamp(-10, 10).float()
                disc_return = disc_return * ~done

            only_trunc = trunc & ~term
            final_buf[t] = 0
            if "final_idx" in info and len(info["final_idx"]):
                with torch.no_grad():
                    values = agent.value_of(info["final_obs"])
                final_buf[t, info["final_idx"]] = values
            rew_buf[t], term_buf[t], trunc_buf[t] = reward, term.float(), only_trunc.float()

        with torch.no_grad():
            next_value = agent.value_of(obs)
        bufs = (rew_buf, val_buf, term_buf, trunc_buf, final_buf)
        adv, returns = gae(*bufs, next_value, cfg.gamma, cfg.gae_lambda, cfg.bootstrap_truncated)
        sync(dev)
        t1 = time.time()

        b_x = {k: v.flatten(0, 1) for k, v in x_buf.items()}
        b_act = act_buf.reshape(batch, contract.ACT_DIM)
        b_logp, b_adv, b_ret, b_val = (x.reshape(batch) for x in (logp_buf, adv, returns, val_buf))
        clipfracs = []
        for _ in range(cfg.update_epochs):
            perm = torch.randperm(batch, device=dev)
            for s in range(0, batch, minibatch):
                idx = perm[s : s + minibatch]
                mb = {k: v[idx] for k, v in b_x.items()}
                dist = agent.dist(mb)
                logratio = dist.log_prob(b_act[idx]).sum(-1) - b_logp[idx]
                ratio = logratio.exp()
                entropy = dist.entropy().sum(-1).mean()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1).abs() > cfg.clip_coef).float().mean())

                mb_adv = b_adv[idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                clipped_ratio = ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(-mb_adv * ratio, -mb_adv * clipped_ratio).mean()

                value = agent.value(mb)
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
        sync(dev)
        t2 = time.time()

        var_ret = b_ret.var()
        explained_var = (1 - (b_ret - b_val).var() / var_ret).item() if var_ret > 0 else np.nan
        episodes, successes, collisions, timeouts, lengths = stats.cpu().numpy()
        total = episodes.sum()
        if total:
            log.scalar("train/success_rate", successes.sum() / total, step)
            log.scalar("train/collision_rate", collisions.sum() / total, step)
            log.scalar("train/timeout_rate", timeouts.sum() / total, step)
            log.scalar("train/episode_length", lengths.sum() / total, step)
        for lvl in np.flatnonzero(episodes):
            n = episodes[lvl]
            log.scalar(f"train/sr_level{lvl}", successes[lvl] / n, step)
            log.scalar(f"train/cr_level{lvl}", collisions[lvl] / n, step)
            log.scalar(f"train/timeout_level{lvl}", timeouts[lvl] / n, step)
        log.scalar("train/level", env.level, step)
        log.scalar("train/policy_loss", pg_loss.item(), step)
        log.scalar("train/value_loss", v_loss.item(), step)
        log.scalar("train/entropy", entropy.item(), step)
        log.scalar("train/approx_kl", approx_kl.item(), step)
        log.scalar("train/clipfrac", torch.stack(clipfracs).mean().item(), step)
        log.scalar("train/explained_variance", explained_var, step)
        log.scalar("train/policy_std", agent.log_std.exp().mean().item(), step)
        log.scalar("train/learning_rate", optimizer.param_groups[0]["lr"], step)
        log.scalar("train/sps", step / (time.time() - start), step)
        log.scalar("time/rollout_s", t1 - t0, step)
        log.scalar("time/update_s", t2 - t1, step)

        if step >= next_eval or it == iterations - 1:
            while next_eval <= step:
                next_eval += cfg.eval_every
            returns, extra = eval_row(validate(agent.policy, val))
            row = log.eval(step, returns, level=env.level, **extra)
            if on_eval is not None:
                on_eval(step, row)
            if score(row) >= best:
                best, best_step = score(row), step
                agent.save(ckpt / "best.pt", step)

    agent.save(ckpt / "last.pt", step)
    best_agent = type(agent).load(ckpt / "best.pt", dev)
    test = eval_envs("test", cfg.eval_levels, dev, cfg.eval_condition)
    returns, extra = eval_row(validate(best_agent.policy, test))
    log.final(best_step, returns, **extra)
    log.close()
    return run_dir


if __name__ == "__main__":
    train(parse_args(Config))
