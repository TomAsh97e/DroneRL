"""rsl_rl PPO on the GPU navigation environment, the reference for ppo_gpu.py (M2.6).

Same network, hyperparameters, reward normalization and evaluation as ppo_gpu.py. What rsl_rl
does differently is listed in the M2 report.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from rsl_rl.env import VecEnv
from rsl_rl.models import CNNModel
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from dronerl import contract
from dronerl.agents.ppo import RunningMeanStd
from dronerl.agents.ppo_gpu import Encoder, eval_envs, eval_row, validate
from dronerl.envs import maps as mapgen
from dronerl.envs.nav_gpu import NavEnv, load_maps
from dronerl.eval.evaluate import score
from dronerl.utils.logger import Logger
from dronerl.utils.run import make_run_dir, parse_args, set_seed

GROUPS = ["depth", "state", "lidar"]


@dataclass
class Config:
    env_id: str = f"nav-v{contract.VERSION}"
    seed: int = 0
    exp: str = "default"
    variant: str = "rsl_rl"
    log_dir: str = "runs"
    device: str = "cuda"
    total_steps: int = 30_000_000
    num_envs: int = 2048
    num_steps: int = 32
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 5
    num_minibatches: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    init_std: float = 0.5
    norm_reward: bool = True
    level: int = 0
    curriculum: bool = False
    mix: float = 0.2
    randomize: bool = False
    prox_w: float = contract.PROXIMITY_W
    eval_every: int = 2_000_000
    eval_levels: dict = field(default_factory=lambda: dict(mapgen.TEST_LEVELS))


class DepthEncoder(Encoder):
    """The encoder of ppo_gpu.py with the attributes CNNModel reads from its CNNs."""

    output_channels = None
    output_dim = 128


def model_cfg(init_std=None):
    """CNNModel arguments of the actor (with the Gaussian std) or the critic."""
    # a ready encoder instead of cnn_cfg, rsl_rl's CNN has no linear layer after the convolutions;
    # CNNModel then prints that it shares encoders, which it does not
    cfg = {
        "hidden_dims": [256, 256],
        "activation": "elu",
        "obs_normalization": True,
        "cnns": {"depth": DepthEncoder()},
    }
    if init_std is not None:
        dist = {"class_name": "GaussianDistribution", "init_std": init_std, "std_type": "log"}
        cfg["distribution_cfg"] = dist
    return cfg


def train_cfg(cfg: Config):
    return {
        "num_steps_per_env": cfg.num_steps,
        "save_interval": 100,
        "obs_groups": {"actor": GROUPS, "critic": GROUPS},
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": cfg.update_epochs,
            "num_mini_batches": cfg.num_minibatches,
            "clip_param": cfg.clip_coef,
            "gamma": cfg.gamma,
            "lam": cfg.gae_lambda,
            # ppo_gpu.py halves the squared error
            "value_loss_coef": 0.5 * cfg.vf_coef,
            "entropy_coef": cfg.ent_coef,
            "learning_rate": cfg.learning_rate,
            "max_grad_norm": cfg.max_grad_norm,
            "use_clipped_value_loss": False,
            "schedule": "fixed",
            "desired_kl": None,
            "normalize_advantage_per_mini_batch": True,
        },
        "actor": {"class_name": "CNNModel", **model_cfg(cfg.init_std)},
        "critic": {"class_name": "CNNModel", **model_cfg()},
    }


class NavVecEnv(VecEnv):
    """NavEnv behind the rsl_rl interface: TensorDict observations, time-outs and episode
    statistics in `extras`."""

    def __init__(self, env: NavEnv, gamma, norm_reward):
        self.env, self.gamma, self.norm_reward = env, gamma, norm_reward
        self.num_envs, self.num_actions = env.num_envs, contract.ACT_DIM
        self.max_episode_length = contract.EPISODE_STEPS
        self.device, self.cfg = env.device, {}
        self.ret_rms = RunningMeanStd().to(env.device)
        self.disc_return = torch.zeros(env.num_envs, device=env.device)
        self.obs = TensorDict(env.reset(), batch_size=[env.num_envs])

    @property
    def episode_length_buf(self):
        return self.env.steps

    def get_observations(self):
        return self.obs

    def step(self, actions):
        obs, reward, term, trunc, info = self.env.step(actions)
        done = term | trunc
        if self.norm_reward:
            # as in ppo_gpu.py, where it happens in the training loop
            self.disc_return = self.disc_return * self.gamma + reward
            self.ret_rms.update(self.disc_return)
            reward = (reward / torch.sqrt(self.ret_rms.var + 1e-8)).clamp(-10, 10)
            self.disc_return = self.disc_return * ~done
        keys = ("success", "collision", "timeout", "length", "return")
        log = {f"Nav/{k}": info[k][done].float() for k in keys}
        self.obs = TensorDict(obs, batch_size=[self.num_envs])
        return self.obs, reward.float(), done.long(), {"time_outs": trunc & ~term, "log": log}


def save(actor, path, step):
    torch.save({"actor": actor.state_dict(), "step": step, "version": contract.VERSION}, path)


def load(path, device="cpu"):
    """The saved actor as a deterministic `act(obs) -> action`."""
    obs = {
        "depth": torch.zeros(1, 1, contract.IMG_H, contract.IMG_W),
        "state": torch.zeros(1, contract.STATE_DIM),
        "lidar": torch.zeros(1, contract.LIDAR_BEAMS),
    }
    actor = CNNModel(obs, {"actor": GROUPS}, "actor", contract.ACT_DIM, **model_cfg(1.0))
    actor.load_state_dict(torch.load(path, map_location="cpu")["actor"])
    return actor.to(device).eval()


def train(cfg: Config):
    set_seed(cfg.seed)
    dev = torch.device(cfg.device)
    torch.backends.cudnn.benchmark = True
    run_dir = make_run_dir(cfg)
    log = Logger(run_dir)
    ckpt = run_dir / "checkpoints"
    nav = NavEnv(
        load_maps("train"),
        cfg.num_envs,
        dev,
        cfg.seed,
        cfg.level,
        cfg.curriculum,
        cfg.mix,
        cfg.randomize,
        cfg.prox_w,
    )
    env = NavVecEnv(nav, cfg.gamma, cfg.norm_reward)
    runner = OnPolicyRunner(env, train_cfg(cfg), str(run_dir / "rsl_rl"), cfg.device)
    val = eval_envs("val", cfg.eval_levels, dev)

    batch = cfg.num_envs * cfg.num_steps
    iterations = cfg.total_steps // batch
    chunk = math.ceil(cfg.eval_every / batch)
    done, best, best_step = 0, (-np.inf,), 0
    while done < iterations:
        n = min(chunk, iterations - done)
        runner.learn(n)
        # learn() leaves the counter on its last iteration and opens a new writer on every call
        runner.current_learning_iteration += 1
        runner.logger.writer.close()
        done += n
        step = done * batch
        actor = runner.get_inference_policy()
        returns, extra = eval_row(validate(actor, val))
        row = log.eval(step, returns, level=nav.level, **extra)
        if score(row) >= best:
            best, best_step = score(row), step
            save(actor, ckpt / "best.pt", step)

    save(actor, ckpt / "last.pt", step)
    best_actor = load(ckpt / "best.pt", dev)
    returns, extra = eval_row(validate(best_actor, eval_envs("test", cfg.eval_levels, dev)))
    log.final(best_step, returns, **extra)
    log.close()
    return run_dir


if __name__ == "__main__":
    train(parse_args(Config))
