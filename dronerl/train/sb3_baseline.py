"""Stable-Baselines3 reference runs evaluated with the same protocol as our own agents."""

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from dronerl.eval.evaluate import VAL_SEED, evaluate, score
from dronerl.utils.logger import Logger
from dronerl.utils.run import make_run_dir, parse_args, set_seed

ALGOS = {"dqn": DQN, "ppo": PPO, "sac": SAC}


@dataclass
class Config:
    algo: str = "ppo"
    env_id: str = "Pendulum-v1"
    env_kwargs: dict = field(default_factory=dict)
    seed: int = 0
    exp: str = "default"
    variant: str = "sb3"
    log_dir: str = "runs"
    total_steps: int = 100_000
    n_envs: int = 1
    # environments in worker processes, pays off when stepping the simulation dominates
    subproc: bool = False
    normalize: bool = False
    # Passed straight to the SB3 constructor, e.g. values from rl-baselines3-zoo
    params: dict = field(default_factory=dict)
    eval_every: int = 5_000
    eval_episodes: int = 20
    final_eval_episodes: int = 100


def policy(model, norm):
    def act(obs):
        if norm is not None:
            obs = norm.normalize_obs(obs)
        return model.predict(obs, deterministic=True)[0]

    return act


class EvalCallback(BaseCallback):
    """Evaluates every `eval_every` environment steps and keeps the best model."""

    def __init__(self, cfg, log, ckpt):
        super().__init__()
        self.cfg, self.log, self.ckpt = cfg, log, ckpt
        self.env = gym.make(cfg.env_id, **cfg.env_kwargs)
        self.next_eval, self.last_eval, self.best, self.best_step = cfg.eval_every, 0, (-np.inf,), 0

    def evaluate(self):
        norm = self.model.get_vec_normalize_env()
        act = policy(self.model, norm)
        returns, extra = evaluate(self.env, act, self.cfg.eval_episodes, VAL_SEED)
        row = self.log.eval(self.num_timesteps, returns, **extra)
        self.last_eval = self.num_timesteps
        if score(row) >= self.best:
            self.best, self.best_step = score(row), self.num_timesteps
            self.model.save(self.ckpt / "best.zip")
            if norm is not None:
                norm.save(self.ckpt / "best_vecnormalize.pkl")

    def _on_step(self):
        if self.num_timesteps >= self.next_eval:
            while self.next_eval <= self.num_timesteps:
                self.next_eval += self.cfg.eval_every
            self.evaluate()
        return True

    def _on_training_end(self):
        if self.last_eval < self.num_timesteps:
            self.evaluate()


def train(cfg: Config):
    set_seed(cfg.seed)
    run_dir = make_run_dir(cfg)
    ckpt = run_dir / "checkpoints"
    log = Logger(run_dir)
    vec_cls = SubprocVecEnv if cfg.subproc else DummyVecEnv
    env = make_vec_env(
        cfg.env_id, cfg.n_envs, cfg.seed, env_kwargs=cfg.env_kwargs, vec_env_cls=vec_cls
    )
    if cfg.normalize:
        env = VecNormalize(env, gamma=cfg.params.get("gamma", 0.99))
    algo = ALGOS[cfg.algo]
    tb = str(run_dir / "tb")
    model = algo("MlpPolicy", env, seed=cfg.seed, device="cpu", tensorboard_log=tb, **cfg.params)
    callback = EvalCallback(cfg, log, ckpt)
    model.learn(cfg.total_steps, callback=callback, tb_log_name="sb3")
    model.save(ckpt / "last.zip")
    if cfg.normalize:
        env.save(str(ckpt / "last_vecnormalize.pkl"))

    best = algo.load(ckpt / "best.zip", device="cpu")
    norm = None
    if cfg.normalize:
        dummy = make_vec_env(cfg.env_id, env_kwargs=cfg.env_kwargs)
        norm = VecNormalize.load(str(ckpt / "best_vecnormalize.pkl"), dummy)
    returns, extra = evaluate(callback.env, policy(best, norm), cfg.final_eval_episodes)
    log.final(callback.best_step, returns, **extra)
    log.close()
    env.close()
    return run_dir


if __name__ == "__main__":
    train(parse_args(Config))
