"""PPO with an asymmetric critic (M3.11): the actor is the student (3 frames, the student's
observations), the critic gets the privileged observations of the teacher.

The training loop is the one of ppo_gpu.py. Checkpoints keep the actor under the student's keys,
so nav_eval.py and dagger.py load them as a student.
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.distributions import Normal

from dronerl import contract
from dronerl.agents import ppo_gpu
from dronerl.agents.ppo import RunningMeanStd
from dronerl.agents.student import Policy, Student, from_uint8, to_uint8
from dronerl.envs.nav_gpu import STUDENT
from dronerl.utils.run import parse_args


@dataclass
class Config(ppo_gpu.Config):
    variant: str = "asym"
    sensors: dict | None = field(default_factory=lambda: dict(STUDENT))
    eval_condition: str | None = "test"


class AsymAgent(nn.Module):
    def __init__(self, init_std=0.5):
        super().__init__()
        self.vec_rms = RunningMeanStd(ppo_gpu.VEC_DIM)
        self.actor = Student("stack3")
        self.critic = ppo_gpu.Head(1, 1.0)
        self.log_std = nn.Parameter(torch.full((contract.ACT_DIM,), math.log(init_std)))
        self.frames = None  # input frames of the actor in the training envs

    vec = ppo_gpu.Agent.vec
    value = ppo_gpu.Agent.value
    value_of = ppo_gpu.Agent.value_of

    def inputs(self, obs, update=False, start=None):
        self.frames = self.actor.stack(obs["student_depth"], self.frames, start)
        n, h, w = len(self.frames), contract.IMG_H, contract.IMG_W
        return {
            "image": to_uint8(self.frames.view(n, 3, contract.STUDENT_CHANNELS, h, w)),
            "state": self.actor.norm(obs["student_state"], update),
            "depth": obs["depth"].half(),
            "vec": self.vec(obs, update),
        }

    def dist(self, x):
        mean = self.actor(from_uint8(x["image"]).flatten(1, 2), x["state"])[0]
        return Normal(mean, self.log_std.exp().expand_as(mean))

    def policy(self):
        return Policy(self.actor)

    def save(self, path, step):
        ckpt = {"student": self.actor.state_dict(), "history": "stack3", "agent": self.state_dict()}
        torch.save({**ckpt, "step": step, "version": contract.VERSION}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device)
        agent = cls().to(device)
        agent.load_state_dict(ckpt["agent"])
        return agent.eval()


def train(cfg: Config, on_eval=None):
    return ppo_gpu.train(cfg, on_eval, AsymAgent(cfg.init_std))


if __name__ == "__main__":
    train(parse_args(Config))
