"""DAgger for the student (M3.9): the teacher of M2.7 labels the states the student visits.

In iteration i the teacher's action is executed with probability beta^i per step, the label is
always the teacher's mean action clamped to [-1, 1]. Samples go to a FIFO buffer in RAM and the
student fits them by MSE. With history `gru` a buffer item is a segment of num_steps steps with
the hidden state of the rollout at its start.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from dronerl import contract
from dronerl.agents import ppo_gpu
from dronerl.agents.student import FRAMES, GRU, Policy, Student, from_uint8, to_uint8
from dronerl.envs import maps as mapgen
from dronerl.envs.nav_gpu import STUDENT, NavEnv, load_maps
from dronerl.eval.evaluate import score
from dronerl.utils.logger import Logger
from dronerl.utils.run import make_run_dir, parse_args, set_seed

TEACHER = "runs/m2/teacher_rand/curriculum_rand_nav-v1_s1_20261026-170647/checkpoints/best.pt"


@dataclass
class Config:
    env_id: str = f"nav-v{contract.VERSION}"
    seed: int = 0
    exp: str = "default"
    variant: str = "dagger"
    log_dir: str = "runs"
    device: str = "cuda"
    teacher: str = TEACHER
    history: str = "stack3"
    total_steps: int = 20_000_000
    num_envs: int = 2048
    num_steps: int = 32
    beta: float = 0.5
    buffer_size: int = 1_000_000  # samples
    batch_size: int = 4096
    updates: int = 32  # minibatches per iteration
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0
    # levels as at the end of the teacher's curriculum; labels need no reward, so no promotions
    level: int = 5
    mix: float = 0.2
    randomize: bool = True
    sensors: dict = field(default_factory=lambda: dict(STUDENT))
    eval_every: int = 2_000_000
    eval_levels: dict = field(default_factory=lambda: dict(mapgen.TEST_LEVELS))


def beta_schedule(beta, iteration):
    return beta**iteration


class Buffer:
    """FIFO of the latest items in RAM; an item is a sample, or a segment for the GRU."""

    def __init__(self, capacity, shapes):
        self.data = {k: torch.empty(capacity, *s, dtype=dt) for k, (s, dt) in shapes.items()}
        self.capacity, self.size, self.pos = capacity, 0, 0

    def add(self, items):
        n = len(next(iter(items.values())))
        idx = (self.pos + torch.arange(n)) % self.capacity
        for k, v in items.items():
            self.data[k][idx] = v.cpu()
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, n, gen, device):
        idx = torch.randint(self.size, (n,), generator=gen)
        return {k: v[idx].to(device) for k, v in self.data.items()}


def fit(student, batch, history):
    """MSE of the student's action to the teacher's label on a minibatch from the buffer."""
    image = from_uint8(batch["image"])
    state = student.norm(batch["state"])
    if history == "gru":
        # segments [S, T, ...], the GRU runs time first
        image, state, start = image.transpose(0, 1), state.transpose(0, 1), batch["start"].T
        out = student.unroll(image, state, start, batch["h0"])
        return ((out - batch["label"].transpose(0, 1)) ** 2).mean()
    out, _ = student(image.flatten(1, 2), state)
    return ((out - batch["label"]) ** 2).mean()


def train(cfg: Config):
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
        mix=cfg.mix,
        randomize=cfg.randomize,
        student=cfg.sensors,
    )
    val = ppo_gpu.eval_envs("val", cfg.eval_levels, dev, "test")
    teacher = ppo_gpu.Agent.load(cfg.teacher, dev)
    student = Student(cfg.history).to(dev)
    optimizer = torch.optim.Adam(student.parameters(), lr=cfg.learning_rate, eps=1e-5)

    T, N, H, W = cfg.num_steps, cfg.num_envs, contract.IMG_H, contract.IMG_W
    k, gru = FRAMES[cfg.history], cfg.history == "gru"
    if gru:
        shapes = {
            "image": ((T, H, W), torch.uint8),
            "state": ((T, contract.STATE_DIM), torch.float32),
            "label": ((T, contract.ACT_DIM), torch.float32),
            "start": ((T,), torch.bool),
            "h0": ((GRU,), torch.float32),
        }
        buffer, batch = Buffer(cfg.buffer_size // T, shapes), cfg.batch_size // T
    else:
        shapes = {
            "image": ((k, H, W), torch.uint8),
            "state": ((contract.STATE_DIM,), torch.float32),
            "label": ((contract.ACT_DIM,), torch.float32),
        }
        buffer, batch = Buffer(cfg.buffer_size, shapes), cfg.batch_size
    sampler = torch.Generator().manual_seed(cfg.seed)

    teacher_val = ppo_gpu.eval_row(ppo_gpu.validate(teacher.policy, val))[1]
    log.tb.add_text("teacher_val", str(teacher_val))
    obs = env.reset()
    memory, start = None, torch.ones(N, dtype=torch.bool, device=dev)
    step, next_eval, best, best_step = 0, cfg.eval_every, (-np.inf,), 0
    iterations = cfg.total_steps // (T * N)
    begin = time.time()

    for it in range(iterations):
        beta = beta_schedule(cfg.beta, it)
        roll = {"image": [], "state": [], "label": [], "start": []}
        h0 = torch.zeros(N, GRU, device=dev) if memory is None or not gru else memory
        done_count, success_count, collision_count = 0, 0, 0
        t0 = time.time()
        for _ in range(T):
            with torch.no_grad():
                label = teacher.act(obs).clamp(-1, 1)
                action, memory, frames = student.step(obs, memory, start, update=True)
            roll["image"].append(to_uint8(frames.view(N, k, contract.STUDENT_CHANNELS, H, W)))
            roll["state"].append(obs["student_state"])
            roll["label"].append(label)
            roll["start"].append(start)
            expert = torch.rand(N, device=dev) < beta
            obs, _, term, trunc, info = env.step(torch.where(expert[:, None], label, action))
            start = term | trunc
            step += N
            done_count += start.sum()
            success_count += info["success"].sum()
            collision_count += info["collision"].sum()
        if gru:
            items = {key: torch.stack(v, 1) for key, v in roll.items()}
            items["image"], items["h0"] = items["image"][:, :, 0], h0
        else:
            items = {key: torch.cat(v) for key, v in roll.items() if key != "start"}
        buffer.add(items)
        t1 = time.time()

        losses = []
        for _ in range(cfg.updates):
            loss = fit(student, buffer.sample(batch, sampler, dev), cfg.history)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), cfg.max_grad_norm)
            optimizer.step()
            losses.append(loss.detach())
        ppo_gpu.sync(dev)
        t2 = time.time()

        episodes = done_count.item()
        if episodes:
            log.scalar("train/success_rate", success_count.item() / episodes, step)
            log.scalar("train/collision_rate", collision_count.item() / episodes, step)
        log.scalar("train/loss", torch.stack(losses).mean().item(), step)
        log.scalar("train/beta", beta, step)
        log.scalar("train/buffer", buffer.size, step)
        log.scalar("train/sps", step / (time.time() - begin), step)
        log.scalar("time/rollout_s", t1 - t0, step)
        log.scalar("time/update_s", t2 - t1, step)

        if step >= next_eval or it == iterations - 1:
            while next_eval <= step:
                next_eval += cfg.eval_every
            results = ppo_gpu.validate(lambda: Policy(student.eval()), val)
            returns, extra = ppo_gpu.eval_row(results)
            student.train()
            row = log.eval(step, returns, **extra)
            if score(row) >= best:
                best, best_step = score(row), step
                student.save(ckpt / "best.pt", step)

    student.save(ckpt / "last.pt", step)
    best_student = Student.load(ckpt / "best.pt", dev)
    test = ppo_gpu.eval_envs("test", cfg.eval_levels, dev, "test")
    returns, extra = ppo_gpu.eval_row(ppo_gpu.validate(lambda: Policy(best_student), test))
    teacher_test = ppo_gpu.eval_row(ppo_gpu.validate(teacher.policy, test))[1]
    extra.update({f"teacher_{key}": value for key, value in teacher_test.items()})
    log.final(best_step, returns, **extra)
    log.close()
    return run_dir


if __name__ == "__main__":
    train(parse_args(Config))
