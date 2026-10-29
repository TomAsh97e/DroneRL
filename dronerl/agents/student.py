"""Student policy (M3.9): what the drone can run, the depth image with a validity mask and a noisy
state, no lidar. Trained by DAgger (train/dagger.py) or PPO with an asymmetric critic (M3.11).

History: `none`, `stack3` (the last 3 frames as channels) or `gru` (GRU 128 on the features of
one frame).
"""

import torch
import torch.nn as nn

from dronerl import contract
from dronerl.agents.ppo import RunningMeanStd, layer
from dronerl.agents.ppo_gpu import Encoder

FRAMES = {"none": 1, "stack3": 3, "gru": 1}
GRU = 128


def to_uint8(image):
    """Student image [..., 2, H, W] -> one uint8 channel [..., H, W], 0 marks missing data."""
    depth, valid = image.unbind(-3)
    return torch.where(valid > 0, 1 + (depth * 254).round(), 0).to(torch.uint8)


def from_uint8(x):
    valid = x > 0
    depth = (x.float() - 1).clamp(min=0) / 254
    return torch.stack([depth, valid.float()], -3)


class Student(nn.Module):
    def __init__(self, history="stack3"):
        super().__init__()
        self.history = history
        self.state_rms = RunningMeanStd(contract.STATE_DIM)
        self.encoder = Encoder(FRAMES[history] * contract.STUDENT_CHANNELS)
        features = 128 + contract.STATE_DIM
        self.gru = nn.GRUCell(features, GRU) if history == "gru" else None
        self.mlp = nn.Sequential(
            layer(GRU if self.gru else features, 256),
            nn.ELU(),
            layer(256, 256),
            nn.ELU(),
            layer(256, contract.ACT_DIM, 0.01),
        )

    def norm(self, state, update=False):
        if update:
            self.state_rms.update(state)
        z = (state - self.state_rms.mean) / torch.sqrt(self.state_rms.var + 1e-8)
        return z.clamp(-10, 10).float()

    def forward(self, image, state, h=None):
        """One step: input frames [N, C, H, W] and normalized state [N, 10] -> (action, hidden)."""
        x = torch.cat([self.encoder(image), state], -1)
        if self.gru is not None:
            h = x = self.gru(x, h)
        return self.mlp(x), h

    def unroll(self, images, states, starts, h):
        """GRU over a segment [T, N, ...] from the hidden state h of the rollout, reset at the
        starts of episodes (truncated backpropagation through time)."""
        T, N = starts.shape
        x = torch.cat([self.encoder(images.flatten(0, 1)).view(T, N, -1), states], -1)
        out = []
        for t in range(T):
            h = self.gru(x[t], h * ~starts[t, :, None])
            out.append(h)
        return self.mlp(torch.stack(out))

    def stack(self, image, frames, start):
        """Input frames after a new image [N, 2, H, W], frames are the previous input (None at the
        start of all episodes), start [N] marks the first frames of new episodes."""
        if self.history != "stack3":
            return image
        # a new episode has no older frames yet, its first one is repeated
        first = image.repeat(1, 3, 1, 1)
        if frames is None:
            return first
        return torch.where(start[:, None, None, None], first, torch.cat([frames[:, 2:], image], 1))

    def step(self, obs, memory=None, start=None, update=False):
        """Action for observations of the student. memory is what the previous step returned,
        None at the start of all episodes, start [N] marks the first frames of new episodes.
        Returns (action, memory, input frames)."""
        image, state = obs["student_depth"], self.norm(obs["student_state"], update)
        if start is None:
            start = torch.zeros(len(image), dtype=torch.bool, device=image.device)
        if self.history == "gru":
            h = torch.zeros(len(image), GRU, device=image.device) if memory is None else memory
            action, h = self(image, state, h * ~start[:, None])
            return action, h, image
        frames = self.stack(image, memory, start)
        return self(frames, state)[0], frames, frames

    def save(self, path, step):
        ckpt = {"student": self.state_dict(), "history": self.history, "step": step}
        torch.save({**ckpt, "version": contract.VERSION}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device)
        student = cls(ckpt["history"]).to(device)
        student.load_state_dict(ckpt["student"])
        return student.eval()


class Policy:
    """act(obs) of a student for a batch of episodes that all start at the first call."""

    def __init__(self, student):
        self.student, self.memory = student, None

    @torch.no_grad()
    def __call__(self, obs):
        action, self.memory, _ = self.student.step(obs, self.memory)
        return action
