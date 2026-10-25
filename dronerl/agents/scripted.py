import math

import numpy as np
import torch

from dronerl import contract


class PController:
    """Velocity command v = clip(k_p * (goal - pos), v_max), as a normalized `vel` action."""

    def __init__(self, k_p=1.0, v_max=1.0):
        self.k_p = k_p
        self.v_max = v_max

    def __call__(self, obs):
        v = self.k_p * np.asarray(obs[:3], dtype=np.float64)
        speed = np.linalg.norm(v)
        if speed > self.v_max:
            v *= self.v_max / speed
        return (v / self.v_max).astype(np.float32)


def wrap(angle):
    return torch.atan2(angle.sin(), angle.cos())


class GoalP:
    """Flies straight at the goal from the state vector alone, blind to obstacles (M2.4).

    Velocity proportional to the goal offset in the heading frame, yaw rate turning to the goal.
    """

    def __init__(self, k_p=1.0, k_yaw=2.0):
        self.k_p, self.k_yaw = k_p, k_yaw

    def __call__(self, obs):
        s = obs["state"]
        bearing, dist = torch.atan2(s[:, 5], s[:, 4]), s[:, 6].expm1()
        cmd = torch.stack(
            [self.k_p * dist * s[:, 4], self.k_p * dist * s[:, 5], self.k_yaw * bearing], -1
        )
        return contract.command_to_action(cmd)


class PurePursuit:
    """Follows the A* path of each env's map with full knowledge of the map (M2.4).

    Steers towards the path point `lookahead` metres ahead of the drone's progress. Forward
    speed falls from v_max to 0 as the heading error grows to `max_err`: turning while moving
    at full speed drove it into obstacles next to the start (plan/M2.md, M2.4).
    """

    def __init__(self, env, paths, lookahead=1.0, k_yaw=4.0, max_err=0.5, v_max=1.0):
        self.env, self.lookahead, self.k_yaw, self.v_max = env, lookahead, k_yaw, v_max
        self.min_cos = math.cos(max_err)
        self.paths = torch.as_tensor(paths, dtype=torch.float32, device=env.device)
        self.seg = self.paths[:, 1:] - self.paths[:, :-1]
        self.seg_len = self.seg.norm(dim=-1)  # padding after the goal has zero length
        cum = self.seg_len.cumsum(-1)
        self.cum = torch.cat([torch.zeros_like(cum[:, :1]), cum], -1)
        self.progress = torch.zeros(env.num_envs, device=env.device)

    def __call__(self, obs):
        env, rows = self.env, self.env.rows
        m = env.map_idx
        start, seg, length, cum = self.paths[m, :-1], self.seg[m], self.seg_len[m], self.cum[m]
        t = ((env.pos[:, None] - start) * seg).sum(-1) / length.clamp(min=1e-9) ** 2
        t = t.clamp(0, 1)
        gap = (env.pos[:, None] - start - t[..., None] * seg).norm(dim=-1)
        arc = cum[:, :-1] + t * length
        # the nearest point not far behind the progress, so a path passing close to itself
        # cannot pull the follower back
        progress = torch.where(env.steps == 0, 0.0, self.progress)
        gap = torch.where((arc >= progress[:, None] - 0.5) & (length > 0), gap, math.inf)
        k = gap.argmin(-1)
        self.progress = torch.maximum(progress, arc[rows, k])
        ahead = torch.minimum(self.progress + self.lookahead, cum[:, -1])
        j = (cum[:, 1:-1] < ahead[:, None]).sum(-1)
        frac = ((ahead - cum[rows, j]) / length[rows, j].clamp(min=1e-9)).clamp(0, 1)
        to = start[rows, j] + frac[:, None] * seg[rows, j] - env.pos
        err = wrap(torch.atan2(to[:, 1], to[:, 0]) - env.yaw)
        fwd = self.v_max * ((err.cos() - self.min_cos) / (1 - self.min_cos)).clamp(0, 1)
        yaw_rate = (self.k_yaw * err).clamp(-contract.YAW_RATE_MAX, contract.YAW_RATE_MAX)
        return contract.command_to_action(torch.stack([fwd, torch.zeros_like(fwd), yaw_rate], -1))
