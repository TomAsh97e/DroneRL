"""Vectorized navigation environment in PyTorch (M2.3).

Velocity-level dynamics at constant height, depth and a virtual lidar by ray casting, the reward
of the contract and a curriculum over the obstacle density of the map levels.
"""

import math
from collections import deque

import torch

from dronerl import contract
from dronerl.envs import maps as mapgen
from dronerl.envs import raycast

SUBSTEPS = 5
SUB_DT = contract.DT / SUBSTEPS
NOMINAL = {"tau_v": 0.3, "tau_yaw": 0.15, "accel": 3.0, "delay": 0.04, "height": 1.2}
RANGES = {
    "tau_v": (0.15, 0.6),
    "tau_yaw": (0.05, 0.3),
    "accel": (1.5, 4.0),
    "delay": (0.0, 0.12),
    "height": (1.0, 1.5),
}
MAX_DELAY = round(RANGES["delay"][1] / SUB_DT)  # in substeps
PROMOTE_SR = 0.8
PROMOTE_WINDOW = 1000
CHUNK = 4096  # envs per ray casting batch, bounds the memory of the [E, R, B] intermediates
OBSTACLES = ("circles", "circle_mask", "boxes", "box_mask")


def load_maps(split, levels=None):
    levels = mapgen.levels(split) if levels is None else levels
    return mapgen.concat([mapgen.load(split, level) for level in levels])


class NavEnv:
    """A batch of navigation episodes stepped together.

    Observations are dicts of tensors: depth [E, 1, 48, 64], state [E, 10], lidar [E, 72].
    With autoreset (training) finished episodes restart within the same step, and
    info["final_obs"] holds the last observation of the truncated ones (rows info["final_idx"]).
    Without it (evaluation) env i plays map i once and then stays frozen.
    """

    def __init__(
        self,
        maps,
        num_envs=None,
        device="cuda",
        seed=0,
        level=0,
        curriculum=False,
        mix=0.2,
        randomize=False,
        prox_w=contract.PROXIMITY_W,
        autoreset=True,
    ):
        dev = self.device = torch.device(device)
        self.num_envs = E = len(maps["start"]) if num_envs is None else num_envs
        assert autoreset or E == len(maps["start"]), "evaluation plays every map once"
        self.level, self.curriculum, self.mix = level, curriculum, mix
        self.randomize, self.prox_w, self.autoreset = randomize, prox_w, autoreset
        self.window = deque(maxlen=PROMOTE_WINDOW)
        self.gen = torch.Generator(dev).manual_seed(seed)

        def f(x):
            return torch.as_tensor(x, dtype=torch.float32, device=dev)

        def mask(key):
            n = torch.as_tensor(maps[f"n_{key}"], device=dev)
            return torch.arange(maps[key].shape[1], device=dev) < n[:, None]

        self.maps = {
            "circles": f(maps["circles"]),
            "circle_mask": mask("circles"),
            "boxes": raycast.boxes_from_maps(f(maps["boxes"])),
            "box_mask": mask("boxes"),
            "goal": f(maps["goal"]),
            "size": f(maps["size"]),
            "obstacle_height": f(maps["height"]),
        }
        self.starts, self.yaw0 = f(maps["start"]), f(maps["yaw0"])
        self.map_level = torch.as_tensor(maps["level"], device=dev).long()
        self.order = self.map_level.argsort(stable=True)
        self.counts = torch.bincount(self.map_level, minlength=len(mapgen.DENSITY))
        self.offsets = self.counts.cumsum(0) - self.counts

        # each env keeps a copy of its current map, refreshed on reset
        self.env = {k: v.new_zeros((E, *v.shape[1:])) for k, v in self.maps.items()}
        self.ray_angles = torch.cat([raycast.COLUMN_ANGLES, raycast.LIDAR_ANGLES]).to(dev)
        self.rows = torch.arange(E, device=dev)
        vectors = {"pos": 2, "vel": 2, "target": 2, "prev_action": contract.ACT_DIM}
        for name, dim in vectors.items():
            setattr(self, name, torch.zeros(E, dim, device=dev))
        floats = ["yaw", "yaw_rate", "yaw_target", "dist", "ep_return", "path_len", "min_clear"]
        floats += ["alpha_v", "alpha_yaw", "accel", "height"]
        for name in floats:
            setattr(self, name, torch.zeros(E, device=dev))
        for name in ("steps", "env_level", "map_idx", "delay"):
            setattr(self, name, torch.zeros(E, dtype=torch.long, device=dev))
        self.hist = torch.zeros(E, MAX_DELAY // SUBSTEPS + 1, 3, device=dev)
        self.alive = torch.ones(E, dtype=torch.bool, device=dev)

    def rand(self, n):
        return torch.rand(n, generator=self.gen, device=self.device)

    def obstacles(self, rows=None):
        rows = slice(None) if rows is None else rows
        return tuple(self.env[k][rows] for k in OBSTACLES)

    def sample_levels(self, n):
        levels = torch.full((n,), self.level, device=self.device)
        if self.mix and self.level > 0:
            lower = (self.rand(n) * self.level).long()
            levels = torch.where(self.rand(n) < self.mix, lower, levels)
        return levels

    def params(self, n):
        if not self.randomize:
            return {k: torch.full((n,), v, device=self.device) for k, v in NOMINAL.items()}
        return {k: lo + (hi - lo) * self.rand(n) for k, (lo, hi) in RANGES.items()}

    def reset(self):
        self.alive[:] = True
        self.reset_envs(self.rows)
        return self.observe()

    def reset_envs(self, ids):
        n = len(ids)
        if self.autoreset:
            levels = self.sample_levels(n)
            assert (self.counts[levels] > 0).all(), "no maps for a sampled level"
            m = self.order[self.offsets[levels] + (self.rand(n) * self.counts[levels]).long()]
            to_goal = self.maps["goal"][m] - self.starts[m]
            yaw = torch.atan2(to_goal[:, 1], to_goal[:, 0]) + (self.rand(n) - 0.5) * math.pi
        else:
            m, yaw = ids, self.yaw0[ids]
        self.map_idx[ids], self.env_level[ids] = m, self.map_level[m]
        for key, value in self.maps.items():
            self.env[key][ids] = value[m]
        self.pos[ids], self.yaw[ids] = self.starts[m], yaw
        for x in (self.vel, self.target, self.yaw_rate, self.yaw_target, self.hist):
            x[ids] = 0
        for x in (self.prev_action, self.steps, self.ep_return, self.path_len):
            x[ids] = 0
        self.min_clear[ids] = math.inf
        self.dist[ids] = (self.env["goal"][ids] - self.pos[ids]).norm(dim=-1)
        p = self.params(n)
        self.alpha_v[ids] = 1 - torch.exp(-SUB_DT / p["tau_v"])
        self.alpha_yaw[ids] = 1 - torch.exp(-SUB_DT / p["tau_yaw"])
        self.accel[ids], self.height[ids] = p["accel"], p["height"]
        self.delay[ids] = (p["delay"] / SUB_DT).round().long()

    def observe(self, ids=None):
        rows = self.rows if ids is None else ids
        out = {"depth": [], "state": [], "lidar": []}
        for chunk in rows.split(CHUNK):
            pos, yaw, height = self.pos[chunk], self.yaw[chunk], self.height[chunk]
            hits = raycast.cast(pos, yaw[:, None] + self.ray_angles, self.obstacles(chunk))
            columns, beams = hits[:, : contract.IMG_W], hits[:, contract.IMG_W :]
            depth = raycast.depth_image(columns, height, self.env["obstacle_height"][chunk])
            out["depth"].append(contract.encode_depth(depth)[:, None])
            out["lidar"].append(contract.encode_depth(beams))
            goal = self.env["goal"][chunk] - pos
            vel, rate, prev = self.vel[chunk], self.yaw_rate[chunk], self.prev_action[chunk]
            out["state"].append(contract.state_vector(vel, yaw, rate, height, goal, prev))
        return {k: torch.cat(v) for k, v in out.items()}

    def step(self, action):
        a = action.clamp(-1, 1)
        alive = self.alive.clone()
        self.hist = torch.cat([contract.action_to_command(a)[:, None], self.hist[:, :-1]], 1)
        # the command issued `lag` steps ago reaches the autopilot at substep `phase` of this step
        lag, phase = self.delay // SUBSTEPS, self.delay % SUBSTEPS
        cmd = self.hist[self.rows, lag]
        moving = alive.clone()
        success, collision, out = (torch.zeros_like(alive) for _ in range(3))
        for j in range(SUBSTEPS):
            # as in the bridge to PX4, a heading-frame command becomes a world-frame setpoint
            # with the yaw of the moment it arrives
            new = phase == j
            c, s = self.yaw.cos(), self.yaw.sin()
            world = torch.stack([c * cmd[:, 0] - s * cmd[:, 1], s * cmd[:, 0] + c * cmd[:, 1]], -1)
            self.target = torch.where(new[:, None], world, self.target)
            self.yaw_target = torch.where(new, cmd[:, 2], self.yaw_target)

            dv = (self.target - self.vel) * self.alpha_v[:, None]
            limit = self.accel[:, None] * SUB_DT / dv.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            vel = self.vel + dv * limit.clamp(max=1)
            self.vel = torch.where(moving[:, None], vel, self.vel)
            self.pos = torch.where(moving[:, None], self.pos + vel * SUB_DT, self.pos)
            self.path_len += moving * vel.norm(dim=-1) * SUB_DT
            rate = self.yaw_rate + (self.yaw_target - self.yaw_rate) * self.alpha_yaw
            self.yaw_rate = torch.where(moving, rate, self.yaw_rate)
            self.yaw = torch.where(moving, self.yaw + rate * SUB_DT, self.yaw)

            clear = raycast.distance(self.pos, self.obstacles())
            dist = (self.env["goal"] - self.pos).norm(dim=-1)
            outside = (self.pos < 0).any(-1) | (self.pos > self.env["size"]).any(-1)
            hit = moving & (clear < contract.COLLISION_DIST)
            reached = moving & ~hit & (dist < contract.SUCCESS_RADIUS)
            left = moving & ~hit & ~reached & outside
            collision |= hit
            success |= reached
            out |= left
            moving &= ~(hit | reached | left)

        self.yaw = torch.atan2(self.yaw.sin(), self.yaw.cos())
        progress, self.dist = self.dist - dist, dist
        change = ((a - self.prev_action) ** 2).sum(-1)
        self.prev_action = torch.where(alive[:, None], a, self.prev_action)
        margin = clear - contract.COLLISION_DIST
        reward = contract.reward(progress, success, collision, change, margin, self.prox_w)
        reward = reward * alive
        self.steps += alive
        self.ep_return += reward
        self.min_clear = torch.where(alive, torch.minimum(self.min_clear, clear), self.min_clear)
        terminated = success | collision | out
        truncated = alive & ~terminated & (self.steps >= contract.EPISODE_STEPS)
        done = terminated | truncated
        info = {
            "success": success,
            "collision": collision,
            "out_of_area": out,
            "timeout": truncated,
            "length": self.steps.clone(),
            "return": self.ep_return.clone(),
            "level": self.env_level.clone(),
            "path_length": self.path_len.clone(),
            "min_clearance": self.min_clear - contract.COLLISION_DIST,
        }
        obs = self.observe()
        if not self.autoreset:
            self.alive &= ~done
            return obs, reward, terminated, truncated, info
        ids = done.nonzero().squeeze(-1)
        if len(ids):
            self.track(info["level"][ids], success[ids])
            final = truncated.nonzero().squeeze(-1)
            info["final_idx"], info["final_obs"] = final, {k: v[final] for k, v in obs.items()}
            self.reset_envs(ids)
            for key, value in self.observe(ids).items():
                obs[key][ids] = value
        return obs, reward, terminated, truncated, info

    def track(self, levels, success):
        """Curriculum: promotion after PROMOTE_SR over the last episodes on the current level."""
        if not self.curriculum or self.level == len(mapgen.DENSITY) - 1:
            return
        self.window.extend(success[levels == self.level].tolist())
        if len(self.window) == PROMOTE_WINDOW and sum(self.window) >= PROMOTE_SR * PROMOTE_WINDOW:
            self.level += 1
            self.window.clear()
