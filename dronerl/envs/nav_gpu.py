"""Vectorized navigation environment in PyTorch (M2.3).

Velocity-level dynamics at constant height, depth and a virtual lidar by ray casting, the reward
of the contract and a curriculum over the obstacle density of the map levels. With `student`
(M3.8) it also renders what the student will get on the drone: a tilted, noisy and delayed depth
image and a noisy state with odometry drift.
"""

import math
from collections import deque

import torch

from dronerl import contract
from dronerl.envs import maps as mapgen
from dronerl.envs import raycast
from dronerl.envs.depth_noise import DepthNoise

SUBSTEPS = 5
SUB_DT = contract.DT / SUBSTEPS
NOMINAL = {"tau_v": 0.3, "tau_yaw": 0.15, "accel": 3.0, "delay": 0.04, "height": 1.2}
RANGES = {
    "tau_v": (0.15, 0.6),
    "tau_yaw": (0.05, 0.3),
    "accel": (1.5, 4.0),
    "delay": (0.0, 0.3),  # PX4 in SITL answers a velocity step 0.20-0.26 s late (M4.9)
    "height": (1.0, 1.5),
}
MAX_DELAY = round(RANGES["delay"][1] / SUB_DT)  # in substeps
PROMOTE_SR = 0.8
PROMOTE_WINDOW = 1000
CHUNK = 4096  # envs per ray casting batch, bounds the memory of the [E, R, B] intermediates
OBSTACLES = ("circles", "circle_mask", "boxes", "box_mask")

# sensors of the student; evaluation conditions also fix the dynamics: randomized in the test
# ones, nominal in `clean`. The Gazebo profile takes half of the mono share, the error of the model
# in a new domain is not the one of NYU (M4.12); the active share keeps a depth camera an option
STUDENT = {
    "mix": {"mono": 0.35, "gazebo": 0.35, "active": 0.3},
    "state_noise": True,
    "tilt": True,
    "latency": True,
}
CONDITIONS = {
    "test": {**STUDENT, "mix": {"mono": 1.0}},
    "test_active": {**STUDENT, "mix": {"active": 1.0}},
    "clean": {"mix": {"none": 1.0}, "state_noise": False, "tilt": False, "latency": False},
}
G = 9.81
TILT_TAU = (0.05, 0.15)  # lag of the attitude behind the acceleration [s]
MOUNT = (math.radians(5), math.radians(2))  # camera mount error, pitch and roll
VIBRATION = math.radians(0.5)
# white noise std and bias range of v_fwd, v_lat, yaw_rate and height, roughly the PX4 EKF with
# optical flow and a rangefinder
STATE_NOISE = (0.05, 0.05, 0.02, 0.03)
STATE_BIAS = (0.05, 0.05, 0.01, 0.05)
# odometry position error per metre of path; EKF2 with flow and a magnetometer in SITL turns its
# frame by 6-7°, about 10% of the path (M4.11)
DRIFT = (0.01, 0.1)


def load_maps(split, levels=None):
    levels = mapgen.levels(split) if levels is None else levels
    return mapgen.concat([mapgen.load(split, level) for level in levels])


class NavEnv:
    """A batch of navigation episodes stepped together.

    Observations are dicts of tensors: depth [E, 1, 48, 64], state [E, 10], lidar [E, 72], and with
    `student` (a dict like CONDITIONS) also student_depth [E, 2, 48, 64] and student_state [E, 10].
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
        student=None,
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

        self.student = student
        if student:
            self.noise = DepthNoise(E, self.gen, student["mix"])
            self.ray_angles = torch.cat([raycast.FAN_ANGLES, raycast.LIDAR_ANGLES]).to(dev)
            for name in ("tilt", "mount", "odometry"):  # (pitch, roll) and (x, y)
                setattr(self, name, torch.zeros(E, 2, device=dev))
            for name in ("alpha_tilt", "drift", "drift_angle"):
                setattr(self, name, torch.zeros(E, device=dev))
            self.state_bias = torch.zeros(E, 4, device=dev)
            self.state_noise = torch.tensor(STATE_NOISE, device=dev)
            self.lag = torch.zeros(E, dtype=torch.bool, device=dev)
            self.fresh = torch.ones(E, dtype=torch.bool, device=dev)
            shape = (E, contract.STUDENT_CHANNELS, contract.IMG_H, contract.IMG_W)
            self.last_image = torch.zeros(shape, device=dev)

    def rand(self, n):
        return torch.rand(n, generator=self.gen, device=self.device)

    def randn(self, n):
        return torch.randn(n, generator=self.gen, device=self.device)

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
        if self.student:
            self.reset_student(ids)

    def reset_student(self, ids):
        n, s = len(ids), self.student
        self.noise.reset(ids)
        self.tilt[ids] = 0
        tau = TILT_TAU[0] + (TILT_TAU[1] - TILT_TAU[0]) * self.rand(n)
        self.alpha_tilt[ids] = 1 - torch.exp(-SUB_DT / tau)
        mount = torch.tensor(MOUNT, device=self.device)
        self.mount[ids] = (2 * self.rand((n, 2)) - 1) * mount * s["tilt"]
        self.lag[ids] = (self.rand(n) < 0.5) & s["latency"]
        self.fresh[ids] = True
        self.odometry[ids] = 0
        self.drift[ids] = (DRIFT[0] + (DRIFT[1] - DRIFT[0]) * self.rand(n)) * s["state_noise"]
        self.drift_angle[ids] = (2 * self.rand(n) - 1) * math.pi
        bias = torch.tensor(STATE_BIAS, device=self.device)
        self.state_bias[ids] = (2 * self.rand((n, 4)) - 1) * bias * s["state_noise"]

    def observe(self, ids=None):
        rows = self.rows if ids is None else ids
        out = {"depth": [], "state": [], "lidar": []}
        if self.student:
            out.update(student_depth=[], student_state=[])
        for chunk in rows.split(CHUNK):
            pos, yaw, height = self.pos[chunk], self.yaw[chunk], self.height[chunk]
            hits = raycast.cast(pos, yaw[:, None] + self.ray_angles, self.obstacles(chunk))
            rays, beams = hits[:, : -contract.LIDAR_BEAMS], hits[:, -contract.LIDAR_BEAMS :]
            # the fan of the student contains the image columns of the level teacher camera
            columns = rays[:, raycast.FAN_COLUMNS] if self.student else rays
            depth = raycast.depth_image(columns, height, self.env["obstacle_height"][chunk])
            out["depth"].append(contract.encode_depth(depth)[:, None])
            out["lidar"].append(contract.encode_depth(beams))
            goal = self.env["goal"][chunk] - pos
            vel, rate, prev = self.vel[chunk], self.yaw_rate[chunk], self.prev_action[chunk]
            out["state"].append(contract.state_vector(vel, yaw, rate, height, goal, prev))
            if self.student:
                out["student_depth"].append(self.student_image(chunk, rays, depth))
                out["student_state"].append(self.student_state(chunk))
        return {k: torch.cat(v) for k, v in out.items()}

    def student_image(self, rows, fan, level):
        if self.student["tilt"]:
            vibration = VIBRATION * self.randn((len(rows), 2))
            pitch, roll = (self.tilt[rows] + self.mount[rows] + vibration).unbind(-1)
            obstacle_height = self.env["obstacle_height"][rows]
            level = raycast.tilted_depth(fan, self.height[rows], obstacle_height, pitch, roll)
        image = contract.encode_student_depth(self.noise(level, rows))
        # a delayed camera shows the frame of the previous step, a new episode has none yet
        old = (self.lag[rows] & ~self.fresh[rows])[:, None, None, None]
        out = torch.where(old, self.last_image[rows], image)
        self.last_image[rows], self.fresh[rows] = image, False
        return out

    def student_state(self, rows):
        goal = self.env["goal"][rows] - self.pos[rows] - self.odometry[rows]
        vel, yaw, rate = self.vel[rows], self.yaw[rows], self.yaw_rate[rows]
        height, prev = self.height[rows], self.prev_action[rows]
        state = contract.state_vector(vel, yaw, rate, height, goal, prev)
        if not self.student["state_noise"]:
            return state
        noise = self.state_bias[rows] + self.state_noise * self.randn((len(rows), 4))
        return torch.cat([state[:, :4] + noise, state[:, 4:]], 1)

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
            if self.student:
                self.move_student(vel, moving)
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

    def move_student(self, vel, moving):
        """Attitude from the acceleration of this substep and the odometry error of its motion."""
        if self.student["tilt"]:
            a_fwd, a_lat = contract.to_heading(*((vel - self.vel) / SUB_DT).unbind(-1), self.yaw)
            # forward acceleration pitches the nose down, to the left lowers the left side
            target = torch.stack([torch.atan(a_fwd / G), -torch.atan(a_lat / G)], -1)
            self.tilt += (target - self.tilt) * (self.alpha_tilt * moving)[:, None]
        c, s = self.drift_angle.cos(), self.drift_angle.sin()
        dx, dy = (vel * SUB_DT * moving[:, None]).unbind(-1)
        self.odometry += self.drift[:, None] * torch.stack([c * dx - s * dy, s * dx + c * dy], -1)

    def track(self, levels, success):
        """Curriculum: promotion after PROMOTE_SR over the last episodes on the current level."""
        if not self.curriculum or self.level == len(mapgen.DENSITY) - 1:
            return
        self.window.extend(success[levels == self.level].tolist())
        if len(self.window) == PROMOTE_WINDOW and sum(self.window) >= PROMOTE_SR * PROMOTE_WINDOW:
            self.level += 1
            self.window.clear()
