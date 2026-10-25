import contextlib
import io

import gymnasium as gym
import numpy as np
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import ActionType, DroneModel, Physics
from gymnasium import spaces

XY_LIMIT = 2.0
Z_MIN, Z_MAX = 0.05, 2.5
MAX_TILT = 1.0


class GoalReachAviary(BaseRLAviary):
    """Fly from a random start to a random goal, either on motor RPMs or on velocity commands."""

    def __init__(
        self,
        act="vel",
        reward="dense",
        v_max=1.0,
        ctrl_freq=48,
        episode_sec=8.0,
        success_radius=0.15,
        gamma=0.99,
        physics=Physics.PYB,
        gui=False,
    ):
        assert act in ("rpm", "vel") and reward in ("dense", "sparse", "potential")
        self.ACT = act
        self.REWARD = reward
        self.V_MAX = v_max
        self.EPISODE_STEPS = int(episode_sec * ctrl_freq)
        self.SUCCESS_RADIUS = success_radius
        self.GAMMA = gamma
        self.goal = np.zeros(3)
        self.prev_action = np.zeros(4 if act == "rpm" else 3)
        self.prev_dist = 0.0
        # BaseAviary prints the URDF parameters on every construction, 16 workers make it noise
        with contextlib.redirect_stdout(io.StringIO()):
            super().__init__(
                drone_model=DroneModel.CF2X,
                physics=physics,
                pyb_freq=240,
                ctrl_freq=ctrl_freq,
                gui=gui,
                act=ActionType.RPM if act == "rpm" else ActionType.VEL,
            )

    def _actionSpace(self):
        return spaces.Box(-1.0, 1.0, shape=self.prev_action.shape, dtype=np.float32)

    def _observationSpace(self):
        dim = 13 + len(self.prev_action)
        return spaces.Box(-np.inf, np.inf, shape=(dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        # seed np_random here, BaseAviary.reset gets seed=None and keeps the generator
        gym.Env.reset(self, seed=seed)
        start = np.array([*self.np_random.uniform(-1, 1, 2), self.np_random.uniform(0.5, 1.5)])
        self.goal = self._sample_goal(start)
        self.INIT_XYZS = start[None]
        self.prev_action = np.zeros_like(self.prev_action)
        self.prev_dist = np.linalg.norm(self.goal - start)
        if self.ACT == "vel":
            self.ctrl[0].reset()
        return super().reset(options=options)

    def _sample_goal(self, start):
        while True:
            direction = self.np_random.normal(size=3)
            goal = start + self.np_random.uniform(0.5, 2.0) * direction / np.linalg.norm(direction)
            if np.all(np.abs(goal[:2]) <= XY_LIMIT) and 0.5 <= goal[2] <= 2.0:
                return goal

    def _preprocessAction(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1, 1)
        self.prev_action = action
        if self.ACT == "rpm":
            return (self.HOVER_RPM * (1 + 0.05 * action))[None]
        v = self.V_MAX * action
        speed = np.linalg.norm(v)
        if speed > self.V_MAX:
            v *= self.V_MAX / speed
        state = self._getDroneStateVector(0)
        rpm, _, _ = self.ctrl[0].computeControl(
            control_timestep=self.CTRL_TIMESTEP,
            cur_pos=state[0:3],
            cur_quat=state[3:7],
            cur_vel=state[10:13],
            cur_ang_vel=state[13:16],
            target_pos=state[0:3],
            target_rpy=np.zeros(3),
            target_vel=v,
        )
        return rpm[None]

    def _computeObs(self):
        pos, vel, rpy, ang_v = self.pos[0], self.vel[0], self.rpy[0], self.ang_v[0]
        obs = np.concatenate([self.goal - pos, vel, rpy, ang_v, pos[2:], self.prev_action])
        return obs.astype(np.float32)

    def _status(self):
        pos, (roll, pitch, _) = self.pos[0], self.rpy[0]
        dist = np.linalg.norm(self.goal - pos)
        tilt = np.arccos(np.clip(np.cos(roll) * np.cos(pitch), -1, 1))
        crash = pos[2] < Z_MIN or tilt > MAX_TILT
        out = np.any(np.abs(pos[:2]) > XY_LIMIT) or pos[2] > Z_MAX
        return dist, dist < self.SUCCESS_RADIUS, crash, out

    def _computeReward(self):
        dist, success, crash, out = self._status()
        sparse = 10.0 * success - 10.0 * (crash or out)
        if self.REWARD == "dense":
            reward = 10 * (self.prev_dist - dist) - 0.01 * np.sum(self.ang_v[0] ** 2) + sparse
        elif self.REWARD == "potential":
            # phi(s) = -10 d(s), phi = 0 in terminal states
            next_phi = 0.0 if success or crash or out else -10 * dist
            reward = sparse + self.GAMMA * next_phi + 10 * self.prev_dist
        else:
            reward = sparse
        self.prev_dist = dist
        return float(reward)

    def _computeTerminated(self):
        _, success, crash, out = self._status()
        return bool(success or crash or out)

    def _computeTruncated(self):
        # step_counter is advanced after this call, so the current step is counted here
        return self.step_counter // self.PYB_STEPS_PER_CTRL + 1 >= self.EPISODE_STEPS

    def _computeInfo(self):
        dist, success, crash, out = self._status()
        return {
            "is_success": bool(success),
            "crash": bool(crash),
            "out_of_bounds": bool(out),
            "distance": float(dist),
        }
