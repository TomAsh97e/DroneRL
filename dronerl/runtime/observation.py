"""Input of the student on the drone (M4.5): the state from PX4 data through
`contract.state_vector`, as NavEnv builds it, and the stack of the last 3 student images.

The state is computed in ENU, where the policy was trained; in NED the heading turns the other way
and the heading frame would come out mirrored.
"""

import numpy as np
import torch

from dronerl import contract
from dronerl.runtime import frames


def student_state(pos, vel, yaw, yaw_rate, height, goal, prev_action):
    """State [10] from PX4 data: local NED position and velocity, heading, body yaw rate (FRD,
    about the down axis), height above the floor [m], goal in local NED and the previous action."""

    def t(x):
        return torch.as_tensor(x, dtype=torch.float32)

    v = t(frames.ned_to_enu(vel[:2]))
    g = t(frames.ned_to_enu(np.asarray(goal[:2]) - pos[:2]))
    psi = t(frames.yaw_ned_to_enu(yaw))
    return contract.state_vector(v, psi, t(-yaw_rate), t(height), g, t(prev_action))


class Observation:
    """Student input over one flight, toward goal (local NED). Like NavEnv after a reset: no
    previous action and the first image repeated over the whole stack."""

    def __init__(self, goal, frames=3):
        self.goal, self.frames = np.asarray(goal, dtype=float), frames
        self.prev = torch.zeros(contract.ACT_DIM)
        self.stack = None

    def image(self, image):
        """Frames [frames * 2, H, W] after a new student image [2, H, W]."""
        if self.stack is None:
            self.stack = image.repeat(self.frames, 1, 1)
        else:
            self.stack = torch.cat([self.stack[contract.STUDENT_CHANNELS :], image])
        return self.stack

    def state(self, pos, vel, yaw, yaw_rate, height):
        return student_state(pos, vel, yaw, yaw_rate, height, self.goal, self.prev)

    def acted(self, action):
        """The action the policy chose for the last input, it is the previous action next time.
        What the safety filter makes of it does not count, NavEnv has no filter either."""
        self.prev = torch.as_tensor(action, dtype=torch.float32).clamp(-1, 1)
