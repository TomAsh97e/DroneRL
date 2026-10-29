"""Safety filter between the policy and PX4 (M4.5): the limits of the contract, a soft geofence,
a height band and a watchdog. Pure Python, the ROS node only feeds it messages.
"""

import numpy as np

from dronerl import contract
from dronerl.runtime import frames

WATCHDOG = 0.3  # s without a fresh command -> zero command, PX4 holds the position
# m inside the arena that the drone keeps from its edges by its estimate: COLLISION_DIST from a
# wall on the edge and the drift of the estimate, up to 0.4 m after 9 m with flow in SITL (M4.8);
# goals lie 1.04 m or more from the edges, so they stay within SUCCESS_RADIUS of the margin
FENCE_MARGIN = 1.1
# the speed towards an edge is limited to the room left before the margin over BRAKE, a stop from
# V_FWD_MAX takes 0.7 m in SITL
BRAKE = 0.75  # s
SLACK = 0.05  # m/s over that limit before a stop, the noise of the estimated velocity
HEIGHTS = (0.5, 2.0)  # m above the floor, outside the drone only holds its position
LOW = np.array([-contract.V_BACK_MAX, -contract.V_LAT_MAX, -contract.YAW_RATE_MAX])
HIGH = np.array([contract.V_FWD_MAX, contract.V_LAT_MAX, contract.YAW_RATE_MAX])


def clip(cmd):
    """Command (v_fwd, v_lat, yaw_rate) within the ranges of the contract."""
    return np.clip(cmd, LOW, HIGH)


def fence(vel, pos, box, margin=FENCE_MARGIN, now=(0.0, 0.0)):
    """Horizontal velocity [2] with the speed towards each edge of box (x0, y0, x1, y1) limited
    to the room left before margin over BRAKE, and stopped while the current velocity `now` is
    faster than that limit. A command cut to zero in a band along the edges kept the drone from
    goals near them (M4.11). Any frame where the box is axis aligned."""
    now = np.asarray(now, dtype=float)
    low = (pos[:2] - np.asarray(box[:2]) - margin).clip(min=0) / BRAKE
    high = (np.asarray(box[2:]) - pos[:2] - margin).clip(min=0) / BRAKE
    vel = np.clip(np.asarray(vel, dtype=float), -low, high)
    vel[(now < -low - SLACK) & (vel < 0)] = 0
    vel[(now > high + SLACK) & (vel > 0)] = 0
    return vel


class SafetyFilter:
    """Last fresh command of the policy, made safe for the current state.

    box is the arena in local NED (north, east), None without a fence. Times in seconds, the
    simulation time in SITL.
    """

    def __init__(self, box=None, margin=FENCE_MARGIN, heights=HEIGHTS, timeout=WATCHDOG):
        self.box, self.margin, self.heights, self.timeout = box, margin, heights, timeout
        self.cmd, self.stamp = np.zeros(3), -np.inf

    def command(self, cmd, t):
        self.cmd, self.stamp = clip(cmd), t

    def __call__(self, t, pos, yaw, height, vel=(0.0, 0.0)):
        """(command, reason): the command to send at time t and why it differs from the policy
        ("ok", "watchdog", "height", "fence"). vel is the current horizontal velocity."""
        if t - self.stamp > self.timeout:
            return np.zeros(3), "watchdog"
        if not self.heights[0] <= height <= self.heights[1]:
            return np.zeros(3), "height"
        if self.box is None:
            return self.cmd, "ok"
        target, yawspeed = frames.command_to_ned(self.cmd, yaw)
        safe = fence(target, pos, self.box, self.margin, vel)
        if np.array_equal(safe, target):
            return self.cmd, "ok"
        return frames.ned_to_command(safe, yawspeed, yaw), "fence"
