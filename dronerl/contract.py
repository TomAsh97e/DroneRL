"""Interface contract (PLAN.md, section 2.5): the one definition of observations and actions.

Simulators, evaluation and later the ROS 2 nodes import it. Any change means a new VERSION and
retraining. v2 (plan/M3.md) adds the student image with a validity mask; the teacher keeps the
one-channel v1 image, it exists only in simulation.
"""

import math

import torch

VERSION = 2

CTRL_HZ = 10
DT = 1 / CTRL_HZ
EPISODE_STEPS = 60 * CTRL_HZ

# action a in [-1, 1]^3 -> (v_fwd, v_lat, yaw_rate) in the heading frame (FLU)
V_FWD_MAX = 1.0
V_BACK_MAX = 0.3
V_LAT_MAX = 0.5
YAW_RATE_MAX = math.radians(60)
ACT_DIM = 3

IMG_H, IMG_W = 48, 64
HFOV = math.radians(90)
FOCAL = IMG_W / 2 / math.tan(HFOV / 2)
DEPTH_MIN, DEPTH_MAX = 0.2, 8.0
# student image: inverse depth and validity mask, so a hole of a depth camera is not "far"
STUDENT_CHANNELS = 2

STATE_DIM = 10
# privileged teacher input, exists only in simulation
LIDAR_BEAMS = 72

FLIGHT_HEIGHT = 1.2
DRONE_RADIUS = 0.4  # X500 V2 with 10" propellers
COLLISION_MARGIN = 0.1
COLLISION_DIST = DRONE_RADIUS + COLLISION_MARGIN
SUCCESS_RADIUS = 0.5

GOAL_REWARD = 10.0
COLLISION_PENALTY = 10.0
TIME_PENALTY = 0.01
ACTION_RATE_W = 0.05
PROXIMITY_W = 0.1
PROXIMITY_DIST = 0.5


def action_to_command(a):
    """Action [..., 3] -> command [..., 3] in m/s and rad/s.

    Forward speed is scaled piecewise, so the zero action means hover, as for the watchdog of the
    safety filter. An affine scaling would map it to 0.35 m/s forward.
    """
    a = a.clamp(-1, 1)
    fwd = torch.where(a[..., 0] >= 0, V_FWD_MAX, V_BACK_MAX) * a[..., 0]
    return torch.stack([fwd, V_LAT_MAX * a[..., 1], YAW_RATE_MAX * a[..., 2]], -1)


def command_to_action(cmd):
    """Inverse of `action_to_command`, clipped to the action range."""
    fwd = cmd[..., 0] / torch.where(cmd[..., 0] >= 0, V_FWD_MAX, V_BACK_MAX)
    a = torch.stack([fwd, cmd[..., 1] / V_LAT_MAX, cmd[..., 2] / YAW_RATE_MAX], -1)
    return a.clamp(-1, 1)


def encode_depth(d):
    """Metric depth along the optical axis -> inverse depth in [0, 1].

    No hit (inf) and d >= DEPTH_MAX give 0, d <= DEPTH_MIN gives 1.
    """
    return ((1 / d - 1 / DEPTH_MAX) / (1 / DEPTH_MIN - 1 / DEPTH_MAX)).clamp(0, 1)


def encode_student_depth(d):
    """Metric depth [..., H, W] -> student image [..., 2, H, W]: inverse depth and validity mask.

    NaN marks missing data (holes, out of range of a depth camera, cropped borders) and gives 0 in
    both channels. inf and d >= DEPTH_MAX are a valid "far" reading.
    """
    valid = ~torch.isnan(d)
    depth = encode_depth(torch.where(valid, d, math.inf))
    return torch.stack([depth, valid.to(depth.dtype)], -3)


def decode_depth(v):
    """Inverse of `encode_depth` on (0, 1]; 0 decodes to DEPTH_MAX."""
    return 1 / (v * (1 / DEPTH_MIN - 1 / DEPTH_MAX) + 1 / DEPTH_MAX)


def to_heading(x, y, yaw):
    """World-frame vector components -> (forward, left) components in the heading frame."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    return c * x + s * y, c * y - s * x


def state_vector(vel, yaw, yaw_rate, height, goal_offset, prev_action):
    """State [..., 10] from the world-frame velocity and goal offset [..., 2].

    Order: v_fwd, v_lat, yaw_rate, height, cos and sin of the goal bearing, log(1 + distance),
    previous action (3).
    """
    v_fwd, v_lat = to_heading(vel[..., 0], vel[..., 1], yaw)
    gx, gy = to_heading(goal_offset[..., 0], goal_offset[..., 1], yaw)
    bearing = torch.atan2(gy, gx)
    goal = [bearing.cos(), bearing.sin(), torch.log1p(torch.hypot(gx, gy))]
    return torch.stack([v_fwd, v_lat, yaw_rate, height, *goal, *prev_action.unbind(-1)], -1)


def reward(progress, success, collision, action_change, clearance, prox_w=PROXIMITY_W):
    """Reward of PLAN.md, section 2.5.

    progress is d_{t-1} - d_t [m], action_change is ‖a_t - a_{t-1}‖², clearance is the distance to
    the nearest obstacle beyond COLLISION_DIST [m].
    """
    proximity = (1 - clearance / PROXIMITY_DIST).clamp(0, 1)
    return (
        progress
        + GOAL_REWARD * success
        - COLLISION_PENALTY * collision
        - TIME_PENALTY
        - ACTION_RATE_W * action_change
        - prox_w * proximity
    )
