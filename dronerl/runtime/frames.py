"""Frames of the drone runtime (plan/M4.md). Maps, Gazebo and the policy use ENU and FLU with the
heading counter-clockwise from x (east), PX4 uses NED and FRD with the heading clockwise from
north. All conversions between them live here.
"""

import math

import numpy as np


def wrap(angle):
    """Angle in [-pi, pi)."""
    return (np.asarray(angle) + math.pi) % (2 * math.pi) - math.pi


def ned_to_enu(v):
    """Vectors [..., 2] or [..., 3]: x and y swap, z flips. The map is its own inverse."""
    v = np.array(v, dtype=float)
    v[..., :2] = v[..., 1::-1].copy()
    v[..., 2:] *= -1
    return v


enu_to_ned = ned_to_enu


def frd_to_flu(v):
    """Body vectors [..., 3]: y and z flip. Its own inverse as well."""
    v = np.array(v, dtype=float)
    v[..., 1:] *= -1
    return v


flu_to_frd = frd_to_flu


def yaw_ned_to_enu(yaw):
    """psi_ENU = pi/2 - psi_NED, and the other way round."""
    return wrap(math.pi / 2 - np.asarray(yaw))


yaw_enu_to_ned = yaw_ned_to_enu


def command_to_ned(cmd, yaw):
    """Command (v_fwd, v_lat, yaw_rate) of the policy in the heading frame (FLU) and the PX4
    heading yaw -> NED velocity (v_north, v_east) and yawspeed of a TrajectorySetpoint."""
    v_fwd, v_lat, rate = cmd
    c, s = math.cos(yaw), math.sin(yaw)
    # forward is (c, s) and left is (s, -c) in (north, east)
    return np.array([c * v_fwd + s * v_lat, s * v_fwd - c * v_lat]), -rate


def ned_to_command(vel, yawspeed, yaw):
    """Inverse of `command_to_ned`."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * vel[0] + s * vel[1], s * vel[0] - c * vel[1], -yawspeed])


def map_to_ned(p, start, origin):
    """Point [..., 2] of the map (ENU) -> PX4 local NED, where the map point `start` is at the local
    position `origin`. PX4 keeps its axes on north and east, as Gazebo keeps the map on ENU."""
    return np.asarray(origin, dtype=float)[:2] + enu_to_ned(np.asarray(p) - start)


def ned_to_map(p, start, origin):
    """Inverse of `map_to_ned`."""
    return np.asarray(start, dtype=float) + ned_to_enu(np.asarray(p)[..., :2] - origin[:2])
