"""What the nodes share: QoS profiles, the flight (world and start-goal pair) and time."""

from pathlib import Path

import numpy as np
import rclpy
import torch
from geometry_msgs.msg import TwistStamped
from rclpy.executors import ExternalShutdownException
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)

import dronerl
from dronerl.envs import maps as mapgen

WORLDS = Path(dronerl.__file__).parents[1] / "sim" / "gazebo" / "worlds"

POSITION = "/fmu/out/vehicle_local_position_v1"
STATUS = "/fmu/out/vehicle_status_v1"
# body rates, vehicle_angular_velocity is not in the DDS topics of PX4
ODOMETRY = "/fmu/out/vehicle_odometry"

# the uXRCE-DDS client publishes best effort; to PX4 as in the px4_ros_com examples
PX4_OUT = qos_profile_sensor_data
PX4_IN = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)
# phase and origin of the flight reach nodes that subscribe late
LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
# a node busy with one frame takes the newest one next, older frames are dropped
LATEST = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

# phases of a flight on /dronerl/phase, in order; "pilot" when the pilot took over
PHASES = ("wait", "takeoff", "hover", "fly", "land", "pilot", "done")


def flight(world, pair, root=WORLDS):
    """Map of one flight: the obstacles of the world and its start-goal pair `pair`."""
    with np.load(Path(root) / world / "maps.npz") as f:
        return mapgen.get_map(dict(f), pair)


def seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def now(node):
    """Time of the node in seconds, the simulation time with use_sim_time."""
    return node.get_clock().now().nanoseconds * 1e-9


def px4_time(node):
    """Timestamp of a message to PX4 [us]."""
    return node.get_clock().now().nanoseconds // 1000


def twist(cmd, stamp):
    """Command (v_fwd, v_lat, yaw_rate) as TwistStamped, stamp of the camera frame it came from."""
    msg = TwistStamped()
    msg.header.stamp = stamp
    msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z = (float(c) for c in cmd)
    return msg


def command(msg):
    return np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z])


def run(make, threads=1):
    """Spins the node made by make() until Ctrl-C or the end of the launch. The nodes share the
    CPU with Gazebo and PX4, so torch gets few threads."""
    torch.set_num_threads(threads)
    rclpy.init()
    node = make()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
