"""One simulated flight (plan/M4.md): Gazebo with the world and x500_dronerl at the start of a
pair, PX4 SITL, the Micro XRCE-DDS Agent and the bridges (all by sim/sitl.sh), the nodes of
dronerl_ros and a bag. Ends when mission_node is done.

    ros2 launch dronerl_ros sim.launch.py world:=arena pair:=3 source:=rgb gui:=true
"""

import json
import time
from pathlib import Path

import numpy as np
from ament_index_python.packages import get_package_share_directory
from dronerl_ros.common import WORLDS, flight
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ROOT = WORLDS.parents[2]
CONFIG = Path(get_package_share_directory("dronerl_ros")) / "config" / "nodes.yaml"
ARGS = {
    "world": ("empty", "world in sim/gazebo/worlds"),
    "pair": ("0", "start-goal pair of the world"),
    "student": ("runs/m4/onnx/student_asym_nav-v2_s0_20261028-020943.onnx", "student in ONNX"),
    "source": ("rgb", "student image from rgb (depth model) or depth (depth camera)"),
    "depth_model": ("runs/m3/onnx/depth_ft_full_r518.onnx", "depth model in ONNX"),
    "fence": ("true", "soft geofence of safety_node"),
    "geofence": ("", "radius of the hard geofence of PX4 around home [m], from the map if empty"),
    "policy": ("true", "depth_node and policy_node, else /dronerl/cmd comes from elsewhere"),
    "params": ("", "YAML of node parameters over config/nodes.yaml"),
    "gui": ("false", "Gazebo GUI"),
    "record": ("true", "rosbag2 (mcap) of the flight"),
    "out": ("", "output directory, runs/m4/flights/<world>_p<pair>_<source>_<time> if empty"),
}
TOPICS = [
    "/camera/image/compressed",
    "/dronerl/student_image",
    "/dronerl/depth_ms",
    "/dronerl/state",
    "/dronerl/action",
    "/dronerl/cmd",
    "/dronerl/safe_cmd",
    "/dronerl/safety",
    "/dronerl/phase",
    "/dronerl/origin",
    "/dronerl/result",
    "/ground_truth/odometry",
    "/fmu/out/vehicle_local_position_v1",
    "/fmu/out/vehicle_status_v1",
    "/fmu/out/vehicle_odometry",
    "/fmu/in/trajectory_setpoint",
    "/fmu/in/offboard_control_mode",
    "/fmu/in/vehicle_command",
    "/fmu/in/manual_control_input",
]


def setup(context):
    arg = {k: LaunchConfiguration(k).perform(context) for k in ARGS}
    world, pair, source = arg["world"], int(arg["pair"]), arg["source"]
    m = flight(world, pair)
    name = f"{world}_p{pair}_{source}_{time.strftime('%Y%m%d-%H%M%S')}"
    out = ROOT / (arg["out"] or f"runs/m4/flights/{name}")
    out.mkdir(parents=True, exist_ok=True)
    # the whole map seen from the start, so only a drone that got away hits it
    far = np.maximum(m["start"], m["size"] - m["start"])
    radius = float(arg["geofence"] or np.hypot(*far) + 0.5)
    info = {**arg, "out": str(out), "start": m["start"].tolist(), "goal": m["goal"].tolist()}
    info["geofence"] = radius
    (out / "flight.json").write_text(json.dumps(info, indent=2))

    x, y = m["start"]
    cmd = [str(ROOT / "sim" / "sitl.sh"), str(WORLDS / world / "world.sdf"), str(x), str(y)]
    cmd.append(str(m["yaw0"]))
    env = {"SITL_OUT": str(out / "sitl"), "PX4_PARAM_GF_MAX_HOR_DIST": str(radius)}
    if arg["gui"] == "true":
        cmd = ["env", "-u", "HEADLESS", *cmd]
    else:
        env["HEADLESS"] = "1"
    sitl = ExecuteProcess(cmd=cmd, additional_env=env, name="sitl")

    shared = {"use_sim_time": True, "world": world, "pair": pair}
    extra = [str(ROOT / arg["params"])] if arg["params"] else []

    def node(executable, **params):
        return Node(
            package="dronerl_ros",
            executable=executable,
            parameters=[str(CONFIG), *extra, shared, params],
            output="screen",
        )

    mission = node("mission_node")
    actions = [
        sitl,
        node("safety_node", fence=arg["fence"] == "true"),
        node("px4_bridge"),
        node("monitor_node", out=str(out)),
        mission,
    ]
    if arg["policy"] == "true":
        actions.append(node("depth_node", source=source, model=str(ROOT / arg["depth_model"])))
        actions.append(node("policy_node", model=str(ROOT / arg["student"])))
    if arg["record"] == "true":
        bag = ["ros2", "bag", "record", "-s", "mcap", "--use-sim-time", "-o", str(out / "bag")]
        actions.append(ExecuteProcess(cmd=bag + TOPICS, name="bag"))
    # the flight is over when the mission is, or when the simulation stops
    for action in (mission, sitl):
        stop = EmitEvent(event=Shutdown(reason="flight over"))
        actions.append(RegisterEventHandler(OnProcessExit(target_action=action, on_exit=[stop])))
    return actions


def generate_launch_description():
    args = [DeclareLaunchArgument(k, default_value=v, description=d) for k, (v, d) in ARGS.items()]
    return LaunchDescription([*args, OpaqueFunction(function=setup)])
