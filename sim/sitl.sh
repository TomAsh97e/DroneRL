#!/usr/bin/env bash
# One simulated flight inside the container: Gazebo with a world, x500_dronerl spawned at a pose,
# PX4 SITL attached to it with the parameters from sim/px4, the Micro XRCE-DDS Agent and the ROS 2
# bridges. Every run gets a fresh PX4 working directory, so no parameters or estimator state carry
# over from the previous flight (plan/M4.md, D4).
#
# Usage: sim/sitl.sh <world.sdf> [x y yaw]   pose in the world frame (ENU), yaw in rad
# HEADLESS=1 runs without the Gazebo GUI; logs and the PX4 ulog go to $SITL_OUT (runs/m4/sitl/<time>).
# Runs until one of the parts exits or the script is stopped, then stops the rest.
set -euo pipefail

world_file=$(realpath "$1")
x=${2:-0} y=${3:-0} yaw=${4:-0}
world=$(sed -n 's/.*<world name="\([^"]*\)".*/\1/p' "$world_file" | head -1)
repo=$(cd "$(dirname "$0")/.." && pwd)
px4=$repo/external/PX4-Autopilot/build/px4_sitl_default
out=${SITL_OUT:-$repo/runs/m4/sitl/$(date +%Y%m%d-%H%M%S)}
mkdir -p "$out/px4"

# gz_env.sh appends to the Gazebo paths, which may not be set yet
set +u
source "$px4/rootfs/gz_env.sh"
set -u
export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:$repo/sim/gazebo/models

pids=()
gui=()
trap 'kill "${pids[@]}" "${gui[@]}" 2> /dev/null; wait' EXIT
trap 'exit 143' TERM INT

gz sim -s -r --headless-rendering "$world_file" > "$out/gz.log" 2>&1 &
pids+=($!)
if [[ -z ${HEADLESS:-} ]]; then
    gz sim -g > /dev/null 2>&1 &
    gui+=($!)
fi
until gz service -i --service "/world/$world/create" 2>&1 | grep -q "Service providers"; do
    sleep 0.5
done

read -r qz qw < <(python3 -c "import math; print(math.sin($yaw / 2), math.cos($yaw / 2))")
gz service -s "/world/$world/create" --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean \
    --timeout 5000 --req "sdf_filename: 'model://x500_dronerl', name: 'x500_dronerl', \
    pose: {position: {x: $x, y: $y}, orientation: {z: $qz, w: $qw}}" > /dev/null

# the agent brings its own Fast DDS, the ROS one on LD_LIBRARY_PATH would shadow it
env -u LD_LIBRARY_PATH MicroXRCEAgent udp4 -p 8888 > "$out/agent.log" 2>&1 &
pids+=($!)

# PX4 etc with one more airframe: x500 with sim/px4/dronerl.params as its defaults. Set before the
# airframe, as PX4_PARAM_* are, a value equal to the firmware default is lost to the x500 defaults.
cp -rs "$px4/etc" "$out/etc"
{
    echo '. ${R}etc/init.d-posix/airframes/4001_gz_x500'
    sed '/^#/d; /^$/d; s/^/param set-default /' "$repo/sim/px4/dronerl.params"
} > "$out/etc/init.d-posix/airframes/4900_gz_x500_dronerl"
PX4_SYS_AUTOSTART=4900 PX4_GZ_STANDALONE=1 PX4_GZ_WORLD=$world PX4_GZ_MODEL_NAME=x500_dronerl \
    "$px4/bin/px4" -d -w "$out/px4" "$out/etc" > "$out/px4.log" 2>&1 &
pids+=($!)

# not through `ros2 run`: the bridge would be its child and outlive the kill at the exit, so the
# next flight would get every message twice
"$(ros2 pkg prefix ros_gz_bridge)/lib/ros_gz_bridge/parameter_bridge" \
    --ros-args -p config_file:="$repo/sim/gazebo/bridge.yaml" > "$out/bridge.log" 2>&1 &
pids+=($!)

wait -n "${pids[@]}"
