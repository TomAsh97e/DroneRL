#!/usr/bin/env bash
# ROS 2, px4_msgs and, once built, the workspace of the repository
source /opt/ros/jazzy/setup.bash
source /opt/px4_ws/install/setup.bash
if [[ -f /work/ros2_ws/install/setup.bash ]]; then
    source /work/ros2_ws/install/setup.bash
fi
# the gz vendor packages of ROS point the gz tool at themselves only, hiding gz sim of Harmonic
export GZ_CONFIG_PATH=/usr/share/gz:$GZ_CONFIG_PATH
exec "$@"
