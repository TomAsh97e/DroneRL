# Builds the ROS 2 workspace and sources it, for the scripts that fly inside the container:
# source sim/ws.sh from the repository root.
colcon --log-base ros2_ws/log build --base-paths ros2_ws/src --build-base ros2_ws/build \
    --install-base ros2_ws/install --symlink-install > /dev/null
# setup.bash reads unset variables
set +u
source ros2_ws/install/setup.bash
set -u
