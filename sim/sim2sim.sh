#!/usr/bin/env bash
# Sim2sim flights of M4.11 inside the container without GUI: builds the ROS 2 workspace and flies
# the series of sim/sim2sim_flights.py. The series resumes, so after a crash of the script itself
# the leftovers of its flight are stopped and it starts again, up to 5 times.
#
# Usage: sim/sim2sim.sh [series...] [--out DIR]
set -euo pipefail
cd "$(dirname "$0")/.."

source sim/ws.sh

for attempt in 1 2 3 4 5; do
    python3 -u sim/sim2sim_flights.py "$@" && exit 0
    echo "sim2sim_flights.py stopped with $?, start $attempt of 5"
    for name in "ros2 launch dronerl_ros" bin/px4 "gz sim" MicroXRCEAgent parameter_bridge; do
        pkill -f "$name" || true
    done
    sleep 10
done
exit 1
