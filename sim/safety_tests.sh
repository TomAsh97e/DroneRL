#!/usr/bin/env bash
# Safety tests of M4.8 inside the container without GUI: builds the ROS 2 workspace and runs
# sim/safety_tests.py, one recorded flight per test.
#
# Usage: sim/safety_tests.sh [test ...]   policy_kill offboard_loss fence geofence pilot
set -euo pipefail
cd "$(dirname "$0")/.."

source sim/ws.sh

python3 -u sim/safety_tests.py "$@"
