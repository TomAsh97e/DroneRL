#!/usr/bin/env bash
# System identification of M4.9 inside the container without GUI: builds the ROS 2 workspace,
# flies the steps and the policy flights into runs/m4/sysid/<time> and fits them.
#
# Usage: sim/sysid.sh [steps] [latency]
set -euo pipefail
cd "$(dirname "$0")/.."

source sim/ws.sh

out=runs/m4/sysid/$(date +%Y%m%d-%H%M%S)
python3 -u sim/sysid_flights.py "$out" "$@"
python3 -m dronerl.eval.sysid "$out"
