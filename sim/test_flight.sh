#!/usr/bin/env bash
# Integration test of the flight stack (M4.7), inside the container without GUI: builds the ROS 2
# workspace, flies one start-goal pair of the empty world with the whole chain and passes when
# the flight ends in success.
#
# Usage: sim/test_flight.sh [pair] [launch arguments...]
set -euo pipefail
cd "$(dirname "$0")/.."

source sim/ws.sh

out=runs/m4/test_flight/$(date +%Y%m%d-%H%M%S)
timeout 600 ros2 launch dronerl_ros sim.launch.py world:=empty pair:="${1:-0}" out:="$out" \
    gui:=false "${@:2}" || true
python3 - "$out/result.json" << 'PY'
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    sys.exit(f"FAIL: no {path}")
result = json.loads(path.read_text())
print(json.dumps(result, indent=2))
sys.exit(0 if result["outcome"] == "success" else f"FAIL: {result['outcome']}")
PY
