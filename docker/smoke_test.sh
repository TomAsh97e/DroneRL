#!/usr/bin/env bash
# M4.2 smoke test inside the container: PX4 SITL builds, takes off and lands in Gazebo with its
# topics visible in ROS 2, and the depth model runs on the GPU through ONNX Runtime.
# Run from the host: HEADLESS=1 docker/run.sh docker/smoke_test.sh
set -euo pipefail

px4=/work/external/PX4-Autopilot
log=/work/runs/m4/logs
mkdir -p "$log"

echo "building px4_sitl"
make -C "$px4" px4_sitl > "$log/px4_build.log" 2>&1

# the agent brings its own Fast DDS, the ROS one on LD_LIBRARY_PATH would shadow it
env -u LD_LIBRARY_PATH MicroXRCEAgent udp4 -p 8888 > "$log/agent.log" 2>&1 &
cd "$px4/build/px4_sitl_default"
HEADLESS=1 PX4_SYS_AUTOSTART=4001 PX4_SIM_MODEL=gz_x500 PX4_GZ_WORLD=default \
    ./bin/px4 -d > "$log/px4_sitl.log" 2>&1 &
trap 'pkill -f bin/px4; pkill -f MicroXRCEAgent; pkill -f "gz sim"' EXIT

position=""
for _ in $(seq 60); do
    position=$(ros2 topic list | grep -m 1 "/fmu/out/vehicle_local_position" || true)
    [[ -n "$position" ]] && break
    sleep 2
done
[[ -n "$position" ]] || { echo "no PX4 topics in ROS 2"; exit 1; }
echo "PX4 topics in ROS 2: $(ros2 topic list | grep -c /fmu/)"

height() {
    ros2 topic echo --once --qos-reliability best_effort --field z "$position" | head -1
}
# the x500 airframe returns home without a MAVLink ground station, here the companion is on DDS
./bin/px4-param set NAV_DLL_ACT 0
sleep 15  # EKF and GPS fix before takeoff
./bin/px4-commander takeoff
sleep 15
echo "z after takeoff: $(height)"
./bin/px4-commander land
sleep 15
echo "z after landing: $(height)"

python3 - <<'EOF'
import time

import numpy as np
import onnxruntime as ort

ort.preload_dlls()
path = "/work/runs/m3/onnx/depth_ft_full_r518.onnx"
session = ort.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print("ONNX Runtime providers:", session.get_providers())
inp = session.get_inputs()[0]
x = np.random.rand(1, *inp.shape[1:]).astype(np.float32)
for _ in range(5):
    session.run(None, {inp.name: x})
start = time.perf_counter()
for _ in range(20):
    session.run(None, {inp.name: x})
print(f"depth model: {1000 * (time.perf_counter() - start) / 20:.1f} ms per image")
EOF
