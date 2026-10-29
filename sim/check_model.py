"""Checks of x500_dronerl in SITL (M4.3), run in the container:

    docker/run.sh python3 sim/check_model.py

1. Intrinsics from camera_info agree with HFOV 100 deg, the RGB and the depth camera match, and the
   central 90 deg crop reduced to the policy image has the focal length of the contract.
2. Hover with optical flow and the rangefinder: takeoff, then 60 s of position hold at the takeoff
   point, drift of the true position (Gazebo) at most 0.3 m, the M5 criterion checked in SITL.

Saves sample images and results.json to the output directory.
"""

import argparse
import json
import math
import os
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import torch
from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleLocalPosition, VehicleStatus
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from dronerl import contract
from dronerl.perception.depth_model import crop_fov, policy_size

REPO = Path(__file__).resolve().parents[1]
COMMANDER = REPO / "external/PX4-Autopilot/build/px4_sitl_default/bin/px4-commander"
HFOV = math.radians(100)
MAX_DRIFT = 0.3
PX4_QOS = qos_profile_sensor_data


class Probe(Node):
    """Keeps the latest message of every topic the checks read."""

    def __init__(self):
        super().__init__("check_model")
        self.msgs = {}
        self.track = []  # (sim time, truth x, y, z, EKF x, y, z, dist_bottom) during the hover
        self.recording = False
        topics = [
            ("rgb_info", CameraInfo, "/camera/camera_info", 10),
            ("depth_info", CameraInfo, "/camera/depth/camera_info", 10),
            ("rgb", Image, "/camera/image", 10),
            ("depth", Image, "/camera/depth/image", 10),
            ("truth", Odometry, "/ground_truth/odometry", 10),
            ("ekf", VehicleLocalPosition, "/fmu/out/vehicle_local_position_v1", PX4_QOS),
            ("status", VehicleStatus, "/fmu/out/vehicle_status_v1", PX4_QOS),
        ]
        for key, kind, topic, qos in topics:
            self.create_subscription(kind, topic, lambda m, k=key: self.store(k, m), qos)

    def store(self, key, msg):
        self.msgs[key] = msg
        if key == "truth" and self.recording and "ekf" in self.msgs:
            p, e = msg.pose.pose.position, self.msgs["ekf"]
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.track.append((t, p.x, p.y, p.z, e.x, e.y, e.z, e.dist_bottom))

    def wait(self, key, cond=lambda m: True, timeout=60):
        end = time.time() + timeout
        while time.time() < end:
            msg = self.msgs.get(key)
            if msg is not None and cond(msg):
                return msg
            time.sleep(0.1)
        raise TimeoutError(f"no {key} within {timeout} s")


def check_camera(probe, out):
    rgb, depth = probe.wait("rgb_info"), probe.wait("depth_info")
    fx, fy, cx, cy = rgb.k[0], rgb.k[4], rgb.k[2], rgb.k[5]
    expected = rgb.width / 2 / math.tan(HFOV / 2)
    h, w = crop_fov(torch.zeros(rgb.height, rgb.width), fx).shape
    size = policy_size(h, w, fx)
    res = {
        "size": [rgb.width, rgb.height],
        "fx": fx,
        "fy": fy,
        "c": [cx, cy],
        "fx_expected": expected,
        "depth_k_equal": bool(np.allclose(rgb.k, depth.k, rtol=1e-5)),
        "crop": [w, h],
        "policy_size": list(size),
        "policy_focal": fx * contract.IMG_W / w,
    }
    res["ok"] = bool(
        (rgb.width, rgb.height) == (640, 480)
        and abs(fx / expected - 1) < 1e-3
        and abs(fy / fx - 1) < 1e-3
        and res["depth_k_equal"]
        and size == (contract.IMG_H, contract.IMG_W)
        and abs(res["policy_focal"] / contract.FOCAL - 1) < 0.01
    )
    save_images(probe, out, "ground")
    return res


def save_images(probe, out, tag):
    rgb = probe.wait("rgb")
    img = np.frombuffer(rgb.data, np.uint8).reshape(rgb.height, rgb.width, 3)
    cv2.imwrite(str(out / f"rgb_{tag}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    d = probe.wait("depth")
    depth = np.frombuffer(d.data, np.float32).reshape(d.height, d.width)
    inv = contract.encode_depth(torch.from_numpy(depth.copy())).numpy()
    colored = cv2.applyColorMap((255 * inv).astype(np.uint8), cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(out / f"depth_{tag}.png"), colored)
    return depth


def commander(*args):
    subprocess.run([str(COMMANDER), *args], check=True, capture_output=True)


def check_hover(probe, out, duration):
    probe.wait("ekf", lambda m: m.xy_valid and m.z_valid, timeout=120)
    time.sleep(5)  # estimator settling after the first valid position
    commander("takeoff")
    # without a global position PX4 stays in Takeoff and holds the position there instead of Hold
    probe.wait("ekf", lambda m: m.z < -1.15 and abs(m.vz) < 0.1, timeout=60)
    time.sleep(5)
    depth = save_images(probe, out, "hover")
    probe.recording = True
    t0 = probe.wait("truth").header.stamp.sec
    probe.wait("truth", lambda m: m.header.stamp.sec >= t0 + duration, timeout=3 * duration)
    probe.recording = False
    commander("land")
    disarmed = VehicleStatus.ARMING_STATE_DISARMED
    probe.wait("status", lambda m: m.arming_state == disarmed, timeout=60)

    track = np.array(probe.track)
    header = "t,x,y,z,ekf_x,ekf_y,ekf_z,dist_bottom"
    np.savetxt(out / "hover.csv", track, delimiter=",", header=header)
    truth = np.linalg.norm(track[:, 1:3] - track[0, 1:3], axis=1)
    ekf = np.linalg.norm(track[:, 4:6] - track[0, 4:6], axis=1)
    res = {
        "duration": track[-1, 0] - track[0, 0],
        "drift_max": truth.max(),
        "drift_end": truth[-1],
        "ekf_drift_max": ekf.max(),
        "height_std": track[:, 3].std(),
        "truth_z": track[:, 3].mean(),
        "ekf_z": track[:, 6].mean(),
        "dist_bottom": track[:, 7].mean(),
        "floor_depth_bottom_row": float(np.median(depth[-1, 280:360])),
    }
    res["ok"] = bool(res["duration"] >= duration - 1 and res["drift_max"] <= MAX_DRIFT)
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--world", default="sim/gazebo/worlds/hall_empty.sdf")
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--out", type=Path, default=REPO / "runs/m4/checks/model")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ, HEADLESS="1", SITL_OUT=str(args.out / "sitl"))
    sitl = subprocess.Popen([str(REPO / "sim/sitl.sh"), args.world], env=env)
    rclpy.init()
    probe = Probe()
    threading.Thread(target=rclpy.spin, args=(probe,), daemon=True).start()
    try:
        results = {"camera": check_camera(probe, args.out)}
        print(json.dumps(results, indent=2), flush=True)
        results["hover"] = check_hover(probe, args.out, args.duration)
    finally:
        sitl.terminate()
        sitl.wait()
        rclpy.shutdown()
    print(json.dumps(results["hover"], indent=2))
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    ok = all(r["ok"] for r in results.values())
    print("OK" if ok else "FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
