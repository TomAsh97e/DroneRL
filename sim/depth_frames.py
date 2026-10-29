"""Gazebo RGB images with their true depth for the depth model (plan/M4.md, M4.10), inside the
container with the ROS 2 workspace built: the student flies pairs of the depth_* worlds, layouts
of the validation set, and every PERIOD seconds of the policy one frame of both cameras with the
same stamp is kept. One npz per flight in runs/m4/depth_gz/frames, until TARGET frames; flights
already saved are skipped.

    source sim/ws.sh && python3 sim/depth_frames.py
"""

import math
import time
from pathlib import Path

import numpy as np
import rclpy
from dronerl_ros import common
from dronerl_ros.depth_node import pixels
from dronerl_ros.probe import Probe, fly
from sensor_msgs.msg import CameraInfo, Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "m4" / "depth_gz"
WORLDS = ("depth_room", "depth_corridor")
PERIOD = 0.5  # s between kept frames, closer ones would mostly repeat the view
TARGET = 500


class Frames(Probe):
    def __init__(self, world, pair):
        super().__init__(world, pair)
        self.rgb, self.depth, self.kept = {}, {}, []
        self.fx, self.last = None, -math.inf
        self.create_subscription(Image, "/camera/image", self.on_rgb, 5)
        self.create_subscription(Image, "/camera/depth/image", self.on_depth, 5)
        self.create_subscription(CameraInfo, "/camera/camera_info", self.on_info, 1)

    def on_info(self, msg):
        self.fx = msg.k[0]

    def wanted(self, msg):
        return self.phase == "fly" and common.seconds(msg.header.stamp) >= self.last + PERIOD

    def on_rgb(self, msg):
        if self.wanted(msg):
            self.rgb[common.seconds(msg.header.stamp)] = pixels(msg, np.uint8, 3)
            self.match()

    def on_depth(self, msg):
        if self.wanted(msg):
            self.depth[common.seconds(msg.header.stamp)] = pixels(msg, np.float32, 1)[..., 0]
            self.match()

    def match(self):
        both = self.rgb.keys() & self.depth.keys()
        if not both:
            return
        t = max(both)
        self.kept.append((t, self.rgb[t], self.depth[t]))
        self.last = t
        self.rgb, self.depth = {}, {}

    def save(self, path):
        stamps, rgb, depth = zip(*self.kept, strict=True)
        np.savez_compressed(path, rgb=rgb, depth=depth, stamps=stamps, fx=self.fx)


def main():
    (OUT / "frames").mkdir(parents=True, exist_ok=True)
    rclpy.init()
    total = 0
    for world, pair in ((w, p) for p in range(20) for w in WORLDS):
        if total >= TARGET:
            break
        name = f"{world}_p{pair}"
        saved = OUT / "frames" / f"{name}.npz"
        if saved.exists():  # resumes a stopped collection
            with np.load(saved) as f:
                total += len(f["stamps"])
            continue
        out = OUT / "flights" / f"{name}_{time.strftime('%Y%m%d-%H%M%S')}"
        out.mkdir(parents=True)
        probe = Frames(world, pair)
        fly(probe, out, {"world": world, "pair": pair, "gui": "false", "record": "false"})
        if probe.kept:
            probe.save(OUT / "frames" / f"{name}.npz")
        total += len(probe.kept)
        print(f"{name}: {len(probe.kept)} frames, {total} in all", flush=True)
        probe.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
