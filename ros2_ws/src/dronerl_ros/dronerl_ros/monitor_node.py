"""Outcome and metrics of a flight (plan/M4.md) from the ground truth of Gazebo and the map, judged
as NavEnv judges an episode: collision, success, leaving the area, timeout after 60 s of the
policy. Also the frame -> command latency in simulation and wall time, the real-time factor and
JPEG copies of the camera images for the bag.

Writes result.json and run.npz to `out`; run.npz holds the arrays of nav_eval.rollout for this
one episode, sampled at the control rate.
"""

import bisect
import json
import math
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import torch
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32MultiArray, String

from dronerl import contract
from dronerl.envs import maps as mapgen
from dronerl_ros import common
from dronerl_ros.depth_node import pixels


def percentiles(x):
    if not x:
        return None
    p50, p95 = np.percentile(x, [50, 95])
    return {"p50": float(p50), "p95": float(p95), "max": float(np.max(x)), "n": len(x)}


class MonitorNode(Node):
    def __init__(self):
        super().__init__("monitor_node")
        world = self.declare_parameter("world", "empty").value
        pair = self.declare_parameter("pair", 0).value
        self.out = Path(self.declare_parameter("out", "runs/m4/flights/latest").value)
        self.jpeg_period = 1 / self.declare_parameter("jpeg_rate", 10.0).value
        self.quality = self.declare_parameter("jpeg_quality", 80).value
        # s of the policy before the timeout, longer only for flights without it (M4.9)
        episode = contract.EPISODE_STEPS * contract.DT
        self.max_time = self.declare_parameter("max_time", episode).value
        self.map = common.flight(world, pair)
        self.info = {"world": world, "pair": pair}
        self.start = self.prev = self.result = None
        self.ticks, self.events = [], Counter()
        self.latency = {"sim_ms": [], "wall_ms": []}
        self.action = np.zeros(contract.ACT_DIM)
        self.clock = deque(maxlen=5000)  # (simulation, wall) time pairs, 20 s at 250 Hz
        self.path = 0.0
        self.min_clear = math.inf
        self.jpeg_last = -math.inf

        self.create_subscription(Odometry, "/ground_truth/odometry", self.on_odometry, 50)
        self.create_subscription(String, "/dronerl/phase", self.on_phase, common.LATCHED)
        self.create_subscription(Float32MultiArray, "/dronerl/action", self.on_action, 10)
        self.create_subscription(TwistStamped, "/dronerl/cmd", self.on_cmd, 10)
        self.create_subscription(String, "/dronerl/safety", self.on_safety, 10)
        self.create_subscription(Clock, "/clock", self.on_clock, 10)
        self.create_subscription(Image, "/camera/image", self.on_image, common.LATEST)
        self.result_pub = self.create_publisher(String, "/dronerl/result", common.LATCHED)
        self.jpeg_pub = self.create_publisher(CompressedImage, "/camera/image/compressed", 10)

    def on_clock(self, msg):
        self.clock.append((common.seconds(msg.clock), time.monotonic()))

    def wall_at(self, t):
        """Wall time when the simulation was at t, from the /clock history."""
        i = bisect.bisect_left(self.clock, (t, -math.inf))
        return self.clock[min(i, len(self.clock) - 1)][1] if self.clock else math.nan

    def on_cmd(self, msg):
        if self.start is None or self.result is not None:
            return
        t = common.seconds(msg.header.stamp)
        self.latency["sim_ms"].append(1000 * (common.now(self) - t))
        self.latency["wall_ms"].append(1000 * (time.monotonic() - self.wall_at(t)))

    def on_action(self, msg):
        self.action = np.array(msg.data)

    def on_safety(self, msg):
        if self.start is not None and self.result is None:
            self.events[msg.data] += 1

    def on_image(self, msg):
        t = common.seconds(msg.header.stamp)
        if t - self.jpeg_last < self.jpeg_period:
            return
        self.jpeg_last = t
        bgr = cv2.cvtColor(pixels(msg, np.uint8, 3), cv2.COLOR_RGB2BGR)
        _, data = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        self.jpeg_pub.publish(
            CompressedImage(header=msg.header, format="jpeg", data=data.tobytes())
        )

    def on_phase(self, msg):
        if msg.data == "fly" and self.start is None:
            self.start = (common.now(self), time.monotonic())
            self.next_tick = self.start[0]
            self.get_logger().info("flight started")
        elif msg.data in ("land", "pilot") and self.start is not None and self.result is None:
            self.finish("aborted", self.prev, self.ticks[-1][1] if self.ticks else 0.0)

    def on_odometry(self, msg):
        if self.start is None or self.result is not None:
            return
        t = common.seconds(msg.header.stamp)
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        pos = np.array([p.x, p.y])
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
        if self.prev is not None:
            self.path += float(np.linalg.norm(pos - self.prev))
        self.prev = pos
        m = self.map
        clear = float(mapgen.sdf(pos[None], m["circles"], m["boxes"])[0])
        self.min_clear = min(self.min_clear, clear)
        dist = float(np.linalg.norm(m["goal"] - pos))
        if t >= self.next_tick:
            self.ticks.append((pos, yaw, dist, clear, self.action.copy()))
            self.next_tick += contract.DT
        if clear < contract.COLLISION_DIST:
            self.finish("collision", pos, yaw)
        elif dist < contract.SUCCESS_RADIUS:
            self.finish("success", pos, yaw)
        elif (pos < 0).any() or (pos > m["size"]).any():
            self.finish("out_of_area", pos, yaw)
        elif t - self.start[0] >= self.max_time:
            self.finish("timeout", pos, yaw)

    def finish(self, outcome, pos, yaw):
        """Ends the flight: the last pose closes the sampled trajectory, as the pose after the
        last step of NavEnv."""
        m = self.map
        n = len(self.ticks)
        pos = np.zeros(2) if pos is None else pos
        positions = np.array([x[0] for x in self.ticks] + [pos]).reshape(n + 1, 2)
        yaws = np.array([x[1] for x in self.ticks] + [yaw])
        actions = np.array([x[4] for x in self.ticks]).reshape(n, contract.ACT_DIM)
        dist = np.linalg.norm(m["goal"] - positions, axis=-1)
        clear = mapgen.sdf(positions, m["circles"], m["boxes"])
        last = np.arange(n) == n - 1
        prev = np.concatenate([np.zeros((1, contract.ACT_DIM)), actions[:-1]])

        def t(x):
            return torch.as_tensor(x, dtype=torch.float32)

        reward = contract.reward(
            t(dist[:-1] - dist[1:]),
            t(last & (outcome == "success")),
            t(last & (outcome == "collision")),
            t(((actions - prev) ** 2).sum(-1)),
            t(clear[1:] - contract.COLLISION_DIST),
        )
        sim, wall = common.now(self) - self.start[0], time.monotonic() - self.start[1]
        self.result = {
            **self.info,
            "outcome": outcome,
            "length": n,
            "time_s": sim,
            "path_length": self.path,
            "min_clearance": self.min_clear - contract.COLLISION_DIST,
            "return": float(reward.sum()),
            "final_distance": float(dist[-1]),
            "real_time_factor": sim / wall if wall > 0 else None,
            "latency_ms": {k: percentiles(v) for k, v in self.latency.items()},
            "safety_events": dict(self.events),
        }
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "result.json").write_text(json.dumps(self.result, indent=2))
        flags = {k: outcome == k for k in ("success", "collision", "out_of_area", "timeout")}
        np.savez(
            self.out / "run.npz",
            pos=positions,
            yaw=yaws,
            actions=actions,
            alive=np.ones(n, bool),
            length=n,
            path_length=self.path,
            min_clearance=self.result["min_clearance"],
            reward=reward.numpy(),
            **flags,
        )
        self.result_pub.publish(String(data=json.dumps(self.result)))
        self.get_logger().info(f"flight over: {outcome}, {n} steps, path {self.path:.2f} m")


def main():
    common.run(MonitorNode)
