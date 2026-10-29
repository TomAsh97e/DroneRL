"""Flights of the sim2sim evaluation (plan/M4.md, M4.11), inside the container with the ROS 2
workspace built: each series of RUNS flies the start-goal pairs of the three worlds, PX4 and
Gazebo started anew for every flight (D4). A flight without a result or aborted by the chain (no
command from the policy, offboard lost) is an error of the infrastructure: it flies again, up to
RETRIES times, the failed attempt kept as <name>_error<k>. Flights with a result are skipped, so
a stopped series resumes. Each flight also gets estimate.json, the error of the PX4 estimate
against the ground truth. dronerl.eval.sim2sim compares them with NavEnv.

    source sim/ws.sh && python3 sim/sim2sim_flights.py [series ...] [--out DIR]
"""

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rclpy
import torch
from dronerl_ros import common
from dronerl_ros.probe import Probe, fly
from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleLocalPosition

from dronerl import contract
from dronerl.agents.scripted import PurePursuit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "m4" / "sim2sim" / "flights"
WORLDS = ("arena", "corridor", "room")
PAIRS = 20
STUDENTS = {
    "asym_s0": "runs/m4/onnx/student_asym_nav-v2_s0_20261028-020943.onnx",
    "asym_s1": "runs/m4/onnx/student_asym_nav-v2_s1_20261028-043342.onnx",
    "asym_s2": "runs/m4/onnx/student_asym_nav-v2_s2_20261028-065723.onnx",
    "dagger_s0": "runs/m4/onnx/student_main_nav-v2_s0_20261027-191054.onnx",
    # fine-tuned in M4.12
    "asym_ft_s0": "runs/m4/onnx/student_asym_ft_nav-v2_s0_20261029-112310.onnx",
    "asym_ft_s1": "runs/m4/onnx/student_asym_ft_nav-v2_s1_20261029-122140.onnx",
    "asym_ft_s2": "runs/m4/onnx/student_asym_ft_nav-v2_s2_20261029-132023.onnx",
}
DEPTH_MODEL = "runs/m3/onnx/depth_ft_full_r518.onnx"  # chosen in M4.10
# series: (policy, source of the student image), in the order of the plan
RUNS = {
    "astar": ("astar", None),
    "asym_s0_gz_depth": ("asym_s0", "depth"),
    "asym_s0_gz_rgb": ("asym_s0", "rgb"),
    "asym_s1_gz_rgb": ("asym_s1", "rgb"),
    "asym_s2_gz_rgb": ("asym_s2", "rgb"),
    "dagger_s0_gz_rgb": ("dagger_s0", "rgb"),
    "asym_ft_s0_gz_rgb": ("asym_ft_s0", "rgb"),
    "asym_ft_s1_gz_rgb": ("asym_ft_s1", "rgb"),
    "asym_ft_s2_gz_rgb": ("asym_ft_s2", "rgb"),
}
RETRIES = 2
TIMEOUT = 300  # s of wall time, a flight with the start and landing takes 1-2 min


class Flight(Probe):
    """Keeps the ground truth and the estimate of PX4 while the policy flies, to tell the errors
    of the estimate from those of the policy."""

    def __init__(self, world, pair):
        super().__init__(world, pair)
        self.gt, self.est = [], []
        self.create_subscription(Odometry, "/ground_truth/odometry", self.on_odometry, 50)
        self.create_subscription(
            VehicleLocalPosition, common.POSITION, self.on_position, common.PX4_OUT
        )

    def on_odometry(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
        if self.phase == "fly":
            self.gt.append((self.now(), p.x, p.y, yaw))
        return p, yaw

    def on_position(self, msg):
        if self.phase == "fly":
            self.est.append((self.now(), msg.x, msg.y, msg.heading))

    def errors(self):
        """Error of the estimate in the local frame of PX4, NED with the origin at the start."""
        if not self.gt or not self.est:
            return None
        gt, est = np.array(self.gt), np.array(self.est)
        k = np.searchsorted(gt[:, 0], est[:, 0]).clip(max=len(gt) - 1)
        x0, y0 = self.map["start"]
        true = np.stack([gt[k, 2] - y0, gt[k, 1] - x0], -1)
        pos = np.linalg.norm(est[:, 1:3] - true, axis=-1)
        heading = np.abs(
            np.degrees(np.remainder(est[:, 3] - np.pi / 2 + gt[k, 3] + np.pi, math.tau) - np.pi)
        )
        return {
            "position_m": {"final": float(pos[-1]), "max": float(pos.max())},
            "heading_deg": {"p50": float(np.median(heading)), "max": float(heading.max())},
            "path_m": float(np.linalg.norm(np.diff(gt[:, 1:3], axis=0), axis=-1).sum()),
        }


class Pursuit(Flight):
    """Pure pursuit of the A* path as in NavEnv (M2.4) on the pose from Gazebo, at the control
    rate: a check of the chain from the command to PX4, without perception and the student."""

    launch = {"policy": "false"}

    def __init__(self, world, pair):
        super().__init__(world, pair)
        cpu = torch.device("cpu")
        rows, steps = torch.arange(1), torch.zeros(1, dtype=torch.long)
        self.env = SimpleNamespace(num_envs=1, device=cpu, rows=rows, map_idx=rows, steps=steps)
        self.env.pos = self.env.yaw = None
        self.follow = PurePursuit(self.env, self.map["path"][None])
        self.next = None

    def on_odometry(self, msg):
        p, yaw = super().on_odometry(msg)
        self.env.pos, self.env.yaw = torch.tensor([[p.x, p.y]]), torch.tensor([yaw])

    def tick(self):
        if self.phase != "fly" or self.env.pos is None:
            return
        t = self.now()
        if self.next is not None and t < self.next:
            return
        self.next = t + contract.DT
        cmd = contract.action_to_command(self.follow(None))[0]
        self.env.steps += 1
        self.send(cmd.tolist())


def done(out):
    path = out / "result.json"
    return path.exists() and json.loads(path.read_text())["outcome"] != "aborted"


def flight(series, world, pair, out):
    policy, source = RUNS[series]
    out.mkdir(parents=True)
    # the soft fence acts on an estimate that is off by up to 1 m in SITL and NavEnv has none,
    # so it would mix its own failures into the comparison; it is tested on its own in M4.8
    args = {"world": world, "pair": pair, "gui": "false", "fence": "false"}
    if policy == "astar":
        probe = Pursuit(world, pair)
        args |= probe.launch
    else:
        probe = Flight(world, pair)
        args |= {"student": STUDENTS[policy], "source": source, "depth_model": DEPTH_MODEL}
    fly(probe, out, args, TIMEOUT)
    errors = probe.errors()
    if errors is not None:
        (out / "estimate.json").write_text(json.dumps(errors, indent=2))
    probe.destroy_node()


def attempt(root, series, world, pair):
    """Outcome of the flight, flown again after an error of the infrastructure; "error" when the
    last allowed attempt failed too."""
    out = root / series / f"{world}_p{pair:02d}"
    while not done(out):
        failed = sorted(out.parent.glob(f"{out.name}_error*"))
        if out.exists():
            if len(failed) >= RETRIES:
                return "error"
            out.rename(out.with_name(f"{out.name}_error{len(failed)}"))
        flight(series, world, pair, out)
    return json.loads((out / "result.json").read_text())["outcome"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("series", nargs="*", default=list(RUNS))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    rclpy.init()
    for series in args.series:
        for world in WORLDS:
            for pair in range(PAIRS):
                outcome = attempt(args.out.resolve(), series, world, pair)
                print(f"{series} {world}_p{pair:02d}: {outcome}", flush=True)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
