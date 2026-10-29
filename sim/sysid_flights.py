"""Flights of the system identification in SITL (plan/M4.md, M4.9), inside the container with the
ROS 2 workspace built (sim/sysid.sh). "steps" flies steps of the commands without the policy,
"latency" flies the policy and times its chain. The raw records go to runs/m4/sysid/<time>/,
dronerl.eval.sysid fits them.

    python3 sim/sysid_flights.py <out dir> [steps] [latency]
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from dronerl_ros import common
from dronerl_ros.probe import Probe, fly
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from px4_msgs.msg import TrajectorySetpoint, VehicleLocalPosition
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image

from dronerl import contract

WORLD = "empty"
YAW = contract.YAW_RATE_MAX
# one repetition of the steps: (v_fwd, v_lat, yaw_rate) for seconds, a rest at zero after each
STEPS = [
    ((1.0, 0.0, 0.0), 2.0),
    ((0.5, 0.0, 0.0), 3.0),
    ((-0.3, 0.0, 0.0), 3.0),
    ((0.0, 0.5, 0.0), 2.0),
    ((0.0, -0.5, 0.0), 2.0),
    ((0.0, 0.0, YAW), 1.5),
    ((0.0, 0.0, -YAW), 1.5),
]
REST = 2.0
REPS = 5
TURN = math.pi / YAW  # s of a half turn at the end of a repetition, the next one flies back
LATENCY_PAIRS = (0, 1, 2)


def schedule():
    """[(start, end, command)] from the start of the steps; the half turns alternate in sign."""
    out, t = [], 0.0
    for rep in range(REPS):
        turn = ((0.0, 0.0, YAW if rep % 2 == 0 else -YAW), TURN)
        for cmd, length in [*STEPS, turn]:
            out.append((t, t + length, cmd))
            t += length + REST
    return out


class Steps(Probe):
    """Yaws towards the edge with the most room, then flies the schedule. Records the ground truth,
    the estimate, its own commands and the setpoints of px4_bridge."""

    launch = {"policy": "false"}

    def __init__(self, pair):
        super().__init__(WORLD, pair)
        start, size = self.map["start"], self.map["size"]
        room = [size[0] - start[0], size[1] - start[1], start[0], start[1]]
        self.heading = int(np.argmax(room)) * math.pi / 2
        self.plan, self.begin, self.yaw = schedule(), None, None
        self.gt, self.ekf, self.cmds, self.setpoints = [], [], [], []
        self.create_subscription(Odometry, "/ground_truth/odometry", self.on_odometry, 50)
        self.create_subscription(
            VehicleLocalPosition, common.POSITION, self.on_position, common.PX4_OUT
        )
        self.create_subscription(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", self.on_setpoint, common.PX4_OUT
        )

    def on_odometry(self, msg):
        t = common.seconds(msg.header.stamp)
        if self.gt and t <= self.gt[-1][0]:
            return
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
        self.gt.append((t, p.x, p.y, p.z, q.w, q.x, q.y, q.z))

    def on_position(self, msg):
        self.ekf.append((self.now(), msg.x, msg.y, msg.z, msg.vx, msg.vy, msg.vz, msg.heading))

    def on_setpoint(self, msg):
        self.setpoints.append((self.now(), *msg.velocity[:2], msg.yawspeed))

    def tick(self):
        if self.phase != "fly" or self.yaw is None:
            return
        t = self.now()
        if self.begin is None:
            err = math.remainder(self.heading - self.yaw, math.tau)
            if abs(err) > 0.05:
                self.command((0.0, 0.0, float(np.clip(2 * err, -YAW, YAW))))
                return
            self.begin = t + REST
        k = t - self.begin
        if k > self.plan[-1][1] + REST:
            return  # the end: without commands mission_node lands
        cmd = next((c for t0, t1, c in self.plan if t0 <= k < t1), (0.0, 0.0, 0.0))
        self.command(cmd)

    def command(self, cmd):
        self.send(cmd)
        self.cmds.append((self.now(), *cmd))

    def save(self, out):
        # the steps as sent: a tick of 50 ms late against the schedule would read as delay
        plan, prev = [], (0.0, 0.0, 0.0)
        for t, *cmd in self.cmds:
            cmd = tuple(cmd)
            if t < self.begin or cmd == prev:
                continue
            if any(prev):
                plan[-1][1] = t  # a rest follows every step
            if any(cmd):
                plan.append([t, None, *cmd])
            prev = cmd
        plan = np.array([row for row in plan if row[1] is not None])
        np.savez(
            out / "steps.npz",
            gt=np.array(self.gt),
            ekf=np.array(self.ekf),
            cmds=np.array(self.cmds),
            setpoints=np.array(self.setpoints),
            plan=plan,
            start=self.map["start"],
        )


class Latency(Probe):
    """A policy flight: arrival of the student images, commands and setpoints in simulation and
    wall time, and /clock to put the camera frames on the wall clock."""

    def __init__(self, pair):
        super().__init__(WORLD, pair)
        self.images, self.cmds, self.setpoints, self.clock = [], [], [], []
        self.create_subscription(Image, "/dronerl/student_image", self.on_image, common.LATEST)
        self.create_subscription(TwistStamped, "/dronerl/cmd", self.on_cmd, 10)
        self.create_subscription(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", self.on_setpoint, common.PX4_OUT
        )
        self.create_subscription(Clock, "/clock", self.on_clock, qos_profile_sensor_data)

    def arrival(self):
        return self.now(), time.time()

    def on_image(self, msg):
        self.images.append((common.seconds(msg.header.stamp), *self.arrival()))

    def on_cmd(self, msg):
        self.cmds.append((common.seconds(msg.header.stamp), *self.arrival()))

    def on_setpoint(self, msg):
        self.setpoints.append((msg.timestamp * 1e-6, *self.arrival()))

    def on_clock(self, msg):
        t = common.seconds(msg.clock)
        if not self.clock or t - self.clock[-1][0] >= 0.01:
            self.clock.append((t, time.time()))

    def save(self, out):
        arrays = {k: np.array(getattr(self, k)) for k in ("images", "cmds", "setpoints", "clock")}
        np.savez(out / "latency.npz", **arrays)


def run(probe, pair, out, **args):
    out.mkdir(parents=True)
    fly(probe, out, {"world": WORLD, "pair": pair, "gui": "false", **probe.launch, **args})
    probe.save(out)
    probe.destroy_node()


def main():
    out = Path(sys.argv[1])
    kinds = sys.argv[2:] or ["steps", "latency"]
    rclpy.init()
    if "steps" in kinds:
        # the steps take minutes, the monitor would end the flight after the 60 s of an episode
        params = {"monitor_node": {"ros__parameters": {"max_time": 600.0}}}
        out.mkdir(parents=True, exist_ok=True)
        (out / "steps.yaml").write_text(yaml.safe_dump(params))
        run(Steps(0), 0, out / "steps", params=out / "steps.yaml")
    if "latency" in kinds:
        for pair in LATENCY_PAIRS:
            run(Latency(pair), pair, out / f"latency_p{pair}")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
