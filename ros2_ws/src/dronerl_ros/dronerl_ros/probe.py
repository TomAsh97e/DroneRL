"""A node that a script runs beside one flight of sim.launch.py (sim/safety_tests.py,
sim/sysid_flights.py): it watches the flight and may command the drone in place of the policy;
`fly` runs the launch around it."""

import faulthandler
import subprocess
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import TwistStamped
from px4_msgs.msg import VehicleCommand
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String

from dronerl_ros import common

# what sim.launch.py starts; a part slow to exit is killed with sitl.sh and lives on, and the next
# flight gets two simulations at once (M4.11)
LEFTOVERS = (
    "ros2 launch dronerl_ros",
    "sitl.sh",
    "gz sim",
    "bin/px4",
    "MicroXRCEAgent",
    "parameter_bridge",
    "lib/dronerl_ros/",
    "ros2 bag record",
)


class Probe(Node):
    """The phases of the flight, its map and the publishers of commands; `tick` runs at 20 Hz.
    Times are simulation seconds."""

    launch = {}  # arguments of sim.launch.py

    def __init__(self, world, pair):
        sim_time = Parameter("use_sim_time", value=True)
        super().__init__("probe", parameter_overrides=[sim_time])
        self.map = common.flight(world, pair)
        self.phases, self.events = [], {}
        self.create_subscription(String, "/dronerl/phase", self.on_phase, common.LATCHED)
        self.cmd_pub = self.create_publisher(TwistStamped, "/dronerl/cmd", 10)
        self.command_pub = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", common.PX4_IN
        )
        self.create_timer(0.05, self.tick)

    @property
    def phase(self):
        return self.phases[-1][1] if self.phases else "wait"

    def on_phase(self, msg):
        self.phases.append((self.now(), msg.data))
        if msg.data == "fly":
            self.events["fly"] = self.now()

    def now(self):
        return common.now(self)

    def send(self, cmd):
        """Command (v_fwd, v_lat, yaw_rate) on /dronerl/cmd, as from policy_node."""
        self.cmd_pub.publish(common.twist(cmd, self.get_clock().now().to_msg()))

    def vehicle_command(self, command, *params):
        msg = VehicleCommand(timestamp=common.px4_time(self), command=command, from_external=True)
        msg.target_system = msg.target_component = msg.source_system = msg.source_component = 1
        for i, value in enumerate(params, 1):
            setattr(msg, f"param{i}", float(value))
        self.command_pub.publish(msg)

    def tick(self):
        pass


def fly(probe, out, args, timeout=600):
    """Runs sim.launch.py with args into the directory out and spins probe until the flight is
    over or timeout seconds of wall time have passed."""
    cmd = ["ros2", "launch", "dronerl_ros", "sim.launch.py", f"out:={out}"]
    cmd += [f"{k}:={v}" for k, v in args.items()]
    # a spin that never returns would stop the series silently, this ends it with the stacks
    faulthandler.dump_traceback_later(timeout + 120, exit=True)
    with open(Path(out) / "launch.log", "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        end = time.monotonic() + timeout
        try:
            while proc.poll() is None and time.monotonic() < end:
                rclpy.spin_once(probe, timeout_sec=0.05)
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(60)
            stop_leftovers()
            faulthandler.cancel_dump_traceback_later()


def stop_leftovers():
    """Kills what is left of a flight and waits until it is gone."""
    for _ in range(20):
        found = [subprocess.run(["pgrep", "-f", p], capture_output=True).stdout for p in LEFTOVERS]
        alive = [p for p, pids in zip(LEFTOVERS, found) if pids]
        if not alive:
            return
        for p in alive:
            subprocess.run(["pkill", "-9", "-f", p])
        time.sleep(0.5)
    raise RuntimeError(f"processes of the flight still running: {alive}")
