"""The only node that writes to PX4 (plan/M4.md): OffboardControlMode and TrajectorySetpoint
every 50 ms and at once for a new command.

While the policy flies, the setpoint is the safe command as a NED velocity with the height held
by PX4. A command turns into NED once, with the heading expected when PX4 answers it `lead`
seconds later, as NavEnv turns a command with the yaw at its arrival (M4.12). Before that, the
hold setpoints of mission_node pass through. Without a fresh input the stream stops and PX4 runs
its offboard loss failsafe.
"""

import math

from geometry_msgs.msg import PointStamped, TwistStamped
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleLocalPosition,
    VehicleOdometry,
)
from rclpy.node import Node

from dronerl.runtime import frames
from dronerl_ros import common

NAN = math.nan


class Px4Bridge(Node):
    def __init__(self):
        super().__init__("px4_bridge")
        rate = self.declare_parameter("rate", 20.0).value
        self.timeout = self.declare_parameter("timeout", 0.5).value
        self.lead = self.declare_parameter("lead", 0.26).value
        self.heading = self.z = self.setpoint = self.key = None
        self.rate = 0.0
        self.velocity, self.last = False, -math.inf

        self.create_subscription(TwistStamped, "/dronerl/safe_cmd", self.on_cmd, 10)
        self.create_subscription(TrajectorySetpoint, "/dronerl/hold", self.on_hold, 10)
        self.create_subscription(
            VehicleLocalPosition, common.POSITION, self.on_position, common.PX4_OUT
        )
        self.create_subscription(VehicleOdometry, common.ODOMETRY, self.on_odometry, common.PX4_OUT)
        self.create_subscription(PointStamped, "/dronerl/origin", self.on_origin, common.LATCHED)
        self.mode_pub = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", common.PX4_IN
        )
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", common.PX4_IN
        )
        self.create_timer(1 / rate, self.tick)

    def on_position(self, msg):
        self.heading = msg.heading

    def on_odometry(self, msg):
        self.rate = float(msg.angular_velocity[2])

    def on_origin(self, msg):
        self.z = msg.point.z  # flight height as a local NED z

    def on_hold(self, msg):
        self.setpoint, self.velocity, self.key = msg, False, None
        self.last = common.now(self)

    def on_cmd(self, msg):
        if self.heading is None or self.z is None:
            return
        cmd = common.command(msg)
        # safety_node repeats a command at 20 Hz; only a new one is turned with the heading
        key = (msg.header.stamp.sec, msg.header.stamp.nanosec, *cmd)
        self.last = common.now(self)
        if key != self.key:
            heading = self.heading + self.rate * self.lead
            vel, yawspeed = frames.command_to_ned(cmd, heading)
            sp = TrajectorySetpoint()
            sp.position = [NAN, NAN, self.z]
            sp.velocity = [float(vel[0]), float(vel[1]), NAN]
            sp.acceleration = [NAN] * 3
            sp.yaw, sp.yawspeed = NAN, float(yawspeed)
            self.setpoint, self.velocity, self.key = sp, True, key
            self.tick()

    def tick(self):
        if self.setpoint is None or common.now(self) - self.last > self.timeout:
            return
        t = common.px4_time(self)
        mode = OffboardControlMode(timestamp=t, position=True, velocity=self.velocity)
        self.setpoint.timestamp = t
        self.mode_pub.publish(mode)
        self.setpoint_pub.publish(self.setpoint)


def main():
    common.run(Px4Bridge)
