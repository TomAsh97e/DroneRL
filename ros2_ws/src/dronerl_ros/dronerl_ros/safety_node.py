"""Safety filter between the policy and PX4 (plan/M4.md): dronerl.runtime.safety on the PX4
estimate while mission_node is in the phase "fly", at once for a new command and at 20 Hz for the
watchdog and the fence. The safe command goes to px4_bridge on /dronerl/safe_cmd, every change of
the reason ("ok", "watchdog", "height", "fence") to /dronerl/safety.
"""

import numpy as np
from geometry_msgs.msg import PointStamped, TwistStamped
from px4_msgs.msg import VehicleLocalPosition
from rclpy.node import Node
from std_msgs.msg import String

from dronerl.runtime import frames, safety
from dronerl_ros import common


class SafetyNode(Node):
    def __init__(self):
        super().__init__("safety_node")
        world = self.declare_parameter("world", "empty").value
        pair = self.declare_parameter("pair", 0).value
        self.fence = self.declare_parameter("fence", True).value
        margin = self.declare_parameter("margin", safety.FENCE_MARGIN).value
        rate = self.declare_parameter("rate", 20.0).value
        self.map = common.flight(world, pair)
        self.filter = safety.SafetyFilter(margin=margin)
        self.position = self.stamp = None
        self.phase, self.reason = "wait", None

        self.create_subscription(TwistStamped, "/dronerl/cmd", self.on_cmd, 10)
        self.create_subscription(
            VehicleLocalPosition, common.POSITION, self.on_position, common.PX4_OUT
        )
        self.create_subscription(String, "/dronerl/phase", self.on_phase, common.LATCHED)
        self.create_subscription(PointStamped, "/dronerl/origin", self.on_origin, common.LATCHED)
        self.cmd_pub = self.create_publisher(TwistStamped, "/dronerl/safe_cmd", 10)
        self.reason_pub = self.create_publisher(String, "/dronerl/safety", 10)
        self.create_timer(1 / rate, self.tick)

    def on_cmd(self, msg):
        self.filter.command(common.command(msg), common.now(self))
        self.stamp = msg.header.stamp
        self.tick()

    def on_position(self, msg):
        self.position = msg

    def on_phase(self, msg):
        self.phase = msg.data

    def on_origin(self, msg):
        if not self.fence:
            return
        corners = np.array([[0.0, 0.0], self.map["size"]])
        ned = frames.map_to_ned(corners, self.map["start"], [msg.point.x, msg.point.y])
        self.filter.box = (*ned.min(0), *ned.max(0))
        self.get_logger().info(f"fence {np.round(self.filter.box, 2)} in local NED")

    def tick(self):
        if self.phase != "fly" or self.position is None:
            return
        p = self.position
        pos = np.array([p.x, p.y, p.z])
        cmd, reason = self.filter(common.now(self), pos, p.heading, p.dist_bottom, [p.vx, p.vy])
        stamp = self.get_clock().now().to_msg() if self.stamp is None else self.stamp
        self.cmd_pub.publish(common.twist(cmd, stamp))
        if reason != self.reason:
            self.reason = reason
            self.reason_pub.publish(String(data=reason))
            self.get_logger().info(f"safety: {reason}")


def main():
    common.run(SafetyNode)
