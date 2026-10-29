"""Course of one flight (plan/M4.md): wait for the estimate and give the EKF a global origin,
offboard takeoff to the flight height, hover, the policy until monitor_node ends the flight,
landing. Publishes the phase and, at the start of the policy, the origin: the local NED point of
the start of the map, with the flight height as its z. Exits once the drone is disarmed, which
ends the launch.

The height is set above the floor seen by the rangefinder, not at a fixed local z: in the air the
local z of EKF2 drifts some 0.2 m from the floor, so the floor level is tracked until the policy
starts.
"""

import math

from geometry_msgs.msg import PointStamped, TwistStamped
from px4_msgs.msg import TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus
from rclpy.node import Node
from std_msgs.msg import String

from dronerl_ros import common

NAN = math.nan
OFFBOARD = VehicleStatus.NAVIGATION_STATE_OFFBOARD
# simulation seconds a phase may take; wait covers the start of Gazebo and PX4
TIMEOUTS = {"wait": 120.0, "takeoff": 30.0, "hover": 10.0, "land": 60.0, "pilot": 120.0}
FLOOR_ALPHA = 0.1  # per tick of 0.1 s, the floor estimate settles in about 1 s
# any fixed point (lat, lon, alt): without GPS it only gives PX4 the global frame of its geofence
ORIGIN = (47.397742, 8.545594, 488.0)


class MissionNode(Node):
    def __init__(self):
        super().__init__("mission_node")
        self.height = self.declare_parameter("height", 1.2).value
        self.hover = self.declare_parameter("hover", 2.0).value
        self.policy_timeout = self.declare_parameter("policy_timeout", 1.0).value
        self.status = self.position = self.hold = self.result = None
        self.last_cmd = self.sent = self.ready = -math.inf
        self.failed = False

        self.create_subscription(VehicleStatus, common.STATUS, self.on_status, common.PX4_OUT)
        self.create_subscription(
            VehicleLocalPosition, common.POSITION, self.on_position, common.PX4_OUT
        )
        self.create_subscription(String, "/dronerl/result", self.on_result, common.LATCHED)
        self.create_subscription(TwistStamped, "/dronerl/cmd", self.on_cmd, 10)
        self.command_pub = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", common.PX4_IN
        )
        self.hold_pub = self.create_publisher(TrajectorySetpoint, "/dronerl/hold", 10)
        self.phase_pub = self.create_publisher(String, "/dronerl/phase", common.LATCHED)
        self.origin_pub = self.create_publisher(PointStamped, "/dronerl/origin", common.LATCHED)
        self.set_phase("wait")
        self.create_timer(0.1, self.tick)

    def on_status(self, msg):
        self.status = msg

    def on_position(self, msg):
        self.position = msg

    def on_result(self, msg):
        self.result = msg.data

    def on_cmd(self, msg):
        self.last_cmd = common.now(self)

    def set_phase(self, phase):
        self.phase, self.since = phase, common.now(self)
        self.phase_pub.publish(String(data=phase))
        self.get_logger().info(f"phase: {phase}")

    def command(self, command, *params):
        msg = VehicleCommand(timestamp=common.px4_time(self), command=command, from_external=True)
        msg.target_system = msg.target_component = msg.source_system = msg.source_component = 1
        for i, value in enumerate(params, 1):
            setattr(msg, f"param{i}", float(value))
        self.command_pub.publish(msg)
        self.sent = common.now(self)

    def publish_hold(self):
        x, y, z, yaw = self.hold
        sp = TrajectorySetpoint(timestamp=common.px4_time(self), yaw=yaw, yawspeed=NAN)
        sp.position, sp.velocity, sp.acceleration = [x, y, z], [NAN] * 3, [NAN] * 3
        self.hold_pub.publish(sp)

    def start_offboard(self):
        self.command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1, 6)  # custom main mode offboard
        self.command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1)

    def land(self):
        self.command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.set_phase("land")

    def finish(self):
        self.set_phase("done")
        raise SystemExit(1 if self.failed else 0)

    def tick(self):
        s, p = self.status, self.position
        if s is None or p is None:
            return
        t = common.now(self)
        if t - self.since > TIMEOUTS.get(self.phase, math.inf):
            self.get_logger().error(f"phase {self.phase} timed out")
            self.failed = True
            if s.arming_state == VehicleStatus.ARMING_STATE_ARMED and self.phase != "land":
                self.land()
            else:
                self.finish()
            return

        # dist_bottom_valid stays false with the rangefinder as the height reference
        if self.phase in ("takeoff", "hover"):
            self.floor += FLOOR_ALPHA * (p.z + p.dist_bottom - self.floor)
            self.hold = (*self.hold[:2], self.floor - self.height, self.hold[3])

        if self.phase == "wait":
            # pre_flight_checks_pass is for the current mode, Hold, which needs GPS
            if not (p.xy_valid and p.z_valid):
                return
            if not p.xy_global:
                if t - self.sent > 1.0:
                    origin = VehicleCommand.VEHICLE_CMD_SET_GPS_GLOBAL_ORIGIN
                    self.command(origin, 0, 0, 0, 0, *ORIGIN)
                return
            if self.hold is None:
                # z grows downwards, the floor is dist_bottom below the drone
                self.floor = p.z + p.dist_bottom
                self.hold = (p.x, p.y, self.floor - self.height, p.heading)
                self.ready = t
            self.publish_hold()
            # PX4 switches to offboard only when the setpoints are already streaming
            if t - self.ready > 1.0:
                self.start_offboard()
                self.set_phase("takeoff")
        elif self.phase == "takeoff":
            self.publish_hold()
            armed = s.arming_state == VehicleStatus.ARMING_STATE_ARMED
            if not armed or s.nav_state != OFFBOARD:
                if t - self.sent > 1.0:
                    self.start_offboard()
            elif abs(p.z - self.hold[2]) < 0.1 and abs(p.vz) < 0.1:
                self.set_phase("hover")
        elif self.phase == "hover":
            self.publish_hold()
            if t - self.since >= self.hover:
                origin = PointStamped()
                origin.header.stamp = self.get_clock().now().to_msg()
                origin.point.x, origin.point.y, origin.point.z = self.hold[:3]
                self.origin_pub.publish(origin)
                self.last_cmd = t
                self.set_phase("fly")
        elif self.phase == "fly":
            if s.nav_state != OFFBOARD:
                # the pilot or a failsafe of PX4 took over, the mission does not take it back
                self.get_logger().warn(f"offboard lost, navigation state {s.nav_state}")
                self.set_phase("pilot")
            elif self.result is not None:
                self.land()
            elif t - self.last_cmd > self.policy_timeout:
                self.get_logger().warn("no command from the policy")
                self.failed = True
                self.land()
        elif self.phase in ("land", "pilot"):
            if s.arming_state == VehicleStatus.ARMING_STATE_DISARMED:
                self.finish()
            land = s.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LAND
            if self.phase == "land" and not land and t - self.sent > 1.0:
                self.command(VehicleCommand.VEHICLE_CMD_NAV_LAND)


def main():
    common.run(MissionNode)
