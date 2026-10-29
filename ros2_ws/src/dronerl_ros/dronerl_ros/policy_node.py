"""The student in ONNX (plan/M4.md): the frame stack and the state of dronerl.runtime from the
PX4 estimate, the command of the contract on /dronerl/cmd with the stamp of the camera frame it
came from. Acts on each student image, which depth_node sends at 10 Hz, and only while
mission_node is in the phase "fly".
"""

import numpy as np
import onnxruntime as ort
import torch
from geometry_msgs.msg import PointStamped, TwistStamped
from px4_msgs.msg import VehicleLocalPosition, VehicleOdometry
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String

from dronerl import contract
from dronerl.runtime import frames
from dronerl.runtime.observation import Observation
from dronerl_ros import common


class PolicyNode(Node):
    def __init__(self):
        super().__init__("policy_node")
        model = self.declare_parameter("model", "").value
        world = self.declare_parameter("world", "empty").value
        pair = self.declare_parameter("pair", 0).value
        # older images give no command and the watchdog of safety_node stops the drone
        self.max_age = self.declare_parameter("max_age", 0.5).value
        # the student takes 0.1 ms on one thread, idle threads would spin on all cores
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(model, opts, providers=["CPUExecutionProvider"])
        self.map = common.flight(world, pair)
        self.obs = self.position = self.rate = None
        self.phase = "wait"

        self.create_subscription(Image, "/dronerl/student_image", self.on_image, common.LATEST)
        self.create_subscription(
            VehicleLocalPosition, common.POSITION, self.on_position, common.PX4_OUT
        )
        self.create_subscription(VehicleOdometry, common.ODOMETRY, self.on_odometry, common.PX4_OUT)
        self.create_subscription(String, "/dronerl/phase", self.on_phase, common.LATCHED)
        self.create_subscription(PointStamped, "/dronerl/origin", self.on_origin, common.LATCHED)
        self.cmd_pub = self.create_publisher(TwistStamped, "/dronerl/cmd", 10)
        self.action_pub = self.create_publisher(Float32MultiArray, "/dronerl/action", 10)
        self.state_pub = self.create_publisher(Float32MultiArray, "/dronerl/state", 10)

    def on_image(self, msg):
        x = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width, 2)
        self.step(torch.from_numpy(x.copy()).permute(2, 0, 1), msg.header.stamp)

    def on_position(self, msg):
        self.position = msg

    def on_odometry(self, msg):
        self.rate = float(msg.angular_velocity[2])

    def on_phase(self, msg):
        self.phase = msg.data

    def on_origin(self, msg):
        origin = np.array([msg.point.x, msg.point.y])
        goal = frames.map_to_ned(self.map["goal"], self.map["start"], origin)
        self.obs = Observation(goal)
        self.get_logger().info(f"goal at {goal.round(2)} in local NED")

    def step(self, image, stamp):
        if self.phase != "fly" or None in (self.obs, self.position, self.rate):
            return
        if common.now(self) - common.seconds(stamp) > self.max_age:
            return
        p = self.position
        pos, vel = np.array([p.x, p.y, p.z]), np.array([p.vx, p.vy, p.vz])
        state = self.obs.state(pos, vel, p.heading, self.rate, p.dist_bottom)
        feed = {"frames": self.obs.image(image)[None].numpy(), "state": state[None].numpy()}
        action = torch.from_numpy(self.session.run(None, feed)[0][0]).clamp(-1, 1)
        self.obs.acted(action)
        self.cmd_pub.publish(common.twist(contract.action_to_command(action), stamp))
        self.action_pub.publish(Float32MultiArray(data=action.tolist()))
        self.state_pub.publish(Float32MultiArray(data=state.tolist()))


def main():
    common.run(PolicyNode)
