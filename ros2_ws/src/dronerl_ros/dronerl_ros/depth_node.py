"""Camera -> student image (plan/M4.md). Source `rgb`: the RGB image through the depth model in
ONNX, `depth`: the depth camera of the simulation as ideal perception. Both end in the reduction
and encoding of M3 and go out on /dronerl/student_image (32FC2, 48 x 64) with the stamp of the
camera frame.

Frames are taken at the control rate, the newest one once 0.1 s of camera time has passed, so
policy_node can act on each image as soon as it comes with the frame spacing of NavEnv.
"""

import math
import time

import numpy as np
import torch
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32

from dronerl.perception.depth_model import OnnxDepth, student_depth, student_image
from dronerl_ros import common

CAMERAS = {"rgb": "/camera", "depth": "/camera/depth"}


def pixels(msg, dtype, channels):
    """Image message -> array [H, W, channels], rows may be padded to `step`."""
    row = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    size = msg.width * channels * np.dtype(dtype).itemsize
    return row[:, :size].copy().view(dtype).reshape(msg.height, msg.width, channels)


class DepthNode(Node):
    def __init__(self):
        super().__init__("depth_node")
        self.source = self.declare_parameter("source", "rgb").value
        model = self.declare_parameter("model", "runs/m3/onnx/depth_ft_full_r518.onnx").value
        device = self.declare_parameter("device", "cuda").value
        self.period = 1 / self.declare_parameter("rate", 10.0).value
        if self.source not in CAMERAS:
            raise ValueError(f"source {self.source}, expected one of {list(CAMERAS)}")
        self.model = OnnxDepth(model, device=device) if self.source == "rgb" else None
        self.fx, self.due = None, -math.inf
        camera = CAMERAS[self.source]
        self.create_subscription(CameraInfo, f"{camera}/camera_info", self.on_info, 1)
        self.create_subscription(Image, f"{camera}/image", self.on_image, common.LATEST)
        self.image_pub = self.create_publisher(Image, "/dronerl/student_image", 1)
        self.time_pub = self.create_publisher(Float32, "/dronerl/depth_ms", 10)

    def on_info(self, msg):
        self.fx = msg.k[0]

    def on_image(self, msg):
        stamp = common.seconds(msg.header.stamp)
        if self.fx is None or stamp < self.due:
            return
        # a few ms of slack: camera frames are 1/30 s apart, every third one is due
        self.due = stamp + self.period - 0.01
        t = time.perf_counter()
        if self.source == "rgb":
            if msg.encoding != "rgb8":
                raise ValueError(f"camera encoding {msg.encoding}, expected rgb8")
            image = student_image(self.model, pixels(msg, np.uint8, 3), self.fx)
        else:
            depth = torch.from_numpy(pixels(msg, np.float32, 1)[..., 0])
            image = student_depth(depth, self.fx)
        out = Image(header=msg.header, height=image.shape[1], width=image.shape[2])
        out.encoding, out.step = "32FC2", image.shape[2] * 8
        out.data = image.permute(1, 2, 0).contiguous().numpy().tobytes()
        self.image_pub.publish(out)
        self.time_pub.publish(Float32(data=1000 * (time.perf_counter() - t)))


def main():
    common.run(DepthNode, threads=4)  # the resize before the model
