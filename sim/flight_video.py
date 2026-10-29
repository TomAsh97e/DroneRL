"""Video of a recorded flight (plan/M4.md, M4.11), inside the container: the RGB camera next to
the student image (depth, near bright, missing pixels black) with the time and the command,
from the bag of sim.launch.py. Writes video.mp4 into each flight directory.

    python3 sim/flight_video.py <flight dir> [...]
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from dronerl import contract

TOPICS = (
    "/camera/image/compressed",
    "/dronerl/student_image",
    "/dronerl/safe_cmd",
    "/dronerl/phase",
)
SCALE = 5  # of the 64 x 48 student image


def read(bag):
    reader = rosbag2_py.SequentialReader()
    storage = rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap")
    reader.open(storage, rosbag2_py.ConverterOptions("", ""))
    types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(TOPICS)))
    while reader.has_next():
        topic, data, _ = reader.read_next()
        yield topic, deserialize_message(data, types[topic])


def student(msg):
    image = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width, 2)
    # metric depth as in the depth strips of nav_plots, near is bright
    near = 1 - contract.decode_depth(image[..., 0]) / contract.DEPTH_MAX
    color = cv2.applyColorMap((255 * near).astype(np.uint8), cv2.COLORMAP_MAGMA)
    color[image[..., 1] == 0] = 0
    size = (msg.width * SCALE, msg.height * SCALE)
    return cv2.resize(color, size, interpolation=cv2.INTER_NEAREST)


def video(flight):
    result = json.loads((flight / "result.json").read_text())
    writer, side, cmd, phase, start = None, None, np.zeros(3), "wait", None
    for topic, msg in read(flight / "bag"):
        if topic == "/dronerl/phase":
            phase = msg.data
        elif topic == "/dronerl/safe_cmd":
            cmd = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z])
        elif topic == "/dronerl/student_image":
            side = student(msg)
        elif phase in ("fly", "land") and side is not None:
            frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            start = t if start is None else start
            panel = np.zeros((frame.shape[0], side.shape[1], 3), np.uint8)
            panel[: side.shape[0]] = side
            lines = [
                flight.parent.name,
                f"{flight.name}: {result['outcome']}",
                f"t {t - start:5.1f} s, {phase}",
                f"v_fwd {cmd[0]:+.2f} m/s",
                f"v_lat {cmd[1]:+.2f} m/s",
                f"yaw {np.degrees(cmd[2]):+.0f} deg/s",
            ]
            for i, text in enumerate(lines):
                y = side.shape[0] + 30 + 24 * i
                cv2.putText(panel, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            out = np.hstack([frame, panel])
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                size = out.shape[1], out.shape[0]
                writer = cv2.VideoWriter(str(flight / "video.mp4"), fourcc, contract.CTRL_HZ, size)
            writer.write(out)
    if writer is not None:
        writer.release()


def main():
    for flight in map(Path, sys.argv[1:]):
        video(flight)
        print(flight / "video.mp4", flush=True)


if __name__ == "__main__":
    main()
