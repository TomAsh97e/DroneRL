"""Geometry of the Gazebo worlds against NavEnv (M4.4), run in the container:

    docker/run.sh python3 sim/check_parity.py

Depth from the Gazebo depth camera, cropped and reduced as depth_node does it, against the ray
casting of NavEnv at the same camera poses, level and tilted. This catches sign, axis and field
of view mistakes before they look like a sim2sim gap. Criterion as in M3.7: in every world at
least 98% of the pixels at least 1 px from a depth step have a relative error below 1%.
"""

import argparse
import json
import math
import os
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.entity_factory_pb2 import EntityFactory
from gz.msgs10.image_pb2 import Image
from gz.msgs10.pose_pb2 import Pose
from gz.transport13 import Node

from dronerl import contract
from dronerl.envs import maps as mapgen
from dronerl.envs import raycast
from dronerl.envs.depth_noise import edge_distance
from dronerl.envs.nav_gpu import NavEnv
from dronerl.perception.depth_model import crop_fov, reduce_depth

REPO = Path(__file__).resolve().parents[1]
WORLDS = REPO / "sim/gazebo/worlds"
PLUGINS = REPO / "external/PX4-Autopilot/build/px4_sitl_default/src/modules/simulation/gz_plugins"
FX = 320 / math.tan(math.radians(100) / 2)  # 640 px over HFOV 100°, checked in M4.3
MAX_TILT = math.radians(20)
MIN_SHARE = 0.98


class Probe:
    """The depth_probe model in a running world: moves it and returns the next images."""

    def __init__(self, world):
        self.world, self.node = world, Node()
        self.images, self.cond = {}, threading.Condition()
        for key, topic in (("depth", "/probe/depth/image"), ("rgb", "/probe/image")):
            self.node.subscribe(Image, topic, lambda m, k=key: self.store(k, m))
        factory = EntityFactory(sdf_filename="model://depth_probe", name="depth_probe")
        self.call("create", factory, EntityFactory, timeout=60)

    def store(self, key, msg):
        with self.cond:
            self.images[key] = msg
            self.cond.notify_all()

    def call(self, service, request, kind, timeout=10):
        end = time.time() + timeout
        while time.time() < end:
            name = f"/world/{self.world}/{service}"
            ok, reply = self.node.request(name, request, kind, Boolean, 1000)
            if ok and reply.data:
                return
            time.sleep(0.5)
        raise TimeoutError(f"{service} in {self.world}")

    def stamp(self, key):
        s = self.images[key].header.stamp
        return s.sec + s.nsec * 1e-9

    def look(self, x, y, z, yaw, pitch, roll):
        """Depth [H, W] and RGB [H, W, 3] rendered after the probe was moved to the pose."""
        with self.cond:
            self.cond.wait_for(lambda: "depth" in self.images and "rgb" in self.images, 60)
            t = max(self.stamp("depth"), self.stamp("rgb"))
        pose = Pose(name="depth_probe")
        pose.position.x, pose.position.y, pose.position.z = x, y, z
        q = pose.orientation
        q.w, q.x, q.y, q.z = quaternion(yaw, pitch, roll)
        self.call("set_pose", pose, Pose)

        def ready():  # two frames later both cameras surely rendered the new pose
            return min(self.stamp("depth"), self.stamp("rgb")) > t + 2.5 / 30

        with self.cond:
            if not self.cond.wait_for(ready, 30):
                raise TimeoutError("no image after set_pose")
            d, c = self.images["depth"], self.images["rgb"]
        depth = np.frombuffer(d.data, np.float32).reshape(d.height, d.width)
        rgb = np.frombuffer(c.data, np.uint8).reshape(c.height, c.width, 3)
        return depth.copy(), rgb.copy()


def quaternion(yaw, pitch, roll):
    """(w, x, y, z) of R_z(yaw) R_y(pitch) R_x(roll); pitch down and right side down positive, as
    in NavEnv, are positive rotations about y (left) and x (forward) in FLU as well."""
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def poses(m, rng, n, level):
    """n camera poses in free space, the first `level` of them with no tilt."""
    out = []
    while len(out) < n:
        p = rng.uniform(0.5, m["size"] - 0.5)
        if mapgen.sdf(p[None], m["circles"], m["boxes"])[0] < contract.COLLISION_DIST:
            continue
        tilt = (0.0, 0.0) if len(out) < level else rng.uniform(-MAX_TILT, MAX_TILT, 2)
        out.append((*p, rng.uniform(1.0, 1.5), rng.uniform(-np.pi, np.pi), *tilt))
    return out


def compare(gz_depth, env, pose):
    """Relative error of the reduced Gazebo depth against NavEnv and the mask away from edges."""
    x, y, z, yaw, pitch, roll = pose
    image = reduce_depth(crop_fov(torch.from_numpy(gz_depth), FX), (contract.IMG_H, contract.IMG_W))
    origin, heading = torch.tensor([[x, y]]), torch.tensor([[yaw]])
    fan = raycast.cast(origin, heading + raycast.FAN_ANGLES, env.obstacles())
    t = torch.tensor
    nav = raycast.tilted_depth(fan, t([z]), env.env["obstacle_height"], t([pitch]), t([roll]))[0]
    gz, nav = (d.double().clamp(max=contract.DEPTH_MAX) for d in (image, nav))
    error = (nav - gz).abs() / gz
    away = edge_distance(gz[None], 1)[0] >= 1
    return error, away, gz, nav


def side_by_side(gz, nav, rgb):
    """RGB, reduced Gazebo depth and NavEnv depth as one image for a quick look."""

    def color(d):
        inv = contract.encode_depth(d.float()).numpy()
        big = cv2.resize((255 * inv).astype(np.uint8), (320, 240), interpolation=cv2.INTER_NEAREST)
        return cv2.applyColorMap(big, cv2.COLORMAP_INFERNO)

    rgb = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (320, 240))
    return np.concatenate([rgb, color(gz), color(nav)], 1)


def check_world(name, n, level, out, rng):
    sdf = WORLDS / name / "world.sdf"
    with np.load(WORLDS / name / "maps.npz") as f:
        pool = dict(f)
    m = mapgen.get_map(pool, 0)
    env = NavEnv(mapgen.collate([m]), device="cpu", autoreset=False)
    env.reset()

    gz_env = dict(os.environ)
    gz_env["GZ_SIM_RESOURCE_PATH"] = str(REPO / "sim/gazebo/models")
    gz_env["GZ_SIM_SYSTEM_PLUGIN_PATH"] = str(PLUGINS)
    log = open(out / f"gz_{name}.log", "w")
    server = subprocess.Popen(
        ["gz", "sim", "-s", "-r", "--headless-rendering", str(sdf)],
        env=gz_env,
        stdout=log,
        stderr=log,
    )
    try:
        probe = Probe(name)
        rows = []
        for k, pose in enumerate(poses(m, rng, n, level)):
            depth, rgb = probe.look(*pose)
            error, away, gz, nav = compare(depth, env, pose)
            good, total = int((error[away] < 0.01).sum()), int(away.sum())
            rows.append({"pose": [float(v) for v in pose], "good": good, "away": total})
            if k < 4 or good < MIN_SHARE * total:
                cv2.imwrite(str(out / f"{name}_{k:02d}.png"), side_by_side(gz, nav, rgb))
    finally:
        server.terminate()
        server.wait()
        log.close()
    good, total = (np.array([r[k] for r in rows]) for k in ("good", "away"))
    return {"poses": rows, "share": good.sum() / total.sum(), "share_min": (good / total).min()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worlds", nargs="+", default=["arena", "corridor", "room"])
    parser.add_argument("--poses", type=int, default=40)
    parser.add_argument("--level", type=int, default=10, help="poses without tilt")
    parser.add_argument("--out", type=Path, default=REPO / "runs/m4/checks/parity")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    results = {}
    for name in args.worlds:
        results[name] = check_world(name, args.poses, args.level, args.out, rng)
        r = results[name]
        print(f"{name}: share {r['share']:.4f}, worst pose {r['share_min']:.4f}", flush=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    ok = all(r["share"] >= MIN_SHARE for r in results.values())
    print("OK" if ok else "FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
