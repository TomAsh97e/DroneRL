"""Gazebo worlds from maps of the M2 format (M4.4, plan/M4.md D5).

Obstacles are prisms from the floor to the map height, as in NavEnv: circles become cylinders and
oriented boxes boxes, the outer walls included. Visuals are OBJ meshes with UVs in metres, so the
CC0 textures (sim/gazebo/fetch_textures.sh) repeat at their real size instead of stretching over a
wall; collisions are the exact primitives. A world is one obstacle layout with 20 start-goal pairs,
saved next to it as a map set that NavEnv loads, so every flight in Gazebo has its twin in NavEnv.

    python sim/gazebo/export_world.py [world ...]    # all of WORLDS into sim/gazebo/worlds
"""

import argparse
import math
from pathlib import Path

import numpy as np

from dronerl.envs import maps as mapgen

ROOT = Path(__file__).resolve().parent
PAIRS = 20
SEED = 4_000
SEGMENTS = 64  # cylinder sides, the faces lie within 0.12% of the radius from the circle
FLOOR_MARGIN = 1.0
WORLDS = {
    "arena": (5, 18),
    "corridor": (10, 18),
    "room": (10, 18),
    "empty": (5, 8),
    # layouts of the validation set: Gazebo images for the depth model, away from the test (M4.10)
    "depth_room": (10, 18),
    "depth_corridor": (10, 18),
}  # start-goal distance [m]
# texture and the size of one tile of it [m]
MATERIALS = {
    "floor": ("WoodFloor051", 1.5),
    "walls": ("PaintedPlaster017", 2.0),
    "concrete": ("Concrete034", 2.0),
    "metal": ("PaintedMetal004", 1.0),
    "planks": ("Planks037A", 1.5),
    "cardboard": ("Cardboard004", 1.0),
    "bricks": ("Bricks106A", 1.5),
}


def arena():
    """The arena of the M6 criterion, by hand: 10 x 10 m with 5 obstacles."""
    size = (10.0, 10.0)
    circles = [(5.0, 5.0, 0.3), (7.2, 2.8, 0.2)]
    boxes = [(3.2, 3.0, 0.4, 0.4, 0.4), (6.8, 7.0, 0.5, 0.3, -0.3)]
    boxes += [(3.0, 7.2, 0.9, mapgen.SEGMENT, 0.8)]  # a partition panel
    return {
        "kind": 0,
        "level": 2,  # density closest to its 5 obstacles per 100 m²
        "size": np.array(size),
        "circles": np.array(circles),
        "boxes": np.array(mapgen.outer_walls(*size) + boxes),
        "height": 2.5,
    }


def empty():
    """The arena with its walls only, for the integration test of the nodes (M4.7)."""
    return {**arena(), "level": 0, "circles": np.zeros((0, 3)), "boxes": arena()["boxes"][:4]}


def generated(kind, split="test"):
    """The first map of that kind in the medium test (or another) set of M2."""
    pool = mapgen.load(split, mapgen.TEST_LEVELS["medium"])
    i = next(i for i, k in enumerate(pool["kind"]) if k == (kind == "corridor"))
    return mapgen.get_map(pool, i)


def place_pairs(layout, rng, n=PAIRS, dist=(10, 18)):
    """n maps with the obstacles of layout and new start-goal pairs drawn as in the M2 generator:
    start near one end, goal near the other, both clear of obstacles and solvable for A*. Pairs
    run in both directions, in a square map also across."""
    size, circles, boxes = layout["size"], layout["circles"], layout["boxes"]
    length, width = size
    out = []
    while len(out) < n:
        start = np.array([rng.uniform(1.5, 2.5), rng.uniform(1.0, width - 1.0)])
        goal = np.array([length - rng.uniform(1.5, 2.5), rng.uniform(1.0, width - 1.0)])
        k = rng.integers(4) if length == width else 2 * rng.integers(2)
        start, goal = turn(start, size, k), turn(goal, size, k)
        if not dist[0] <= np.linalg.norm(goal - start) <= dist[1]:
            continue
        if mapgen.sdf(np.stack([start, goal]), circles, boxes).min() < mapgen.START_CLEARANCE:
            continue
        path = mapgen.plan_path(start, goal, size, circles, boxes)
        if path is None:
            continue
        direction = math.atan2(goal[1] - start[1], goal[0] - start[0])
        out.append(
            {
                **{k: layout[k] for k in ("kind", "level", "size", "circles", "boxes", "height")},
                "start": start,
                "goal": goal,
                "yaw0": direction + rng.uniform(-np.pi / 2, np.pi / 2),
                "path": path,
                "path_len": np.linalg.norm(np.diff(path, axis=0), axis=1).sum(),
                "attempts": 0,
            }
        )
    return out


def turn(p, size, k):
    """p turned by k * 90° about the centre of the map; odd k only for square maps."""
    c = size / 2
    for _ in range(k):
        p = c + np.array([c[1] - p[1], p[0] - c[0]])
    return p


class Mesh:
    """Triangles with per-vertex UVs and normals, written as OBJ."""

    def __init__(self, tile):
        self.tile = tile
        self.v, self.vt, self.vn, self.faces = [], [], [], []

    def vertex(self, p, uv, n):
        self.v.append(p)
        self.vt.append(np.asarray(uv) / self.tile)
        self.vn.append(n)
        return len(self.v)

    def quad(self, corners, uvs, normals):
        a, b, c, d = (self.vertex(*x) for x in zip(corners, uvs, normals))
        self.faces += [(a, b, c), (a, c, d)]

    def prism(self, points, height, normals=None):
        """Sides and top of a prism over a counter-clockwise outline [N, 2]. normals [N, 2] at the
        points make the sides smooth, without them every side is flat."""
        u = 0.0
        for i in range(len(points)):
            p, q = points[i], points[(i + 1) % len(points)]
            step = np.linalg.norm(q - p)
            if normals is None:
                n0 = n1 = np.array([q[1] - p[1], p[0] - q[0]]) / step
            else:
                n0, n1 = normals[i], normals[(i + 1) % len(points)]
            corners = [(*p, 0), (*q, 0), (*q, height), (*p, height)]
            uvs = [(u, 0), (u + step, 0), (u + step, height), (u, height)]
            self.quad(corners, uvs, [(*n0, 0), (*n1, 0), (*n1, 0), (*n0, 0)])
            u += step
        top = [self.vertex((*p, height), p, (0, 0, 1)) for p in points]
        self.faces += [(top[0], top[i], top[i + 1]) for i in range(1, len(top) - 1)]

    def box(self, x, y, hx, hy, angle, height):
        a = np.array([math.cos(angle), math.sin(angle)])
        b = np.array([-a[1], a[0]])
        corners = ((-1, -1), (1, -1), (1, 1), (-1, 1))
        self.prism([np.array([x, y]) + i * hx * a + j * hy * b for i, j in corners], height)

    def cylinder(self, x, y, r, height):
        t = np.arange(SEGMENTS) * 2 * np.pi / SEGMENTS
        radial = np.stack([np.cos(t), np.sin(t)], -1)
        self.prism(np.array([x, y]) + r * radial, height, radial)

    def floor(self, x0, y0, x1, y1):
        corners = [(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)]
        self.quad(corners, [c[:2] for c in corners], [(0, 0, 1)] * 4)

    def obj(self):
        lines = [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in self.v]
        lines += [f"vt {u:.5f} {v:.5f}" for u, v in self.vt]
        lines += [f"vn {x:.5f} {y:.5f} {z:.5f}" for x, y, z in self.vn]
        lines += ["f " + " ".join(f"{i}/{i}/{i}" for i in f) for f in self.faces]
        return "\n".join(lines) + "\n"


def material(shape, i, params):
    """Texture variant of the i-th circle or box; partitions are brick walls."""
    if shape == "circle":
        return ("concrete", "metal")[i % 2]
    if min(params[2:4]) <= mapgen.SEGMENT + 1e-9:
        return "bricks"
    return ("planks", "cardboard")[i % 2]


def meshes(layout):
    """Mesh per material of the floor, the outer walls (the first 4 boxes) and the obstacles."""
    out = {name: Mesh(tile) for name, (_, tile) in MATERIALS.items()}
    h = layout["height"]
    length, width = layout["size"]
    m = FLOOR_MARGIN
    out["floor"].floor(-m, -m, length + m, width + m)
    for i, c in enumerate(layout["circles"]):
        out[material("circle", i, c)].cylinder(*c, h)
    for i, b in enumerate(layout["boxes"]):
        out["walls" if i < 4 else material("box", i - 4, b)].box(*b, h)
    return {name: mesh for name, mesh in out.items() if mesh.faces}


VISUAL = """\
        <visual name="{name}">
          <geometry>
            <mesh>
              <uri>meshes/{name}.obj</uri>
            </mesh>
          </geometry>
          <material>
            <diffuse>1 1 1 1</diffuse>
            <specular>0.2 0.2 0.2 1</specular>
            <pbr>
              <metal>
                <albedo_map>{tex}_Color.jpg</albedo_map>
                <normal_map>{tex}_NormalGL.jpg</normal_map>
                <roughness_map>{tex}_Roughness.jpg</roughness_map>
                <metalness>0</metalness>
              </metal>
            </pbr>
          </material>
        </visual>"""

COLLISION = """\
        <collision name="{name}">
          <pose>{x:.5f} {y:.5f} {z:.5f} 0 0 {yaw:.6f}</pose>
          <geometry>
            {geometry}
          </geometry>
        </collision>"""

WORLD = """\
<?xml version="1.0" encoding="UTF-8"?>
<sdf version="1.9">
  <!-- generated by sim/gazebo/export_world.py from the maps in maps.npz next to it -->
  <world name="{name}">
    <physics type="ode">
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>250</real_time_update_rate>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <plugin filename="gz-sim-air-pressure-system" name="gz::sim::systems::AirPressure"/>
    <plugin filename="gz-sim-magnetometer-system" name="gz::sim::systems::Magnetometer"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="libOpticalFlowSystem.so" name="custom::OpticalFlowSystem"/>
    <gravity>0 0 -9.8</gravity>
    <magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>
    <atmosphere type="adiabatic"/>
    <scene>
      <grid>false</grid>
      <ambient>0.5 0.5 0.5 1</ambient>
      <background>0.7 0.7 0.7 1</background>
      <shadows>true</shadows>
    </scene>
    <!-- no shadows: the shadow of the drone moves with it under the optical flow camera and
         pulls the measured flow towards zero; ceiling lights of a hall give soft ones anyway -->
    <light name="ceiling" type="directional">
      <pose>0 0 8 0 0 0</pose>
      <cast_shadows>false</cast_shadows>
      <intensity>1</intensity>
      <direction>0.2 0.3 -1</direction>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
    </light>
    <model name="map">
      <static>true</static>
      <link name="link">
{collisions}
{visuals}
      </link>
    </model>
    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <world_frame_orientation>ENU</world_frame_orientation>
      <latitude_deg>47.397971057728974</latitude_deg>
      <longitude_deg>8.546163739800146</longitude_deg>
      <elevation>0</elevation>
    </spherical_coordinates>
  </world>
</sdf>
"""


def world_sdf(name, layout, materials):
    h = layout["height"]

    def collision(name, geometry, x=0.0, y=0.0, z=0.0, yaw=0.0):
        return COLLISION.format(name=name, x=x, y=y, z=z, yaw=yaw, geometry=geometry)

    collisions = [collision("ground", "<plane><normal>0 0 1</normal></plane>")]
    for i, (x, y, r) in enumerate(layout["circles"]):
        cylinder = f"<cylinder><radius>{r:.5f}</radius><length>{h:.5f}</length></cylinder>"
        collisions.append(collision(f"circle_{i}", cylinder, x, y, h / 2))
    for i, (x, y, hx, hy, angle) in enumerate(layout["boxes"]):
        box = f"<box><size>{2 * hx:.5f} {2 * hy:.5f} {h:.5f}</size></box>"
        collisions.append(collision(f"box_{i}", box, x, y, h / 2, angle))
    textures = {m: "../../media/textures/{0}/{0}".format(MATERIALS[m][0]) for m in materials}
    visuals = [VISUAL.format(name=m, tex=textures[m]) for m in materials]
    return WORLD.format(name=name, collisions="\n".join(collisions), visuals="\n".join(visuals))


def export(name, maps, root=ROOT / "worlds"):
    """Writes root/name: world.sdf, the meshes and maps.npz with the start-goal pairs."""
    out = root / name
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    layout = maps[0]
    parts = meshes(layout)
    for material_name, mesh in parts.items():
        (out / "meshes" / f"{material_name}.obj").write_text(mesh.obj())
    (out / "world.sdf").write_text(world_sdf(name, layout, parts))
    pool = mapgen.collate(maps)
    np.savez_compressed(out / "maps.npz", **pool)
    return pool


def build(name):
    """The maps of one world: its obstacles with PAIRS start-goal pairs."""
    if name.startswith("depth_"):
        layout = generated(name.removeprefix("depth_"), "val")
    else:
        layout = {"arena": arena, "empty": empty}.get(name, lambda: generated(name))()
    rng = np.random.default_rng([SEED, list(WORLDS).index(name)])
    return place_pairs(layout, rng, dist=WORLDS[name])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", default=list(WORLDS), help="worlds, all by default")
    parser.add_argument("--out", type=Path, default=ROOT / "worlds")
    args = parser.parse_args(argv)
    for name in args.names:
        print(f"{name}: {mapgen.summary(export(name, build(name), args.out))}")


if __name__ == "__main__":
    main()
