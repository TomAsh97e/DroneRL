import importlib.util
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import torch

from dronerl.envs import maps as mapgen
from dronerl.envs.nav_gpu import NavEnv

ROOT = Path(__file__).parents[1] / "sim" / "gazebo"
spec = importlib.util.spec_from_file_location("export_world", ROOT / "export_world.py")
export_world = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export_world)


@pytest.fixture(scope="module")
def arena(tmp_path_factory):
    out = tmp_path_factory.mktemp("worlds")
    maps = export_world.build("arena")
    pool = export_world.export("arena", maps, out)
    return maps, pool, out / "arena"


def floats(text):
    return [float(v) for v in text.split()]


def read_obj(path):
    v, vn, faces = [], [], []
    for line in path.read_text().splitlines():
        kind, *rest = line.split()
        if kind == "v":
            v.append(floats(" ".join(rest)))
        elif kind == "vn":
            vn.append(floats(" ".join(rest)))
        elif kind == "f":
            faces.append([int(x.split("/")[0]) - 1 for x in rest])
    return np.array(v), np.array(vn), np.array(faces)


def test_collisions_match_the_map(arena):
    maps, _, out = arena
    m = maps[0]
    h = m["height"]
    link = ET.parse(out / "world.sdf").find("world/model[@name='map']/link")
    collisions = {c.get("name"): c for c in link.findall("collision")}
    assert len(collisions) == 1 + len(m["circles"]) + len(m["boxes"])
    for i, (x, y, r) in enumerate(m["circles"]):
        c = collisions[f"circle_{i}"]
        np.testing.assert_allclose(floats(c.find("pose").text), [x, y, h / 2, 0, 0, 0], atol=1e-5)
        assert float(c.find("geometry/cylinder/radius").text) == pytest.approx(r, abs=1e-5)
        assert float(c.find("geometry/cylinder/length").text) == pytest.approx(h, abs=1e-5)
    for i, (x, y, hx, hy, angle) in enumerate(m["boxes"]):
        c = collisions[f"box_{i}"]
        np.testing.assert_allclose(
            floats(c.find("pose").text), [x, y, h / 2, 0, 0, angle], atol=1e-5
        )
        np.testing.assert_allclose(
            floats(c.find("geometry/box/size").text), [2 * hx, 2 * hy, h], atol=1e-5
        )


def test_meshes_cover_the_primitives_and_face_outwards(arena):
    maps, _, out = arena
    m = maps[0]
    visuals = ET.parse(out / "world.sdf").findall("world/model/link/visual")
    assert {v.find("geometry/mesh/uri").text for v in visuals} == {
        f"meshes/{p.stem}.obj" for p in (out / "meshes").iterdir()
    }
    for path in (out / "meshes").iterdir():
        v, vn, faces = read_obj(path)
        a, b, c = (v[faces[:, k]] for k in range(3))
        normal = np.cross(b - a, c - a)
        # counter-clockwise from outside, so back-face culling keeps the visible side
        assert (np.einsum("ij,ij->i", normal, vn[faces[:, 0]]) > 0).all()
        if path.stem == "floor":
            assert (v[:, 2] == 0).all()
            continue
        assert np.isin(v[:, 2], [0, m["height"]]).all()
        assert np.abs(mapgen.sdf(v[:, :2], m["circles"], m["boxes"])).max() < 1e-4


def test_pairs_are_clear_solvable_and_share_the_obstacles(arena):
    maps, pool, _ = arena
    assert len(maps) == export_world.PAIRS
    for m in maps:
        np.testing.assert_array_equal(m["boxes"], maps[0]["boxes"])
        np.testing.assert_array_equal(m["circles"], maps[0]["circles"])
        ends = np.stack([m["start"], m["goal"]])
        assert mapgen.sdf(ends, m["circles"], m["boxes"]).min() >= mapgen.START_CLEARANCE
        assert m["path_len"] >= np.linalg.norm(m["goal"] - m["start"]) - 1e-9
        direction = math.atan2(*(m["goal"] - m["start"])[::-1])
        assert abs(math.remainder(m["yaw0"] - direction, 2 * math.pi)) <= math.pi / 2
    # pairs run in more than one direction
    starts = np.array([m["start"] for m in maps])
    assert starts[:, 0].std() > 1 and starts[:, 1].std() > 1


def test_navenv_flies_the_world_maps(arena):
    _, pool, _ = arena
    env = NavEnv(pool, device="cpu", autoreset=False)
    env.reset()
    torch.testing.assert_close(env.pos, torch.as_tensor(pool["start"], dtype=torch.float32))
    assert env.num_envs == export_world.PAIRS


def test_committed_arena_is_up_to_date(arena):
    _, _, out = arena
    committed = ROOT / "worlds" / "arena"
    for path in [out / "world.sdf", *sorted((out / "meshes").iterdir())]:
        assert path.read_text() == (committed / path.relative_to(out)).read_text(), path.name
    with np.load(committed / "maps.npz") as f, np.load(out / "maps.npz") as g:
        assert mapgen.digest(dict(f)) == mapgen.digest(dict(g))


def test_turn_rotates_about_the_centre():
    size = np.array([10.0, 10.0])
    p = np.array([2.0, 3.0])
    np.testing.assert_allclose(export_world.turn(p, size, 1), [7.0, 2.0])
    np.testing.assert_allclose(export_world.turn(p, size, 2), size - p)
    np.testing.assert_allclose(export_world.turn(p, size, 4), p)
