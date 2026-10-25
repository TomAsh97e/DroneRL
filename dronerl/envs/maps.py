"""Procedural maps (room, corridor) checked with A*, and the fixed map sets (M2.2).

Obstacles are 2D primitives extruded vertically: circles (x, y, r) and oriented boxes
(x, y, half length, half width, angle). Outer walls are boxes too. A map set is a dict of arrays
padded to a common number of primitives, with the true counts in the `n_*` entries.
"""

import argparse
import hashlib
import heapq
import math
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from dronerl import contract

MAPGEN_VERSION = 1
DENSITY = (0, 2, 4, 6, 8, 11)  # obstacles per 100 m², one entry per level
TEST_LEVELS = {"easy": 2, "medium": 4, "hard": 5}
SEEDS = {"train": 0, "val": 20_000, "test": 10_000}
SIZES = {"train": 4096, "val": 100, "test": 200}
ROOT = Path("runs/maps") / f"v{MAPGEN_VERSION}"

CELL = 0.1
INFLATE = contract.COLLISION_DIST + 0.15
START_CLEARANCE = 1.0
WALL = 0.2
SEGMENT = 0.075  # half thickness of wall segments
MAX_ATTEMPTS = 200


def sdf(points, circles, boxes):
    """Signed distance from points [N, 2] to the nearest primitive, negative inside."""
    d = np.full(len(points), np.inf)
    if len(circles):
        diff = points[:, None] - circles[None, :, :2]
        d = np.minimum(d, (np.linalg.norm(diff, axis=-1) - circles[:, 2]).min(1))
    if len(boxes):
        rel = points[:, None] - boxes[None, :, :2]
        c, s = np.cos(boxes[:, 4]), np.sin(boxes[:, 4])
        local = np.stack([c * rel[..., 0] + s * rel[..., 1], c * rel[..., 1] - s * rel[..., 0]], -1)
        q = np.abs(local) - boxes[:, 2:4]
        outside = np.linalg.norm(np.maximum(q, 0), axis=-1)
        d = np.minimum(d, (outside + np.minimum(q.max(-1), 0)).min(1))
    return d


def segment_sdf(a, b, circles, boxes, step=0.05):
    n = max(2, int(np.ceil(np.linalg.norm(b - a) / step)) + 1)
    t = np.linspace(0, 1, n)[:, None]
    return sdf(a + t * (b - a), circles, boxes).min()


def outer_walls(length, width):
    w = WALL / 2
    return [
        (length / 2, -w, length / 2 + WALL, w, 0),
        (length / 2, width + w, length / 2 + WALL, w, 0),
        (-w, width / 2, w, width / 2 + WALL, 0),
        (length + w, width / 2, w, width / 2 + WALL, 0),
    ]


def random_obstacle(rng, kind, length, width, center=None):
    """Returns ("circle" | "box", params). In corridors walls are partitions on a side wall."""
    shape = rng.choice(3, p=[0.4, 0.4, 0.2] if kind == "room" else [0.35, 0.35, 0.3])
    if center is None:
        center = rng.uniform(0.3, length - 0.3), rng.uniform(0.3, width - 0.3)
    x, y = center
    if shape == 0:
        return "circle", (x, y, rng.uniform(0.1, 0.5))
    if shape == 1:
        return "box", (x, y, rng.uniform(0.2, 1.0), rng.uniform(0.2, 1.0), rng.uniform(0, np.pi))
    if kind == "room":
        return "box", (x, y, rng.uniform(0.5, 2.0), SEGMENT, rng.uniform(0, np.pi))
    half = rng.uniform(0.1, 0.25) * width
    return "box", (x, half if rng.integers(2) else width - half, SEGMENT, half, 0)


def astar(free, start, goal):
    """Shortest 8-connected path of cells on a boolean grid, without cutting corners."""
    nx, ny = free.shape
    grid = free.tolist()
    moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0)]
    moves += [(dx, dy, math.sqrt(2)) for dx in (1, -1) for dy in (1, -1)]
    cost, parent = {start: 0.0}, {start: None}
    heap = [(math.dist(start, goal), 0.0, start)]
    while heap:
        _, g, cur = heapq.heappop(heap)
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1]
        if g > cost[cur]:
            continue
        x, y = cur
        for dx, dy, step in moves:
            i, j = x + dx, y + dy
            if not (0 <= i < nx and 0 <= j < ny) or not grid[i][j]:
                continue
            if dx and dy and not (grid[i][y] and grid[x][j]):
                continue
            new = g + step
            if new < cost.get((i, j), math.inf):
                cost[(i, j)], parent[(i, j)] = new, cur
                heapq.heappush(heap, (new + math.dist((i, j), goal), new, (i, j)))
    return None


def occupancy(size, circles, boxes):
    """Cell centers [nx, ny, 2] and free cells of the grid with obstacles inflated by INFLATE."""
    nx, ny = int(np.ceil(size[0] / CELL)), int(np.ceil(size[1] / CELL))
    xs, ys = (np.arange(nx) + 0.5) * CELL, (np.arange(ny) + 0.5) * CELL
    centers = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1)
    free = sdf(centers.reshape(-1, 2), circles, boxes).reshape(nx, ny) > INFLATE
    return centers, free


def plan_path(start, goal, size, circles, boxes):
    """A* on the inflated grid shortened by string pulling, None if the map is unsolvable."""
    centers, free = occupancy(size, circles, boxes)
    cells = astar(free, tuple((start // CELL).astype(int)), tuple((goal // CELL).astype(int)))
    if cells is None:
        return None
    raw = [start] + [centers[c] for c in cells[1:-1]] + [goal]
    path, anchor = [start], 0
    for j in range(2, len(raw)):
        if segment_sdf(raw[anchor], raw[j], circles, boxes) <= INFLATE:
            anchor = j - 1
            path.append(raw[anchor])
    path.append(goal)
    return np.array(path)


def layout(rng, level, kind):
    """Size, start, goal and obstacles of a map that may still be unsolvable."""
    if kind == "room":
        size = np.array([rng.uniform(14, 22), rng.uniform(6, 12)])
    else:
        size = np.array([rng.uniform(14, 22), rng.uniform(2.5, 4.0)])
    length, width = size
    while True:
        start = np.array([rng.uniform(1.5, 2.5), rng.uniform(1.0, width - 1.0)])
        goal = np.array([length - rng.uniform(1.5, 2.5), rng.uniform(1.0, width - 1.0)])
        if 10 <= np.linalg.norm(goal - start) <= 18:
            break
    normal = np.array([start[1] - goal[1], goal[0] - start[0]]) / np.linalg.norm(goal - start)
    count = round(DENSITY[level] * length * width / 100)
    shapes = {"circle": [], "box": []}
    while len(shapes["circle"]) + len(shapes["box"]) < count:
        # the first obstacle blocks the straight line, so flying straight is not enough
        blocker = not shapes["circle"] and not shapes["box"]
        center = None
        if blocker:
            offset = rng.uniform(-0.5, 0.5) * normal
            center = start + rng.uniform(0.3, 0.7) * (goal - start) + offset
        shape, params = random_obstacle(rng, kind, length, width, center)
        one = {"circle": np.zeros((0, 3)), "box": np.zeros((0, 5))}
        one[shape] = np.array([params])
        single = one["circle"], one["box"]
        if sdf(np.stack([start, goal]), *single).min() < START_CLEARANCE:
            continue
        if blocker and segment_sdf(start, goal, *single) >= contract.COLLISION_DIST:
            continue
        shapes[shape].append(params)
    circles = np.array(shapes["circle"]).reshape(-1, 3)
    boxes = np.concatenate([outer_walls(length, width), np.array(shapes["box"]).reshape(-1, 5)])
    return size, start, goal, circles, boxes


def make_map(rng, level, kind):
    """One solvable map. `attempts` counts the layouts rejected by A* before it."""
    for attempt in range(MAX_ATTEMPTS):
        size, start, goal, circles, boxes = layout(rng, level, kind)
        path = plan_path(start, goal, size, circles, boxes)
        if path is not None:
            break
    else:
        raise RuntimeError(f"no solvable {kind} map on level {level}")
    direction = math.atan2(goal[1] - start[1], goal[0] - start[0])
    return {
        "kind": int(kind == "corridor"),
        "level": level,
        "size": size,
        "circles": circles,
        "boxes": boxes,
        "start": start,
        "goal": goal,
        "yaw0": direction + rng.uniform(-np.pi / 2, np.pi / 2),
        "height": rng.uniform(2.0, 4.0),
        "path": path,
        "path_len": np.linalg.norm(np.diff(path, axis=0), axis=1).sum(),
        "attempts": attempt,
    }


def map_from_seed(job):
    base, level, i = job
    # even maps are rooms and odd ones corridors, so every set is split evenly
    return make_map(np.random.default_rng([base, level, i]), level, ("room", "corridor")[i % 2])


def pad(arrays, n, repeat_last=False):
    out = np.zeros((len(arrays), n, arrays[0].shape[-1]))
    for k, a in enumerate(arrays):
        out[k, : len(a)] = a
        if repeat_last:
            out[k, len(a) :] = a[-1]
    return out


def collate(maps):
    pool = {k: np.array([m[k] for m in maps]) for k in maps[0] if np.ndim(maps[0][k]) < 2}
    for key in ("circles", "boxes", "path"):
        counts = np.array([len(m[key]) for m in maps])
        pool[key] = pad([m[key] for m in maps], max(1, counts.max()), key == "path")
        pool[f"n_{key}"] = counts
    return pool


def concat(pools):
    """Joins map sets, padding the primitives and paths to the largest count."""
    out = {}
    for key in pools[0]:
        if pools[0][key].ndim == 3:
            n = max(p[key].shape[1] for p in pools)
            out[key] = np.concatenate([pad(p[key], n, key == "path") for p in pools])
        else:
            out[key] = np.concatenate([p[key] for p in pools])
    return out


def get_map(pool, i):
    m = {k: v[i] for k, v in pool.items() if not k.startswith("n_")}
    for key in ("circles", "boxes", "path"):
        m[key] = m[key][: pool[f"n_{key}"][i]]
    return m


def generate(split, level, n=None, workers=None):
    jobs = [(SEEDS[split], level, i) for i in range(SIZES[split] if n is None else n)]
    if workers == 1:
        return collate([map_from_seed(job) for job in jobs])
    with Pool(workers or os.cpu_count()) as pool:
        return collate(pool.map(map_from_seed, jobs, chunksize=4))


def load(split, level, root=ROOT):
    """Map set from the cache, generated on first use."""
    path = Path(root) / f"{split}_l{level}.npz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **generate(split, level))
    with np.load(path) as f:
        return dict(f)


def levels(split):
    return range(len(DENSITY)) if split == "train" else TEST_LEVELS.values()


def digest(pool):
    h = hashlib.sha256()
    for key in sorted(pool):
        h.update(key.encode() + np.ascontiguousarray(pool[key]).tobytes())
    return h.hexdigest()[:16]


def summary(pool):
    straight = np.linalg.norm(pool["goal"] - pool["start"], axis=1)
    return {
        "maps": len(straight),
        "obstacles": float(np.mean(pool["n_circles"] + pool["n_boxes"] - 4)),
        "rejected_per_map": float(pool["attempts"].mean()),
        "straight_m": float(straight.mean()),
        "astar_ratio": float(np.mean(pool["path_len"] / straight)),
        "hash": digest(pool),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SEEDS, nargs="+", default=list(SEEDS))
    args = parser.parse_args(argv)
    for split in args.split:
        for level in levels(split):
            print(f"{split} l{level}: {summary(load(split, level))}", flush=True)


if __name__ == "__main__":
    main()
