"""Top-down plots of navigation maps and episodes; as a script, previews of the map sets."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patches  # noqa: E402

from dronerl import contract  # noqa: E402
from dronerl.envs import maps  # noqa: E402

FOV = contract.HFOV / 2


def draw_map(ax, m, path=True, inflated=True):
    """Obstacles, inflated zones of the A* check, start, goal and the A* path of one map."""
    length, width = m["size"]
    if inflated:
        centers, free = maps.occupancy(m["size"], m["circles"], m["boxes"])
        ax.contourf(centers[..., 0], centers[..., 1], free, levels=[-0.5, 0.5], colors=["0.88"])
    for x, y, r in m["circles"]:
        ax.add_patch(patches.Circle((x, y), r, color="0.3"))
    for x, y, hx, hy, angle in m["boxes"]:
        rect = patches.Rectangle(
            (x - hx, y - hy),
            2 * hx,
            2 * hy,
            angle=np.degrees(angle),
            rotation_point="center",
            color="0.3",
        )
        ax.add_patch(rect)
    ax.add_patch(patches.Circle(m["goal"], contract.SUCCESS_RADIUS, color="tab:green", alpha=0.5))
    ax.plot(*m["start"], "o", color="tab:gray", markersize=4)
    heading = np.array([np.cos(m["yaw0"]), np.sin(m["yaw0"])])
    ax.arrow(*m["start"], *heading, width=0.05, color="tab:gray", length_includes_head=True)
    if path:
        ax.plot(*m["path"].T, "--", color="tab:green", linewidth=1)
    ax.set_xlim(-maps.WALL - 0.3, length + maps.WALL + 0.3)
    ax.set_ylim(-maps.WALL - 0.3, width + maps.WALL + 0.3)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)


def plot_maps(pool, out, title, n=12):
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(14, 2.4 * rows))
    for k, ax in enumerate(axes.flat[:n]):
        draw_map(ax, maps.get_map(pool, k))
        ax.set_title(f"mapa {k}", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90)
    plt.close(fig)


COLORS = {
    "success": "tab:blue",
    "collision": "tab:red",
    "out_of_area": "tab:purple",
    "timeout": "tab:orange",
}


# metric depth reads better than the inverse one, which is nearly black beyond 1 m
DEPTH_STYLE = {"cmap": "magma_r", "vmin": 0, "vmax": contract.DEPTH_MAX}


def metric(inverse_depth):
    return contract.decode_depth(inverse_depth)


def outcome(run, i):
    return next(k for k in COLORS if run[k][i])


def plot_runs(run, pool, out, title, ids=range(6)):
    """Trajectories on the maps `ids`, colored by the outcome."""
    rows = (len(ids) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(14, 2.6 * rows), squeeze=False)
    for i, ax in zip(ids, axes.flat):
        draw_map(ax, maps.get_map(pool, i))
        steps = int(run["length"][i])
        ax.plot(*run["pos"][: steps + 1, i].T, color=COLORS[outcome(run, i)], linewidth=1.5)
        ax.set_title(f"mapa {i}: {outcome(run, i)}, {steps / contract.CTRL_HZ:.1f} s", fontsize=8)
    for name, color in COLORS.items():
        axes.flat[0].plot([], [], color=color, label=name)
    axes.flat[0].legend(fontsize=6, loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


def plot_depth_strip(run, out, n=8):
    """Depth images of the first recorded episode at evenly spaced steps."""
    steps = np.linspace(0, int(run["length"][0]) - 1, n).astype(int)
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.2))
    for ax, t in zip(axes, steps):
        ax.imshow(metric(run["depth"][t, 0]), **DEPTH_STYLE)
        ax.set_title(f"t = {t / contract.CTRL_HZ:.1f} s", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def episode_frames(run, m, i=0):
    """Video frames: top-down view with the camera field of view next to the depth image.

    Needs the depth images of env i, i.e. a rollout with frames=range(k), k > i.
    """
    fig, (top, cam) = plt.subplots(1, 2, figsize=(12, 4), width_ratios=[2.2, 1])
    draw_map(top, m)
    trail = top.plot([], [], color="tab:blue", linewidth=1.5)[0]
    fov = patches.Wedge((0, 0), 3, 0, 1, color="tab:blue", alpha=0.2)
    top.add_patch(fov)
    image = cam.imshow(metric(run["depth"][0, i]), **DEPTH_STYLE)
    cam.axis("off")
    frames = []
    for t in range(int(run["length"][i]) + 1):
        pos, yaw = run["pos"][t, i], np.degrees(run["yaw"][t, i])
        trail.set_data(*run["pos"][: t + 1, i].T)
        fov.set_center(pos)
        fov.set_theta1(yaw - np.degrees(FOV))
        fov.set_theta2(yaw + np.degrees(FOV))
        image.set_data(metric(run["depth"][min(t, len(run["depth"]) - 1), i]))
        top.set_title(f"t = {t / contract.CTRL_HZ:.1f} s", fontsize=9)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    return frames


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=maps.SEEDS, nargs="+", default=["train"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("-n", type=int, default=12)
    args = parser.parse_args(argv)
    for split in args.split:
        for level in maps.levels(split):
            pool = maps.load(split, level)
            plot_maps(pool, args.out / f"{split}_l{level}.png", f"{split}, poziom {level}", args.n)


if __name__ == "__main__":
    main()
