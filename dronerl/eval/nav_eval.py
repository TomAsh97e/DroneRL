"""Fixed navigation test set (M2.4): metrics per level, top-down plots, depth strips, videos.

A policy is `act(obs) -> action` on tensors. Specs: "astar[:lookahead[:k_yaw]]" for pure pursuit
on the A* path, "straight[:k_p]" for flying straight at the goal, a run directory or checkpoint
of a trained policy.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import yaml

from dronerl import contract
from dronerl.agents import ppo_gpu
from dronerl.agents.scripted import GoalP, PurePursuit
from dronerl.envs import maps as mapgen
from dronerl.envs.nav_gpu import NavEnv
from dronerl.eval import nav_plots
from dronerl.train import rsl_rl_baseline

OUTCOMES = ("success", "collision", "out_of_area", "timeout")
FOV = contract.HFOV / 2


def checkpoint(spec):
    """Checkpoint file of a trained policy given as a run directory or a file, else None."""
    path = Path(spec)
    if not path.exists():
        return None
    return path / "checkpoints" / "best.pt" if path.is_dir() else path


def load_trained(path, device):
    """`act` of a checkpoint of ppo_gpu.py or rsl_rl_baseline.py."""
    if "actor" in torch.load(path, map_location="cpu"):
        return rsl_rl_baseline.load(path, device)
    return ppo_gpu.Agent.load(path, device).act


def load_policy(spec, device):
    """Returns (variant, name, make) where make(env, pool) builds `act` for one env."""
    ckpt = checkpoint(spec)
    if ckpt is not None:
        cfg = yaml.safe_load((ckpt.parent.parent / "config.yaml").read_text())
        act = load_trained(ckpt, device)
        return cfg["variant"], f"{cfg['variant']} s{cfg['seed']}", lambda env, pool: act
    kind, *args = spec.split(":")
    if kind == "astar":
        lookahead = float(args[0]) if args else 1.0
        k_yaw = float(args[1]) if len(args) > 1 else 4.0
        name = f"A* lookahead={lookahead:g} k_yaw={k_yaw:g}"
        return name, name, lambda env, pool: PurePursuit(env, pool["path"], lookahead, k_yaw)
    if kind == "straight":
        k_p = float(args[0]) if args else 1.0
        name = f"prosto do celu k_p={k_p:g}"
        return name, name, lambda env, pool: GoalP(k_p)
    raise ValueError(f"unknown policy: {spec}")


def snap(x):
    # on a CPU env .cpu() returns the live tensor, which step() then overwrites
    return x.to("cpu", copy=True)


@torch.no_grad()
def rollout(env, act, frames=()):
    """Plays one episode per env; `frames` lists envs whose depth images are kept."""
    obs = env.reset()
    pos, yaw, actions, alive, depth = [snap(env.pos)], [snap(env.yaw)], [], [], []
    result = {k: torch.zeros(env.num_envs, dtype=torch.bool) for k in OUTCOMES}
    stats = {}
    while env.alive.any():
        if len(frames):
            depth.append(obs["depth"][list(frames), 0].cpu())
        alive.append(snap(env.alive))
        a = act(obs).clamp(-1, 1)
        obs, _, terminated, truncated, info = env.step(a)
        done = (terminated | truncated).cpu()
        for key in OUTCOMES:
            result[key] |= info[key].cpu()
        for key in ("length", "return", "path_length", "min_clearance"):
            stats.setdefault(key, torch.zeros(env.num_envs))[done] = info[key].cpu()[done].float()
        actions.append(a.cpu())
        pos.append(snap(env.pos))
        yaw.append(snap(env.yaw))
    out = {k: v.numpy() for k, v in {**result, **stats}.items()}
    out.update(pos=torch.stack(pos).numpy(), yaw=torch.stack(yaw).numpy())
    out.update(actions=torch.stack(actions).numpy(), alive=torch.stack(alive).numpy())
    if len(frames):
        out["depth"] = torch.stack(depth).numpy()
    return out


def metrics(run, pool):
    win = run["success"]
    straight = np.linalg.norm(pool["goal"] - pool["start"], axis=1)
    alive = run["alive"]
    change = np.linalg.norm(np.diff(run["actions"], axis=0), axis=-1)[alive[1:]]
    # heading of the velocity relative to the camera axis, only while really moving
    vel = np.diff(run["pos"], axis=0)[alive]
    moving = np.linalg.norm(vel, axis=-1) > 0.01
    rel = np.arctan2(vel[:, 1], vel[:, 0]) - run["yaw"][:-1][alive]
    off = np.abs(np.arctan2(np.sin(rel), np.cos(rel)))[moving] > FOV

    def mean(x):
        return float(np.mean(x)) if len(x) else None

    result = {f"{k}_rate": float(run[k].mean()) for k in OUTCOMES}
    result.update(
        time_to_goal_s=mean(run["length"][win] * contract.DT),
        path_efficiency=mean((straight - contract.SUCCESS_RADIUS)[win] / run["path_length"][win]),
        astar_efficiency=mean(
            (pool["path_len"] - contract.SUCCESS_RADIUS)[win] / run["path_length"][win]
        ),
        mean_action_change=mean(change),
        min_clearance=mean(run["min_clearance"]),
        off_fov=mean(off),
        mean_return=mean(run["return"]),
    )
    return result


def latency(act, obs, device, repeats=200):
    """Milliseconds per call for a single observation."""
    one = {k: v[:1].to(device) for k, v in obs.items()}
    with torch.no_grad():
        for _ in range(20):
            act(one)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            act(one)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - start) / repeats


def inference_latency(ckpt, obs):
    """Milliseconds per call with batch 1, on one CPU thread and on the GPU."""
    cpu, threads = torch.device("cpu"), torch.get_num_threads()
    torch.set_num_threads(1)
    result = {"cpu_1_thread": latency(load_trained(ckpt, cpu), obs, cpu)}
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        gpu = torch.device("cuda")
        result["gpu"] = latency(load_trained(ckpt, gpu), obs, gpu)
    return result


def evaluate(
    spec,
    split="test",
    levels=mapgen.TEST_LEVELS,
    device="cuda",
    out=None,
    plots=False,
    video=0,
    randomize=False,
    failures=False,
):
    """`video` is the number of maps recorded per level, `failures` plots all failed episodes."""
    device = torch.device(device)
    variant, name, make = load_policy(spec, device)
    ckpt = checkpoint(spec)
    result = {"variant": variant, "name": name, "spec": spec, "levels": {}}
    for label, level in levels.items():
        pool = mapgen.load(split, level)
        env = NavEnv(pool, device=device, autoreset=False, randomize=randomize)
        run = rollout(env, make(env, pool), frames=range(max(int(video), int(plots))))
        m = metrics(run, pool)
        result["levels"][label] = m
        result.setdefault("episodes", {})[label] = [
            {"map": i, "outcome": next(k for k in OUTCOMES if run[k][i]), "steps": int(n)}
            for i, n in enumerate(run["length"])
        ]
        print(
            f"{name}, {label}: SR {100 * m['success_rate']:.1f}%, "
            f"CR {100 * m['collision_rate']:.1f}%",
            flush=True,
        )
        if out is not None and (plots or video or failures):
            slug = out / name.replace(" ", "_").replace("=", "").replace("*", "star")
            slug.mkdir(parents=True, exist_ok=True)
            if plots:
                nav_plots.plot_runs(
                    run, pool, slug / f"trajectories_{label}.png", f"{name}, poziom {label}"
                )
                nav_plots.plot_depth_strip(run, slug / f"depth_{label}.png")
            failed = np.flatnonzero(~run["success"])
            if failures and len(failed):
                title = f"{name}, poziom {label}, nieudane epizody"
                nav_plots.plot_runs(run, pool, slug / f"failures_{label}.png", title, failed)
            for i in range(int(video)):
                record_video(run, pool, slug / f"episode_{label}_map{i}.mp4", i)
    if ckpt is not None:
        result["latency_ms"] = inference_latency(ckpt, env.reset())
    return result


def record_video(run, pool, out, env_index=0):
    frames = nav_plots.episode_frames(run, mapgen.get_map(pool, env_index), env_index)
    imageio.mimsave(out, frames, fps=contract.CTRL_HZ, macro_block_size=1)


COLUMNS = {
    "success_rate": ("SR", 100, ".1f"),
    "collision_rate": ("kolizje", 100, ".1f"),
    "timeout_rate": ("timeout", 100, ".1f"),
    "out_of_area_rate": ("poza obszarem", 100, ".1f"),
    "time_to_goal_s": ("czas [s]", 1, ".1f"),
    "path_efficiency": ("efekt. ścieżki", 1, ".3f"),
    "astar_efficiency": ("efekt. wzgl. A*", 1, ".3f"),
    "mean_action_change": ("‖Δa‖", 1, ".3f"),
    "min_clearance": ("min. zapas [m]", 1, ".2f"),
    "off_fov": ("poza FOV", 100, ".1f"),
}


def table(results):
    """Markdown table per level, mean ± std over the policies (seeds) of each variant."""
    groups = defaultdict(list)
    for r in results:
        groups[r["variant"]].append(r)
    head = ["wariant", "poziom", "n"] + [c[0] for c in COLUMNS.values()] + ["SR per ziarno"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for variant, rs in groups.items():
        for level in rs[0]["levels"]:
            cells = [variant, level, str(len(rs))]
            for key, (_, scale, spec) in COLUMNS.items():
                values = [r["levels"][level][key] for r in rs]
                values = [scale * v for v in values if v is not None]
                if not values:
                    cells.append("-")
                elif len(values) == 1:
                    cells.append(format(values[0], spec))
                else:
                    cells.append(f"{np.mean(values):{spec}} ± {np.std(values):{spec}}")
            sr = [f"{100 * r['levels'][level]['success_rate']:.1f}" for r in rs]
            cells.append(", ".join(sr))
            lines.append("| " + " | ".join(cells) + " |")
    timed = [r for r in results if "latency_ms" in r]
    if timed:
        lines += ["", "| polityka | inferencja CPU, 1 wątek [ms] | inferencja GPU [ms] |"]
        lines.append("|---|---|---|")
        for r in timed:
            cpu, gpu = r["latency_ms"]["cpu_1_thread"], r["latency_ms"].get("gpu")
            lines.append(f"| {r['name']} | {cpu:.2f} | {'-' if gpu is None else f'{gpu:.2f}'} |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policies", nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument(
        "--levels", nargs="+", choices=mapgen.TEST_LEVELS, default=list(mapgen.TEST_LEVELS)
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--randomize", action="store_true", help="randomized dynamics (M3 ranges)")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument(
        "--video", type=int, nargs="?", const=1, default=0, metavar="N", help="videos of N maps"
    )
    parser.add_argument("--failures", action="store_true", help="plots of all failed episodes")
    args = parser.parse_args(argv)
    levels = {k: mapgen.TEST_LEVELS[k] for k in args.levels}
    args.out.mkdir(parents=True, exist_ok=True)
    results = [
        evaluate(
            spec,
            args.split,
            levels,
            args.device,
            args.out,
            args.plots,
            args.video,
            args.randomize,
            args.failures,
        )
        for spec in args.policies
    ]
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2))
    text = table(results)
    (args.out / "metrics.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
