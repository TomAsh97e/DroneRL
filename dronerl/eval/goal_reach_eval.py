"""Fixed GoalReach test set (M1.5): metrics, trajectory plots and videos for any policy."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import imageio.v2 as imageio  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pybullet as p  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from dronerl.agents.ppo import Agent  # noqa: E402
from dronerl.agents.scripted import PController  # noqa: E402
from dronerl.envs.goal_reach import GoalReachAviary  # noqa: E402
from dronerl.eval.evaluate import EVAL_SEED  # noqa: E402
from dronerl.train.sb3_baseline import ALGOS, policy  # noqa: E402

TEST_EPISODES = 100
OUTCOMES = {
    "success": "tab:blue",
    "crash": "tab:red",
    "out_of_bounds": "tab:purple",
    "timeout": "tab:orange",
}


def load_policy(spec):
    """Returns (variant, name, act, env_kwargs) for a run directory, a checkpoint or "p[:k_p]"."""
    if spec.split(":")[0] == "p":
        k_p = float(spec.split(":")[1]) if ":" in spec else 1.0
        name = f"P k_p={k_p:g}"
        return name, name, PController(k_p), {"act": "vel"}
    ckpt = Path(spec)
    if ckpt.is_dir():
        ckpt = next(c for c in sorted((ckpt / "checkpoints").glob("best.*")) if c.suffix != ".pkl")
    cfg = yaml.safe_load((ckpt.parent.parent / "config.yaml").read_text())
    kwargs = cfg["env_kwargs"]
    if ckpt.suffix == ".pt":
        act = Agent.load(ckpt).act
    else:
        model = ALGOS[cfg["algo"]].load(ckpt, device="cpu")
        stats = ckpt.with_name(f"{ckpt.stem}_vecnormalize.pkl")
        norm = None
        if stats.exists():
            dummy = DummyVecEnv([lambda: GoalReachAviary(**kwargs)])
            norm = VecNormalize.load(str(stats), dummy)
            dummy.close()
        act = policy(model, norm)
    return cfg["variant"], f"{cfg['variant']} s{cfg['seed']}", act, kwargs


def run_episode(env, act, seed, on_frame=None):
    """Plays one episode; `on_frame()` runs after the reset and after every step."""
    obs, _ = env.reset(seed=seed)
    if on_frame:
        on_frame()
    start, goal = env.pos[0].copy(), env.goal.copy()
    pos, vel, actions, rates = [start], [env.vel[0].copy()], [], []
    while True:
        a = np.clip(act(obs), -1, 1)
        obs, _, terminated, truncated, info = env.step(a)
        pos.append(env.pos[0].copy())
        vel.append(env.vel[0].copy())
        actions.append(a)
        rates.append(np.linalg.norm(env.ang_v[0]))
        if on_frame:
            on_frame()
        if terminated or truncated:
            break
    if info["is_success"]:
        outcome = "success"
    elif info["crash"]:
        outcome = "crash"
    elif info["out_of_bounds"]:
        outcome = "out_of_bounds"
    else:
        outcome = "timeout"
    return {
        "seed": seed,
        "outcome": outcome,
        "start": start,
        "goal": goal,
        "pos": np.array(pos),
        "vel": np.array(vel),
        "actions": np.array(actions),
        "rates": np.array(rates),
    }


def metrics(episodes, ctrl_freq, success_radius):
    outcomes = [e["outcome"] for e in episodes]
    wins = [e for e in episodes if e["outcome"] == "success"]
    result = {f"{k}_rate": outcomes.count(k) / len(episodes) for k in OUTCOMES}
    result["time_to_goal_s"] = (
        float(np.mean([len(e["actions"]) / ctrl_freq for e in wins])) if wins else None
    )
    # the shortest path ends on the success sphere, so a straight flight scores 1
    efficiency = [
        (np.linalg.norm(e["goal"] - e["start"]) - success_radius)
        / np.sum(np.linalg.norm(np.diff(e["pos"], axis=0), axis=1))
        for e in wins
    ]
    result["path_efficiency"] = float(np.mean(efficiency)) if wins else None
    result["mean_angular_rate"] = float(np.mean(np.concatenate([e["rates"] for e in episodes])))
    deltas = [np.linalg.norm(np.diff(e["actions"], axis=0), axis=1) for e in episodes]
    result["mean_action_change"] = float(np.mean(np.concatenate(deltas)))
    return result


def plot_trajectories(episodes, out, title, n=12):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection="3d")
    for e in episodes[:n]:
        color = OUTCOMES[e["outcome"]]
        ax.plot(*e["pos"].T, color=color, linewidth=1.2)
        ax.scatter(*e["start"], color="gray", s=15)
        ax.scatter(*e["goal"], color=color, marker="*", s=80)
    for outcome, color in OUTCOMES.items():
        ax.plot([], [], color=color, label=outcome)
    ax.set_xlim(-2, 2), ax.set_ylim(-2, 2), ax.set_zlim(0, 2.5)
    ax.set_xlabel("x [m]"), ax.set_ylabel("y [m]"), ax.set_zlabel("z [m]")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_episode(e, ctrl_freq, out, title):
    t = np.arange(len(e["pos"])) / ctrl_freq
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for i, (axis, color) in enumerate(zip("xyz", ["tab:blue", "tab:orange", "tab:green"])):
        axes[0].plot(t, e["pos"][:, i], color=color, label=axis)
        axes[0].axhline(e["goal"][i], color=color, linestyle="--", linewidth=1)
        axes[1].plot(t, e["vel"][:, i], color=color, label=f"v{axis}")
    for i in range(e["actions"].shape[1]):
        axes[2].plot(t[1:], e["actions"][:, i], label=f"a{i}")
    for ax, label in zip(axes, ["pozycja [m]", "prędkość [m/s]", "akcja"]):
        ax.set_ylabel(label)
        ax.grid(alpha=0.4)
        ax.legend(loc="upper right", ncol=4)
    axes[0].set_title(f"{title}, ziarno {e['seed']}: {e['outcome']}")
    axes[-1].set_xlabel("czas [s]")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def add_marker(env, pos, radius, rgba):
    # visual shape only, the drone cannot collide with it
    shape = p.createVisualShape(
        p.GEOM_SPHERE, radius=radius, rgbaColor=rgba, physicsClientId=env.CLIENT
    )
    p.createMultiBody(baseVisualShapeIndex=shape, basePosition=pos, physicsClientId=env.CLIENT)


def record_video(env, act, seed, out, size=(640, 480)):
    w, h = size
    proj = p.computeProjectionMatrixFOV(60, w / h, 0.05, 20)
    frames, view = [], []

    def capture():
        if not view:
            # right after the reset: mark start and goal, look at both from the side
            start, goal = env.pos[0].copy(), env.goal
            add_marker(env, goal, env.SUCCESS_RADIUS, [0, 0.8, 0, 0.4])
            add_marker(env, start, 0.03, [0.5, 0.5, 0.5, 0.8])
            dx, dy = (goal - start)[:2]
            side = np.array([-dy, dx, 0]) if np.hypot(dx, dy) > 0.1 else np.array([1.0, 0, 0])
            dist = 0.5 + np.linalg.norm(goal - start)
            center = (goal + start) / 2
            eye = center + dist * side / np.linalg.norm(side) + [0, 0, 0.3 * dist]
            view.append(p.computeViewMatrix(eye, center, [0, 0, 1]))
        rgb = p.getCameraImage(
            w, h, view[0], proj, renderer=p.ER_TINY_RENDERER, physicsClientId=env.CLIENT
        )[2]
        frames.append(np.reshape(np.asarray(rgb, dtype=np.uint8), (h, w, 4))[:, :, :3])

    run_episode(env, act, seed, on_frame=capture)
    imageio.mimsave(out, frames, fps=env.CTRL_FREQ, macro_block_size=1)


def evaluate_policy(spec, env_overrides, episodes, out=None, plots=False, video=False):
    variant, name, act, kwargs = load_policy(spec)
    env = GoalReachAviary(**{**kwargs, **env_overrides})
    seeds = range(EVAL_SEED, EVAL_SEED + episodes)
    runs = [run_episode(env, act, s) for s in seeds]
    result = metrics(runs, env.CTRL_FREQ, env.SUCCESS_RADIUS)
    if out is not None and (plots or video):
        slug = name.replace(" ", "_").replace("=", "")
        (out / slug).mkdir(parents=True, exist_ok=True)
        if plots:
            plot_trajectories(runs, out / slug / "trajectories.png", name)
            plot_episode(runs[0], env.CTRL_FREQ, out / slug / "episode.png", name)
        if video:
            record_video(env, act, EVAL_SEED, out / slug / "episode.mp4")
    env.close()
    per_episode = [
        {"seed": r["seed"], "outcome": r["outcome"], "steps": len(r["actions"])} for r in runs
    ]
    return {
        "variant": variant,
        "name": name,
        "spec": spec,
        "env_kwargs": {**kwargs, **env_overrides},
        "metrics": result,
        "episodes": per_episode,
    }


COLUMNS = {
    "success_rate": ("SR", 100, ".0f"),
    "crash_rate": ("rozbicia", 100, ".0f"),
    "out_of_bounds_rate": ("poza obszarem", 100, ".0f"),
    "timeout_rate": ("timeout", 100, ".0f"),
    "time_to_goal_s": ("czas dolotu [s]", 1, ".2f"),
    "path_efficiency": ("efektywność ścieżki", 1, ".2f"),
    "mean_angular_rate": ("‖ω‖ [rad/s]", 1, ".2f"),
    "mean_action_change": ("‖Δa‖", 1, ".3f"),
}


def table(results):
    """Markdown table, mean ± std over the policies (seeds) of each variant."""
    groups = defaultdict(list)
    for r in results:
        groups[r["variant"]].append(r)
    head = ["wariant", "n"] + [c[0] for c in COLUMNS.values()] + ["SR per ziarno"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for variant, rs in groups.items():
        cells = [variant, str(len(rs))]
        for key, (_, scale, spec) in COLUMNS.items():
            values = [r["metrics"][key] for r in rs if r["metrics"][key] is not None]
            if not values:
                cells.append("-")
            elif len(values) == 1:
                cells.append(format(scale * values[0], spec))
            else:
                mean, std = scale * np.mean(values), scale * np.std(values)
                cells.append(f"{mean:{spec}} ± {std:{spec}}")
        cells.append(", ".join(f"{100 * r['metrics']['success_rate']:.0f}" for r in rs))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policies", nargs="+", help='run directories, checkpoints or "p[:k_p]"')
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=TEST_EPISODES)
    parser.add_argument(
        "--env",
        nargs="+",
        default=[],
        metavar="KEY=VALUE",
        help="overrides of the environment kwargs from the run config",
    )
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    overrides = {k: yaml.safe_load(v) for k, v in (kv.split("=", 1) for kv in args.env)}
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for spec in args.policies:
        result = evaluate_policy(spec, overrides, args.episodes, args.out, args.plots, args.video)
        sr = result["metrics"]["success_rate"]
        print(f"{result['name']}: SR {100 * sr:.0f}%", flush=True)
        results.append(result)
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2))
    text = table(results)
    (args.out / "metrics.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
