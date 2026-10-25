"""Velocity step tracking of the `vel` action (M1.4) and hover drift of the `rpm` action."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pybullet as p  # noqa: E402

from dronerl.envs.goal_reach import GoalReachAviary  # noqa: E402

# (name, direction, duration in s), each command is followed by 2 s of zero velocity
STEPS = [("x", [1, 0, 0], 2.0), ("diagonal", [1, 1, 0], 2.0), ("up", [0, 0, 1], 1.0)]
HOLD_SEC = 2.0
START = [-1.5, -1.5, 0.5]


def place(env, pos):
    p.resetBasePositionAndOrientation(
        env.DRONE_IDS[0], pos, [0, 0, 0, 1], physicsClientId=env.CLIENT
    )
    p.resetBaseVelocity(env.DRONE_IDS[0], [0, 0, 0], [0, 0, 0], physicsClientId=env.CLIENT)


def command_profile(ctrl_freq, v_max):
    cmds, segments = [], []
    for name, direction, sec in STEPS:
        v = v_max * np.array(direction, dtype=float) / np.linalg.norm(direction)
        segments.append((name, len(cmds), len(cmds) + int(sec * ctrl_freq), v))
        cmds += [v] * int(sec * ctrl_freq) + [np.zeros(3)] * int(HOLD_SEC * ctrl_freq)
    return np.array(cmds), segments


def run_steps(ctrl_freq, v_max=1.0):
    env = GoalReachAviary(act="vel", v_max=v_max, ctrl_freq=ctrl_freq)
    env.reset(seed=0)
    place(env, START)
    cmds, segments = command_profile(ctrl_freq, v_max)
    vel, rates = [], []
    # termination is ignored on purpose, the drone may leave the GoalReach area here
    for v in cmds:
        env.step(v / v_max)
        vel.append(env.vel[0].copy())
        rates.append(np.linalg.norm(env.ang_v[0]))
    env.close()
    return cmds, np.array(vel), np.array(rates), segments


def step_metrics(cmds, vel, segments, ctrl_freq, v_max):
    dt = 1 / ctrl_freq
    tail = int(0.5 * ctrl_freq)
    results = {}
    for name, start, end, v in segments:
        target = np.linalg.norm(v)
        unit = v / target
        along = vel[start:end] @ unit
        reached = np.nonzero(along >= 0.632 * target)[0]
        hold = vel[end : end + int(HOLD_SEC * ctrl_freq)]
        results[name] = {
            "tau_s": float((reached[0] + 1) * dt) if len(reached) else None,
            "overshoot_pct": float(100 * max(0.0, along.max() / target - 1)),
            "steady_error_pct_vmax": float(
                100 * np.mean(np.linalg.norm(vel[end - tail : end] - v, axis=1)) / v_max
            ),
            "cross_axis_max": float(
                np.max(np.linalg.norm(vel[start:end] - np.outer(along, unit), axis=1))
            ),
            "stop_residual_pct_vmax": float(
                100 * np.mean(np.linalg.norm(hold[-tail:], axis=1)) / v_max
            ),
        }
    return results


def plot_steps(cmds, vel, ctrl_freq, out, title):
    t = np.arange(len(cmds)) / ctrl_freq
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for i, (ax, label) in enumerate(zip(axes, ["vx", "vy", "vz"])):
        ax.plot(t, cmds[:, i], "k--", label="zadana")
        ax.plot(t, vel[:, i], label="rzeczywista")
        ax.set_ylabel(f"{label} [m/s]")
        ax.grid(alpha=0.4)
    axes[0].legend(loc="upper right")
    axes[0].set_title(title)
    axes[-1].set_xlabel("czas [s]")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def hover_drift(kick=0.0, seconds=8.0, seed=0):
    """Zero `rpm` action after a single step with motor 0 at (1 + 0.05 * kick) * HOVER_RPM."""
    env = GoalReachAviary(act="rpm")
    env.reset(seed=seed)
    start = env.pos[0].copy()
    pos, rpy = [], []
    for step in range(int(seconds * env.CTRL_FREQ)):
        env.step(np.array([kick if step == 0 else 0.0, 0, 0, 0]))
        pos.append(env.pos[0].copy())
        rpy.append(env.rpy[0].copy())
    env.close()
    drift = np.array(pos) - start
    return {
        "kick": kick,
        "seconds": seconds,
        "final_drift_xyz": drift[-1].round(4).tolist(),
        "max_horizontal_drift": float(np.max(np.linalg.norm(drift[:, :2], axis=1))),
        "max_abs_rpy": np.max(np.abs(rpy), axis=0).round(4).tolist(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ctrl-freq", type=int, nargs="+", default=[48])
    parser.add_argument("--out", default="runs/m1/velocity_step")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"hover_drift_rpm": [hover_drift(), hover_drift(kick=0.1)]}
    for freq in args.ctrl_freq:
        cmds, vel, rates, segments = run_steps(freq)
        metrics = step_metrics(cmds, vel, segments, freq, v_max=1.0)
        metrics["max_angular_rate"] = float(rates.max())
        report[f"ctrl_{freq}hz"] = metrics
        plot_steps(cmds, vel, freq, out / f"steps_{freq}hz.png", f"Skoki prędkości, {freq} Hz")
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
