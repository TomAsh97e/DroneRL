"""System identification in SITL (plan/M4.md, M4.9). The response of PX4 to steps of the policy
commands is fitted with the dynamics of NavEnv: after a delay the velocity follows the command
with a first order lag and an acceleration limit, the yaw rate with a first order lag. Also the
tilt against the acceleration (the model of M3.7) and the latency of the chain from a camera
frame to the setpoint of PX4. Reads the records of sim/sysid_flights.py, writes sysid.json and
sysid.png next to them.

    python -m dronerl.eval.sysid runs/m4/sysid/<time>
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from dronerl.envs.nav_gpu import NOMINAL, RANGES, TILT_TAU, G

DT = 0.005  # s, step of the simulated response
BEFORE, AFTER = 0.2, 2.0  # s of a window around a step edge
TAU = np.arange(0.02, 1.0, 0.01)
ACCEL = np.arange(0.5, 6.01, 0.25)
DELAY = np.arange(0.0, 0.301, 0.01)
NAMES = ("v_fwd", "v_lat", "yaw_rate")


def euler(q):
    """Roll, pitch, yaw of quaternions [N, 4] (w, x, y, z); in FLU a positive pitch is nose down."""
    w, x, y, z = q.T
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return roll, pitch, yaw


def motion(gt):
    """From ground truth rows (t, x, y, z, qw, qx, qy, qz): time, the velocity and acceleration in
    the heading frame [N, 2] (forward, left), the yaw rate, roll and pitch."""
    t = gt[:, 0]
    roll, pitch, yaw = euler(gt[:, 4:8])
    vel = np.gradient(gt[:, 1:3], t, axis=0)
    acc = np.gradient(vel, t, axis=0)
    c, s = np.cos(yaw), np.sin(yaw)
    turn = np.stack([np.stack([c, s], -1), np.stack([-s, c], -1)], 1)  # world -> heading
    rate = np.gradient(np.unwrap(yaw), t)
    return t, turn @ vel[..., None], turn @ acc[..., None], rate, roll, pitch


def edges(plan):
    """Step edges of the schedule: (time, axis, command before, command after, label)."""
    out = []
    for t0, t1, *cmd in plan:
        axis = int(np.flatnonzero(cmd)[0])
        value = cmd[axis]
        shown = f"{value:+.2f}" if axis < 2 else f"{math.degrees(value):+.0f}"
        label = f"{NAMES[axis]} {shown}"
        out.append((t0, axis, 0.0, value, label))
        out.append((t1, axis, value, 0.0, label))
    return out


def respond(times, edge, v0, old, new, tau, accel, delay):
    """Response of the model [P, len(times)] for P parameter sets: from v0 at times[0], the
    target jumps from old to new at edge + delay."""
    tau, accel, delay = np.broadcast_arrays(*map(np.atleast_1d, (tau, accel, delay)))
    alpha, limit = 1 - np.exp(-DT / tau), accel * DT
    steps = np.round((times - times[0]) / DT).astype(int)
    v, out = np.full(len(tau), float(v0)), np.empty((len(tau), len(times)))
    j = 0
    for k in range(steps[-1] + 1):
        while j < len(steps) and steps[j] == k:
            out[:, j] = v
            j += 1
        target = np.where(times[0] + k * DT >= edge + delay, new, old)
        v = v + np.clip((target - v) * alpha, -limit, limit)
    return out


def fit(windows, tau=TAU, accel=ACCEL, delay=DELAY):
    """Least squares over a grid of (tau, accel, delay) for windows [(times, measured, edge, old,
    new)]: the best set, its RMS error and the accel values within 5% of the best error, as the
    limit is not seen when the lag alone keeps the acceleration lower."""
    grid = np.stack(np.meshgrid(tau, accel, delay, indexing="ij"), -1).reshape(-1, 3)
    sse, n = np.zeros(len(grid)), 0
    for times, meas, edge, old, new in windows:
        sim = respond(times, edge, meas[0], old, new, *grid.T)
        sse += ((sim - meas) ** 2).sum(1)
        n += len(times)
    best = int(np.argmin(sse))
    near = grid[sse <= 1.05 * sse[best], 1]
    return {
        "tau": float(grid[best, 0]),
        "accel": float(grid[best, 1]),
        "delay": float(grid[best, 2]),
        "rms": float(math.sqrt(sse[best] / n)),
        "accel_range": [float(near.min()), float(near.max())],
        "windows": len(windows),
    }


def step_windows(rec):
    """Windows of all step edges with the measured response on the axis of the step, each up to
    the next edge."""
    t, vel, _, rate, _, _ = motion(rec["gt"])
    signals = (vel[:, 0, 0], vel[:, 1, 0], rate)
    steps = edges(rec["plan"])
    ends = [e[0] for e in steps[1:]] + [math.inf]
    out = []
    for (edge, axis, old, new, label), end in zip(steps, ends, strict=True):
        keep = (t >= edge - BEFORE) & (t < min(edge + AFTER, end))
        out.append((label, axis, (t[keep], signals[axis][keep], edge, old, new)))
    return out


def tilt_lag(rec, taus=np.arange(0.0, 0.401, 0.005)):
    """First order lag of the tilt behind atan(a / g), as NavEnv models it, over the steps."""
    t, _, acc, _, roll, pitch = motion(rec["gt"])
    keep = (t >= rec["plan"][0, 0]) & (t <= rec["plan"][-1, 1] + 2.0)
    target = np.stack([np.arctan(acc[:, 0, 0] / G), -np.arctan(acc[:, 1, 0] / G)], -1)[keep]
    meas = np.stack([pitch, roll], -1)[keep]
    dt = np.diff(t[keep], prepend=t[keep][0])
    alpha = 1 - np.exp(-dt[:, None] / np.maximum(taus, 1e-6))  # [N, taus]
    tilt, sse = np.repeat(target[:1], len(taus), 0), np.zeros(len(taus))
    for k in range(len(target)):
        tilt += (target[k] - tilt) * alpha[k, :, None]
        sse += ((tilt - meas[k]) ** 2).sum(1)
    errors = np.sqrt(sse / target.size)
    best = int(np.argmin(errors))
    gain = float((meas * target).sum() / (target**2).sum())
    return {
        "tau": float(taus[best]),
        "rms_deg": math.degrees(errors[best]),
        "rms_without_lag_deg": math.degrees(errors[0]),
        "gain": gain,
        "max_tilt_deg": math.degrees(float(np.abs(meas).max())),
        "range": list(TILT_TAU),
    }


def percentiles(x):
    x = np.asarray(x) * 1000
    return {"p50": float(np.median(x)), "p95": float(np.percentile(x, 95)), "max": float(x.max())}


def latency(recs):
    """Frame -> student image -> command -> setpoint to PX4 [ms] over policy flights, in
    simulation and wall time."""
    parts = {k: {"sim": [], "wall": []} for k in ("depth", "policy", "bridge", "total")}
    for rec in recs:
        images, cmds, setpoints, clock = rec["images"], rec["cmds"], rec["setpoints"], rec["clock"]
        at_image = {round(stamp, 6): row for stamp, *row in images}
        for stamp, sim, wall in cmds:
            image = at_image.get(round(stamp, 6))
            k = np.searchsorted(setpoints[:, 2], wall)
            if image is None or k == len(setpoints) or setpoints[k, 2] - wall > 0.1:
                continue
            frame = {"sim": stamp, "wall": float(np.interp(stamp, clock[:, 0], clock[:, 1]))}
            times = {
                "sim": (image[0], sim, setpoints[k, 1]),
                "wall": (image[1], wall, setpoints[k, 2]),
            }
            for clk, (t_image, t_cmd, t_sp) in times.items():
                parts["depth"][clk].append(t_image - frame[clk])
                parts["policy"][clk].append(t_cmd - t_image)
                parts["bridge"][clk].append(t_sp - t_cmd)
                parts["total"][clk].append(t_sp - frame[clk])
    out = {k: {clk: percentiles(v) for clk, v in p.items()} for k, p in parts.items()}
    out["n"] = len(parts["total"]["sim"])
    return out


def inside(name, value):
    lo, hi = RANGES[name]
    return {
        "value": value,
        "nominal": NOMINAL[name],
        "range": [lo, hi],
        "inside": lo <= value <= hi,
    }


def plot(windows, fits, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = {"v_fwd": 0, "v_lat": 1, "yaw_rate": 2}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for (name, axis), ax in zip(groups.items(), axes, strict=True):
        f = fits["horizontal" if axis < 2 else "yaw"]
        for _, a, (times, meas, edge, old, new) in windows:
            if a != axis or new == old:
                continue
            scale = new - old
            sim = respond(times, edge, meas[0], old, new, f["tau"], f["accel"], f["delay"])[0]
            ax.plot(times - edge, (meas - old) / scale, color="C0", alpha=0.4, lw=1)
            ax.plot(times - edge, (sim - old) / scale, color="C1", alpha=0.4, lw=1, ls="--")
        ax.set_title(f"{name}: tau {f['tau']:.2f} s, delay {f['delay']:.2f} s")
        ax.set_xlabel("s from the step")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("normalized response (SITL solid, model dashed)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dir", type=Path, help="output of sim/sysid_flights.py")
    args = parser.parse_args()

    result = {}
    steps = args.dir / "steps" / "steps.npz"
    if steps.exists():
        rec = dict(np.load(steps))
        windows = step_windows(rec)
        horizontal = [w for _, a, w in windows if a < 2]
        yaw = [w for _, a, w in windows if a == 2]
        fits = {
            "horizontal": fit(horizontal),
            "v_fwd": fit([w for _, a, w in windows if a == 0]),
            "v_lat": fit([w for _, a, w in windows if a == 1]),
            "yaw": fit(yaw, accel=[1e3]),
        }
        for label in dict.fromkeys(label for label, _, _ in windows):
            group = [w for lb, a, w in windows if lb == label]
            fits[label] = fit(group, accel=[1e3]) if "yaw" in label else fit(group)
        result["fits"] = fits
        h, y = fits["horizontal"], fits["yaw"]
        result["navenv"] = {
            "tau_v": inside("tau_v", h["tau"]),
            "accel": inside("accel", h["accel"]) | {"accel_range": h["accel_range"]},
            "delay": inside("delay", h["delay"]),
            "tau_yaw": inside("tau_yaw", y["tau"]),
            "delay_yaw": inside("delay", y["delay"]),
        }
        result["tilt"] = tilt_lag(rec)
        plot(windows, fits, args.dir / "sysid.png")
    recs = [dict(np.load(p)) for p in sorted(args.dir.glob("latency_p*/latency.npz"))]
    if recs:
        result["latency_ms"] = latency(recs)
    (args.dir / "sysid.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
