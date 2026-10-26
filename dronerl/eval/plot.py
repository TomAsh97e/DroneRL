"""Learning curves (mean ± std over seeds) and a summary table for groups of runs."""

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402


def load_runs(patterns, metric="mean_return"):
    """Groups run directories by the `variant` stored in their config.yaml."""
    groups = defaultdict(list)
    dirs = sorted({Path(p) for pattern in patterns for p in glob.glob(pattern)})
    for d in dirs:
        if not (d / "eval.csv").exists():
            continue
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        with (d / "eval.csv").open() as f:
            rows = list(csv.DictReader(f))
        final = d / "final.json"
        run = {
            "seed": cfg["seed"],
            "step": np.array([float(r["step"]) for r in rows]),
            "time": np.array([float(r["time"]) for r in rows]),
            "values": np.array([float(r[metric]) for r in rows]),
            "final": json.loads(final.read_text()) if final.exists() else None,
        }
        groups[cfg["variant"]].append(run)
    return groups


def mean_curve(runs, x="step"):
    end = min(r[x][-1] for r in runs)
    grid = np.unique(np.concatenate([r[x] for r in runs]))
    grid = grid[grid <= end]
    ys = np.array([np.interp(grid, r[x], r["values"]) for r in runs])
    return grid, ys.mean(0), ys.std(0)


COLUMNS = ["variant", "seeds", "last eval", "best eval", "final (100 ep.)", "per seed", "min"]
LABELS = {"mean_return": "evaluation return", "success_rate": "success rate"}


def fmt(values, digits=1):
    return f"{np.mean(values):.{digits}f} ± {np.std(values):.{digits}f}"


def row(cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def table(groups, metric="mean_return"):
    digits = 2 if metric == "success_rate" else 1
    lines = [row(COLUMNS), row(["---"] * len(COLUMNS))]
    for name, runs in groups.items():
        runs = sorted(runs, key=lambda r: r["seed"])
        done = [r for r in runs if r["final"] and metric in r["final"]]
        finals = [r["final"][metric] for r in done]
        per_seed = ", ".join(f"s{r['seed']}: {r['final'][metric]:.{digits}f}" for r in done)
        cells = [
            name,
            len(runs),
            fmt([r["values"][-1] for r in runs], digits),
            fmt([r["values"].max() for r in runs], digits),
            fmt(finals, digits) if finals else "-",
            per_seed or "-",
            f"{np.mean([r['time'][-1] for r in runs]) / 60:.1f}",
        ]
        lines.append(row(cells))
    return "\n".join(lines)


def plot(groups, out, title="", threshold=None, metric="mean_return", x="step"):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, runs in groups.items():
        grid, mean, std = mean_curve(runs, x)
        if x == "time":
            grid = grid / 60
        ax.plot(grid, mean, label=f"{name} (n={len(runs)})")
        ax.fill_between(grid, mean - std, mean + std, alpha=0.2)
    if threshold is not None:
        ax.axhline(threshold, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("environment steps" if x == "step" else "wall-clock time [min]")
    ax.set_ylabel(LABELS.get(metric, metric))
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("patterns", nargs="+", help="glob patterns of run directories")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--title", default="")
    parser.add_argument("--threshold", type=float)
    parser.add_argument(
        "--metric", default="mean_return", help="eval.csv column, e.g. success_rate"
    )
    parser.add_argument("--x", choices=["step", "time"], default="step")
    args = parser.parse_args(argv)

    groups = load_runs(args.patterns, args.metric)
    if not groups:
        raise SystemExit("no runs match the given patterns")
    plot(groups, args.out, args.title, args.threshold, args.metric, args.x)
    text = table(groups, args.metric)
    args.out.with_suffix(".md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
