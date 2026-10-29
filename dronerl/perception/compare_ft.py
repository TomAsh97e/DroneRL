"""Zero-shot against the fine-tuned variants on the NYU test set (M3.4): tables and learning curves.

Mean ± std over seeds; the model for export is the run with the best validation AbsRel.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLS = ["abs_rel", "rmse", "d1", "silog", "abs_rel_0_2", "abs_rel_2_4", "abs_rel_4_10"]
VIEWS = ["full", "full_median", "reduced", "reduced_median"]
ZERO_SHOT = Path("runs/m3/depth/zero_shot/metrics.json")


def load_runs(root):
    runs = defaultdict(list)
    for d in sorted(root.glob("*_nyu_s*")):
        if not (d / "final.json").exists():
            continue
        with open(d / "eval.csv") as f:
            curve = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
        final = json.loads((d / "final.json").read_text())
        runs[d.name.split("_nyu_")[0]].append({"dir": d, "final": final, "curve": curve})
    return runs


def best_val(run):
    return min(row["abs_rel"] for row in run["curve"])


def table(zero_shot, runs):
    lines = ["| widok | wariant | " + " | ".join(COLS) + " |", "|---|---|" + "---|" * len(COLS)]
    for view in VIEWS:
        cells = [f"{zero_shot[view][c]:.3f}" for c in COLS]
        lines.append(f"| {view} | zero-shot | " + " | ".join(cells) + " |")
        for variant, rs in runs.items():
            x = np.array([[r["final"][view][c] for c in COLS] for r in rs])
            cells = [f"{m:.3f} ± {s:.3f}" for m, s in zip(x.mean(0), x.std(0))]
            lines.append(f"| {view} | {variant} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot(runs, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for (variant, rs), color in zip(runs.items(), ["C0", "C1"]):
        for i, r in enumerate(rs):
            epoch = [row["epoch"] for row in r["curve"]]
            label = variant if i == 0 else None
            ax[0].plot(epoch, [row["train_silog"] for row in r["curve"]], color=color, label=label)
            ax[1].plot(epoch, [row["abs_rel"] for row in r["curve"]], color=color, label=label)
    ax[0].set_ylabel("SILog, trening")
    ax[1].set_ylabel("AbsRel, walidacja")
    for a in ax:
        a.set_xlabel("epoka")
        a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/m3/depth_ft"))
    parser.add_argument("--zero-shot", type=Path, default=ZERO_SHOT)
    parser.add_argument("--res", default="518")
    args = parser.parse_args(argv)

    zero_shot = json.loads(args.zero_shot.read_text())[args.res]
    runs = load_runs(args.runs)
    best = min((r for rs in runs.values() for r in rs), key=best_val)
    lines = [
        f"# Dostrajanie na NYU, R = {args.res}, test 654 obrazy",
        "",
        table(zero_shot, runs),
        "",
        "Średnia ± odchylenie po ziarnach. Najlepsza epoka według AbsRel na walidacji: "
        + ", ".join(f"{v} {[r['final']['best_epoch'] for r in rs]}" for v, rs in runs.items())
        + ".",
        f"Model do eksportu (najlepszy AbsRel na walidacji {best_val(best):.3f}): "
        f"`{best['dir'] / 'checkpoints' / 'best'}`.",
    ]
    (args.runs / "comparison.md").write_text("\n".join(lines) + "\n")
    plot(runs, args.runs / "curves.png")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
