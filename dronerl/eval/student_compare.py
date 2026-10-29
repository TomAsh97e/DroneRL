"""Students against the teacher (M3.10, M3.11): SR of a student divided by the SR of the teacher in
the same condition, from metrics.json of nav_eval.py runs, one directory per condition.

Criterion of M3: the main student reaches 85% of the teacher's SR on the medium level in `test`
in at least 2 of 3 seeds.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

TEACHER = "curriculum_rand"
CONDITIONS = ["test", "test_active", "clean"]
THRESHOLD = 0.85


def ratios(results):
    """variant -> level -> SR ratios of its seeds."""
    teacher = next(r for r in results if r["variant"] == TEACHER)["levels"]
    out = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r["variant"] == TEACHER:
            continue
        for level, m in r["levels"].items():
            out[r["variant"]][level].append(m["success_rate"] / teacher[level]["success_rate"])
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="directory with one nav_eval output per condition")
    parser.add_argument("--main", default="main", help="variant checked against the criterion")
    args = parser.parse_args(argv)

    table = {}
    for c in CONDITIONS:
        table[c] = ratios(json.loads((args.root / c / "metrics.json").read_text()))
    levels = list(next(iter(table["test"].values())))
    head = ["wariant"] + [f"{c} {lvl}" for c in CONDITIONS for lvl in levels]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for variant in table["test"]:
        cells = [variant]
        for c in CONDITIONS:
            for lvl in levels:
                x = 100 * np.array(table[c][variant][lvl])
                cells.append(f"{x.mean():.1f} ± {x.std():.1f}")
        lines.append("| " + " | ".join(cells) + " |")

    seeds = table["test"][args.main]["medium"]
    passed = sum(r >= THRESHOLD for r in seeds)
    lines += [
        "",
        "SR ucznia / SR nauczyciela w tych samych warunkach [%], średnia ± odchylenie po ziarnach.",
        f"Kryterium M3 ({args.main}, medium, test): "
        + ", ".join(f"{100 * r:.1f}%" for r in seeds)
        + f", próg {100 * THRESHOLD:.0f}% spełniony w {passed} z {len(seeds)} ziaren.",
    ]
    text = "\n".join(lines)
    (args.root / "ratios.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
