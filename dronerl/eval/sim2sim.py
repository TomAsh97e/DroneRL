"""Sim2sim evaluation (plan/M4.md, M4.11): the flights of sim/sim2sim_flights.py against the same
start-goal pairs in NavEnv. NavEnv plays every pair `repeats` times in each condition of TWINS,
which gives a success rate per pair to set against the one flight in Gazebo on it. The gap of the
student splits into dynamics and control (NavEnv -> gz_depth) and perception (gz_depth ->
gz_rgb). Writes metrics.json, metrics.md and top-down plots of all failed flights.

    python -m dronerl.eval.sim2sim
"""

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from dronerl.envs import maps as mapgen
from dronerl.envs import nav_gpu
from dronerl.envs.nav_gpu import CONDITIONS, NavEnv
from dronerl.eval import nav_plots
from dronerl.eval.nav_eval import OUTCOMES, load_policy, metrics, rollout

WORLDS = ("arena", "corridor", "room")
MAPS = Path("sim/gazebo/worlds")
POLICIES = {
    "astar": "astar",
    "asym_s0": "runs/m3/asym/asym_nav-v2_s0_20261028-020943/checkpoints/best.pt",
    "asym_s1": "runs/m3/asym/asym_nav-v2_s1_20261028-043342/checkpoints/best.pt",
    "asym_s2": "runs/m3/asym/asym_nav-v2_s2_20261028-065723/checkpoints/best.pt",
    "dagger_s0": "runs/m3/student/main_nav-v2_s0_20261027-191054/checkpoints/best.pt",
    "asym_ft_s0": "runs/m4/asym_ft/asym_ft_nav-v2_s0_20261029-112310/checkpoints/best.pt",
    "asym_ft_s1": "runs/m4/asym_ft/asym_ft_nav-v2_s1_20261029-122140/checkpoints/best.pt",
    "asym_ft_s2": "runs/m4/asym_ft/asym_ft_nav-v2_s2_20261029-132023/checkpoints/best.pt",
}
# NavEnv conditions: test as the students were trained and evaluated in M3, test with the delay
# range corrected in M4.9, clean
TWINS = ("test_m3", "test", "clean")
M3_DELAY = (0.0, 0.12)
REPEATS = 10
GAP_PP = 15  # gz_rgb this far below NavEnv starts M4.12
MIN_SUCCESS = 14  # of 20 flights in each world, criterion 1 of M4


def series_of(name):
    """(policy, condition) of a series: asym_s0_gz_rgb -> (asym_s0, gz_rgb), astar -> (astar, gz)"""
    policy, _, rest = name.partition("_gz_")
    return policy, f"gz_{rest}" if rest else "gz"


def load_flights(root, world, pairs):
    """The flights of one series in one world: their run.npz and result.json (with the error of
    the PX4 estimate under "estimate"), the pairs flown and the attempts lost to errors of the
    infrastructure."""
    runs, results, flown, errors = [], [], [], 0
    for pair in range(pairs):
        out = root / f"{world}_p{pair:02d}"
        errors += len(list(root.glob(f"{out.name}_error*")))
        path = out / "result.json"
        result = json.loads(path.read_text()) if path.exists() else None
        if result is None or result["outcome"] == "aborted":
            continue
        with np.load(out / "run.npz") as f:
            runs.append(dict(f))
        estimate = out / "estimate.json"
        result["estimate"] = json.loads(estimate.read_text()) if estimate.exists() else None
        results.append(result)
        flown.append(pair)
    return runs, results, np.array(flown, dtype=int), errors


def stack(runs):
    """Flights as the arrays of nav_eval.rollout, time first, padded after the end."""
    steps, n = max(int(r["length"]) for r in runs), len(runs)
    pos, yaw = np.zeros((steps + 1, n, 2)), np.zeros((steps + 1, n))
    actions, alive = np.zeros((steps, n, 3)), np.zeros((steps, n), bool)
    for i, r in enumerate(runs):
        k = int(r["length"])
        pos[: k + 1, i], pos[k + 1 :, i] = r["pos"], r["pos"][-1]
        yaw[: k + 1, i], yaw[k + 1 :, i] = r["yaw"], r["yaw"][-1]
        actions[:k, i], alive[:k, i] = r["actions"], True
    out = {k: np.array([bool(r[k]) for r in runs]) for k in OUTCOMES}
    for key in ("length", "path_length", "min_clearance"):
        out[key] = np.array([float(r[key]) for r in runs])
    out["return"] = np.array([float(r["reward"].sum()) for r in runs])
    return out | {"pos": pos, "yaw": yaw, "actions": actions, "alive": alive}


def subset(pool, ids):
    return {k: v[ids] for k, v in pool.items()}


def pick(x, world):
    """Per pair values of one world, or of all worlds after each other."""
    return x[world] if world != "all" else np.concatenate([x[w] for w in WORLDS])


def gazebo_metrics(runs, results, pool, errors):
    m = metrics(stack(runs), pool)
    p95 = [r["latency_ms"]["wall_ms"]["p95"] for r in results if r["latency_ms"]["wall_ms"]]
    events = {}
    for r in results:
        for k, v in r["safety_events"].items():
            events[k] = events.get(k, 0) + v
    est = [r["estimate"] for r in results if r["estimate"]]
    final = [e["position_m"]["final"] for e in est]
    heading = [e["heading_deg"]["max"] for e in est]
    return m | {
        "estimate_final_m": float(np.median(final)) if final else None,
        "estimate_final_max_m": float(np.max(final)) if final else None,
        "estimate_heading_max_deg": float(np.median(heading)) if heading else None,
        "flights": len(runs),
        "errors": errors,
        "latency_wall_p95_ms": float(np.median(p95)) if p95 else None,
        "real_time_factor": float(np.mean([r["real_time_factor"] for r in results])),
        "safety_events": events,
    }


@contextmanager
def delay_range(bounds):
    old = nav_gpu.RANGES["delay"]
    nav_gpu.RANGES["delay"] = bounds
    try:
        yield
    finally:
        nav_gpu.RANGES["delay"] = old


def twin(spec, pool, condition, device, repeats):
    """NavEnv on the pairs of a world, each played `repeats` times: metrics and the success rate
    of every pair."""
    tiled = mapgen.concat([pool] * repeats)
    _, _, make = load_policy(spec, device)
    sensors = CONDITIONS["test" if condition == "test_m3" else condition]
    bounds = M3_DELAY if condition == "test_m3" else nav_gpu.RANGES["delay"]
    with delay_range(bounds):
        randomize = condition != "clean"
        env = NavEnv(tiled, device=device, autoreset=False, randomize=randomize, student=sensors)
        run = rollout(env, make(env, tiled))
    return metrics(run, tiled), run["success"].reshape(repeats, -1).mean(0)


def paired(gz, p):
    """One flight per pair (success, NaN where none) against the success rates p of NavEnv."""
    ok = ~np.isnan(gz)
    g, q = gz[ok], p[ok]
    if not ok.any():
        return {"pairs": 0}
    return {
        "pairs": int(ok.sum()),
        "gz_sr": float(g.mean()),
        "navenv_sr": float(q.mean()),
        "gap_pp": float(100 * (q.mean() - g.mean())),
        # pairs NavEnv nearly always wins and the flight lost, and the other way round
        "lost": int(((q >= 0.8) & (g == 0)).sum()),
        "gained": int(((q <= 0.2) & (g == 1)).sum()),
    }


def pct(x):
    return "-" if x is None else f"{100 * x:.0f}%"


def num(x, fmt=".2f"):
    return "-" if x is None else format(x, fmt)


def cell(p):
    if not p["pairs"]:
        return "-"
    return (
        f"{pct(p['gz_sr'])} / {pct(p['navenv_sr'])}, {p['gap_pp']:+.0f}, {p['lost']}/{p['gained']}"
    )


def report(res):
    lines = ["## Przeloty w Gazebo", ""]
    lines += [
        "| seria | świat | przeloty | błędy infra | SR | CR | poza | timeout | czas do celu [s] "
        "| min. odstęp [m] | opóźnienie p95 [ms] | RTF | błąd EKF na końcu [m] "
        "| max błąd kursu [°] |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, worlds in res["gazebo"].items():
        for world, m in worlds.items():
            ttg = num(m["time_to_goal_s"], ".1f")
            lat = num(m["latency_wall_p95_ms"], ".0f")
            ekf = f"{num(m['estimate_final_m'])} / {num(m['estimate_final_max_m'])}"
            heading = num(m["estimate_heading_max_deg"], ".0f")
            lines.append(
                f"| {name} | {world} | {m['flights']} | {m['errors']} | {pct(m['success_rate'])} "
                f"| {pct(m['collision_rate'])} | {pct(m['out_of_area_rate'])} "
                f"| {pct(m['timeout_rate'])} | {ttg} | {m['min_clearance']:.2f} | {lat} "
                f"| {m['real_time_factor']:.2f} | {ekf} | {heading} |"
            )
    lines += ["", "Błąd EKF: mediana / maksimum po przelotach; kurs: mediana maksimów."]
    lines += ["", f"## NavEnv na tych samych parach (każda para {res['repeats']} razy)", ""]
    lines += [
        "| polityka | warunki | " + " | ".join([*WORLDS, "all"]) + " |",
        "|---|---|" + "---|" * 4,
    ]
    for policy, conds in res["navenv"].items():
        for cond, worlds in conds.items():
            cells = " | ".join(pct(worlds[w]["success_rate"]) for w in [*WORLDS, "all"])
            lines.append(f"| {policy} | {cond} | {cells} |")
    lines += [
        "",
        "## Porównanie parami (SR Gazebo / SR NavEnv, luka w pp, pary przegrane / wygrane)",
    ]
    lines += [
        "",
        "| seria | NavEnv | " + " | ".join([*WORLDS, "all"]) + " |",
        "|---|---|" + "---|" * 4,
    ]
    for name, conds in res["paired"].items():
        for cond, worlds in conds.items():
            cells = " | ".join(cell(worlds[w]) for w in [*WORLDS, "all"])
            lines.append(f"| {name} | {cond} | {cells} |")
    if "gap" in res:
        lines += ["", f"## Rozbicie luki {res['student']} s0 (pary z oboma przelotami)", ""]
        lines += [
            "| świat | NavEnv test_m3 | NavEnv test | gz_depth | gz_rgb | dynamika [pp] "
            "| percepcja [pp] |",
            "|---|---|---|---|---|---|---|",
        ]
        for world, g in res["gap"].items():
            lines.append(
                f"| {world} | {pct(g['test_m3'])} | {pct(g['test'])} | {pct(g['gz_depth'])} "
                f"| {pct(g['gz_rgb'])} | {g['dynamics_pp']:+.0f} | {g['perception_pp']:+.0f} |"
            )
    if "seeds" in res:
        lines += ["", f"## {res['student']} s0-s2 w gz_rgb (średnia ± odchylenie)", ""]
        lines.append("| " + " | ".join([*WORLDS, "all"]) + " |")
        lines.append("|" + "---|" * 4)
        lines.append("| " + " | ".join(res["seeds"].get(w, "-") for w in [*WORLDS, "all"]) + " |")
    if res.get("failures"):
        lines += ["", "## Porażki w Gazebo z błędem estymaty", ""]
        lines += [
            "| seria | świat | para | wynik | czas [s] | błąd EKF na końcu [m] "
            "| max błąd kursu [°] |",
            "|---|---|---|---|---|---|---|",
        ]
        for f in res["failures"]:
            e = f["estimate"] or {"position_m": {"final": None}, "heading_deg": {"max": None}}
            lines.append(
                f"| {f['series']} | {f['world']} | {f['pair']} | {f['outcome']} "
                f"| {f['time_s']:.1f} "
                f"| {num(e['position_m']['final'])} | {num(e['heading_deg']['max'], '.0f')} |"
            )
    if "criteria" in res:
        c = res["criteria"]
        lines += ["", "## Kryteria", ""]
        lines.append(
            f"- {res['student']} s0 w gz_rgb, sukcesy na świat: {c['successes']} (wymagane "
            f"{MIN_SUCCESS} z 20): {'spełnione' if c['sr_met'] else 'niespełnione'}"
        )
        for name, gaps in c["gaps_pp"].items():
            m3, test = ("-" if x is None else f"{x:+.0f}" for x in gaps.values())
            lines.append(
                f"- {name}: luka do NavEnv test_m3 {m3} pp, test {test} pp "
                f"(próg M4.12: {GAP_PP} pp)"
            )
        lines.append(f"- M4.12 potrzebne: {'tak' if c['m4_12'] else 'nie'}")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--flights", type=Path, default=Path("runs/m4/sim2sim/flights"))
    parser.add_argument("--out", type=Path, default=Path("runs/m4/sim2sim"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=REPEATS)
    # the students of the criteria and the gap: asym of M3, asym_ft after M4.12
    parser.add_argument("--student", default="asym")
    args = parser.parse_args(argv)
    s0 = f"{args.student}_s0"

    args.out.mkdir(parents=True, exist_ok=True)
    pools = {w: dict(np.load(MAPS / w / "maps.npz")) for w in WORLDS}
    names = sorted(p.name for p in args.flights.iterdir() if p.is_dir())
    res = {"repeats": args.repeats, "student": args.student, "gazebo": {}, "navenv": {}}
    res |= {"paired": {}, "failures": []}
    success = {}  # series -> world -> success per pair, NaN without a flight
    for name in names:
        res["gazebo"][name], success[name] = {}, {}
        every = {"runs": [], "results": [], "pools": []}
        for world in WORLDS:
            pool = pools[world]
            runs, results, flown, errors = load_flights(
                args.flights / name, world, len(pool["start"])
            )
            gz = np.full(len(pool["start"]), np.nan)
            gz[flown] = [r["outcome"] == "success" for r in results]
            success[name][world] = gz
            if not runs:
                continue
            sub = subset(pool, flown)
            res["gazebo"][name][world] = gazebo_metrics(runs, results, sub, errors)
            every["runs"] += runs
            every["results"] += results
            every["pools"].append(sub)
            failed = np.flatnonzero([r["outcome"] != "success" for r in results])
            for i in failed:
                r = results[i]
                res["failures"].append(
                    {"series": name, "world": world, "pair": int(flown[i])}
                    | {k: r[k] for k in ("outcome", "time_s", "estimate")}
                )
            if len(failed):
                title = f"{name}, {world}, nieudane przeloty"
                path = args.out / f"failures_{name}_{world}.png"
                nav_plots.plot_runs(stack(runs), sub, path, title, failed)
        if every["runs"]:
            errors = sum(m["errors"] for m in res["gazebo"][name].values())
            pool = mapgen.concat(every["pools"])
            res["gazebo"][name]["all"] = gazebo_metrics(
                every["runs"], every["results"], pool, errors
            )

    rates = {}  # policy -> condition -> world -> success rate per pair
    for policy in dict.fromkeys(series_of(n)[0] for n in names):
        res["navenv"][policy], rates[policy] = {}, {}
        for cond in TWINS:
            res["navenv"][policy][cond], rates[policy][cond] = {}, {}
            for world in WORLDS:
                m, p = twin(POLICIES[policy], pools[world], cond, args.device, args.repeats)
                res["navenv"][policy][cond][world] = m | {"per_pair": p.tolist()}
                rates[policy][cond][world] = p
            every = np.concatenate([rates[policy][cond][w] for w in WORLDS])
            res["navenv"][policy][cond]["all"] = {"success_rate": float(every.mean())}
            print(f"NavEnv {policy} {cond}: SR {100 * every.mean():.1f}%", flush=True)

    for name in names:
        policy, _ = series_of(name)
        res["paired"][name] = {}
        for cond in TWINS:
            p = rates[policy][cond]
            res["paired"][name][cond] = {
                w: paired(pick(success[name], w), pick(p, w)) for w in [*WORLDS, "all"]
            }

    if {f"{s0}_gz_depth", f"{s0}_gz_rgb"} <= set(names):
        res["gap"] = {}
        for world in [*WORLDS, "all"]:
            depth = pick(success[f"{s0}_gz_depth"], world)
            rgb = pick(success[f"{s0}_gz_rgb"], world)
            both = ~np.isnan(depth) & ~np.isnan(rgb)
            if not both.any():
                continue
            g = {c: float(pick(rates[s0][c], world)[both].mean()) for c in ("test_m3", "test")}
            g |= {"gz_depth": float(depth[both].mean()), "gz_rgb": float(rgb[both].mean())}
            g["dynamics_pp"] = 100 * (g["test_m3"] - g["gz_depth"])
            g["perception_pp"] = 100 * (g["gz_depth"] - g["gz_rgb"])
            res["gap"][world] = g | {"pairs": int(both.sum())}

    seeds = [f"{args.student}_s{s}_gz_rgb" for s in range(3)]
    if set(seeds) <= set(names):
        res["seeds"] = {}
        for world in [*WORLDS, "all"]:
            x = [res["paired"][s]["test_m3"][world].get("gz_sr") for s in seeds]
            if None not in x:
                x = 100 * np.array(x)
                res["seeds"][world] = f"{x.mean():.1f} ± {x.std():.1f}"

    if f"{s0}_gz_rgb" in names:
        gz = success[f"{s0}_gz_rgb"]
        counts = {w: int(np.nansum(gz[w])) for w in WORLDS}
        rgb = [n for n in names if n.endswith("gz_rgb")]
        gaps = {
            n: {c: res["paired"][n][c]["all"].get("gap_pp") for c in ("test_m3", "test")}
            for n in rgb
        }
        met = all(v >= MIN_SUCCESS for v in counts.values())
        wide = (gaps[f"{s0}_gz_rgb"]["test_m3"] or 0) > GAP_PP
        res["criteria"] = {
            "successes": counts,
            "sr_met": met,
            "gaps_pp": gaps,
            "m4_12": not met or wide,
        }

    (args.out / "metrics.json").write_text(json.dumps(res, indent=2))
    text = report(res)
    (args.out / "metrics.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
