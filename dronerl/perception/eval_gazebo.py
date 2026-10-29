"""Depth model on Gazebo images (plan/M4.md, M4.10): the zero-shot and the fine-tuned model of M3 on
the frames of sim/depth_frames.py. Metrics as on NYU (M3.3) on the crop the policy sees, at full
size and reduced to the policy image, then the error of each model against the mono noise of the
student training (M3.6): the check of the fit at the fitted amplitude, and whether the error lies
between the noise at the ends of the training amplitude range, of either sign. The error of the
chosen model also gets its own noise profile (M4.12): fitted on every other flight, checked on the
rest, written to runs/m4/noise/gazebo.json.

    python -m dronerl.perception.eval_gazebo
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dronerl.envs.depth_noise import AMP, DepthNoise
from dronerl.perception import metrics, noise_fit, nyu
from dronerl.perception.depth_model import MODEL_ID, DepthModel, crop_fov, reduce_to_policy
from dronerl.perception.eval_depth import plot_examples, predict

MODELS = {
    "zero_shot": MODEL_ID,
    "ft": "runs/m3/depth_ft/full_nyu_s0_20261027-013844/checkpoints/best",
}
MONO = Path("runs/m3/noise/mono.json")
PROFILE = "ft"  # the model chosen in M4.10, the one depth_node runs
MAX_DEPTH = nyu.MAX_DEPTH  # as on NYU, the far walls of a long corridor stay out


def load(root):
    files = sorted(Path(root).glob("*.npz"))
    data = [np.load(f) for f in files]
    return {
        "rgb": np.concatenate([d["rgb"] for d in data]),
        "depth": np.concatenate([d["depth"] for d in data]),
        "fx": float(data[0]["fx"]),
        "flights": [f.stem for f in files],
        "index": np.concatenate([np.full(len(d["rgb"]), i) for i, d in enumerate(data)]),
    }


def simulate(params, gt, samples, amp, sign, seed=0):
    gen = torch.Generator().manual_seed(seed)
    gt = torch.as_tensor(np.repeat(gt, samples, 0))
    noise = DepthNoise(len(gt), gen, {"mono": 1.0}, mono=params, amp=(amp, amp), signs=(sign,))
    noise.reset(torch.arange(len(gt)))
    return noise(gt).numpy()


def envelope(params, gt, valid, samples):
    """AbsRel per depth range of the training noise at the ends of AMP, both signs of the bias."""
    out = {}
    for amp in AMP:
        for sign in (-1, 1):
            noisy = simulate(params, gt, samples, amp, sign)
            m = noise_fit.summary(noisy, np.repeat(gt, samples, 0), np.repeat(valid, samples, 0))
            for lo, hi in metrics.BINS:
                out.setdefault(f"abs_rel_{lo}_{hi}", []).append(m[f"abs_rel_{lo}_{hi}"])
    return {k: (min(v), max(v)) for k, v in out.items()}


def against_noise(pred, gt, params, samples):
    """The error of a model on reduced images against the mono noise of the training."""
    valid = np.isfinite(gt) & (gt > metrics.MIN_DEPTH) & (gt <= MAX_DEPTH)
    real = noise_fit.summary(pred, gt, valid)
    noisy = simulate(params, gt, samples, 1.0, 1)
    sim = noise_fit.summary(noisy, np.repeat(gt, samples, 0), np.repeat(valid, samples, 0))
    text, ok, ks = noise_fit.check(real, sim)
    band = envelope(params, gt, valid, samples)
    inside = {k: lo <= real[k] <= hi for k, (lo, hi) in band.items()}
    rows = [
        f"| {k.removeprefix('abs_rel_').replace('_', '-')} m | {real[k]:.3f} "
        f"| {lo:.3f}-{hi:.3f} | {'tak' if inside[k] else 'nie'} |"
        for k, (lo, hi) in band.items()
    ]
    bias = float(np.median(real["log_ratio"]))
    lines = [
        text,
        "",
        f"Szum treningu (amplituda {AMP[0]}-{AMP[1]}, oba znaki skali): AbsRel modelu w paśmie?",
        "",
        "| zakres | AbsRel model | AbsRel szum treningu | w paśmie |",
        "|---|---|---|---|",
        *rows,
        "",
        f"Mediana log(predykcja / prawda): {bias:+.3f}; w treningu skala ±{params['bias']:.3f} "
        f"× amplituda.",
    ]
    summary = {"fit_check": ok, "ks": ks, "band": band, "inside": inside, "median_log_ratio": bias}
    return "\n".join(lines), summary


def fit_profile(pred, gt, index, samples, out):
    """Noise profile of the model on reduced Gazebo frames in the format of the mono one: fitted
    on the even flights, checked on the odd ones (both worlds in each half)."""
    valid = np.isfinite(gt) & (gt > metrics.MIN_DEPTH) & (gt <= MAX_DEPTH)
    half = index % 2 == 0
    params = noise_fit.fit(pred[half], gt[half], valid[half])
    out.mkdir(parents=True, exist_ok=True)
    (out / "gazebo.json").write_text(json.dumps(params, indent=2))
    h = ~half
    noisy = noise_fit.simulate(params, gt[h], samples, seed=0)
    real = noise_fit.summary(pred[h], gt[h], valid[h])
    sim = noise_fit.summary(noisy, np.repeat(gt[h], samples, 0), np.repeat(valid[h], samples, 0))
    text, ok, ks = noise_fit.check(real, sim)
    head = f"Dopasowanie: {int(half.sum())} klatek, sprawdzenie: {int(h.sum())} × {samples} próbek."
    return f"{head}\n\n{text}", {"fit_check": ok, "ks": ks}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=Path, default=Path("runs/m4/depth_gz/frames"))
    parser.add_argument("--out", type=Path, default=Path("runs/m4/depth_gz"))
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--noise", type=Path, default=Path("runs/m4/noise"))
    args = parser.parse_args(argv)

    data = load(args.frames)
    fx, params = data["fx"], json.loads(MONO.read_text())
    gt = crop_fov(torch.as_tensor(data["depth"]), fx)
    gt_small = reduce_to_policy(gt, fx).numpy()
    results, report = {"frames": len(gt), "flights": data["flights"]}, []
    for name, path in MODELS.items():
        # the model sees the whole image, the crop follows, as in depth_node (M4.6)
        pred = predict(DepthModel(518, path=path), data["rgb"])
        ids = np.linspace(0, len(pred) - 1, 4).astype(int)
        plot_examples(data["rgb"], data["depth"], pred, args.out / f"examples_{name}.png", ids)
        pred = crop_fov(torch.as_tensor(pred), fx)
        small = reduce_to_policy(pred, fx).numpy()
        r, pairs = {}, {"full": (pred.numpy(), gt.numpy()), "reduced": (small, gt_small)}
        for variant, (a, b) in pairs.items():
            r[variant] = metrics.evaluate(a, b, MAX_DEPTH)
            r[f"{variant}_median"] = metrics.evaluate(a, b, MAX_DEPTH, median_scale=True)
        text, r["noise"] = against_noise(small, gt_small, params, args.samples)
        results[name] = r
        report += [f"## {name}", "", text, ""]
        if name == PROFILE:
            text, results["gazebo_profile"] = fit_profile(
                small, gt_small, data["index"], args.samples, args.noise
            )
            report += [f"## Profil szumu gazebo ({name})", "", text, ""]
        print(f"{name}: AbsRel {r['full']['abs_rel']:.3f}, reduced {r['reduced']['abs_rel']:.3f}")
    cols = ["abs_rel", "rmse", "d1", "silog", "abs_rel_0_2", "abs_rel_2_4", "abs_rel_4_10"]
    table = ["| model | wariant | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for name in MODELS:
        for variant in ("full", "full_median", "reduced", "reduced_median"):
            cells = " | ".join(f"{results[name][variant][c]:.3f}" for c in cols)
            table.append(f"| {name} | {variant} | {cells} |")
    header = f"{len(gt)} klatek z {len(data['flights'])} przelotów.\n\n"
    # numpy scalars of the noise statistics as plain numbers
    (args.out / "metrics.json").write_text(
        json.dumps(results, indent=2, default=lambda x: x.item())
    )
    (args.out / "metrics.md").write_text(header + "\n".join(table) + "\n\n" + "\n".join(report))
    print("\n".join(table))


if __name__ == "__main__":
    main()
