"""Fit and check of the mono depth noise (M3.6) on the errors of the zero-shot model at R = 518.

Input: `reduced_r518.npz` from eval_depth.py, prediction and ground truth reduced to the angular
pixel size of the policy. The NYU test set is split by scenes into a fit and a check half. The
filled depth is the truth the noise is applied to, the statistics use only cells where the raw
Kinect depth is valid.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import stats  # noqa: E402

from dronerl.envs.depth_noise import (  # noqa: E402
    DepthNoise,
    blur_inverse,
    edge_distance,
    gaussian_kernel,
    heavy,
)
from dronerl.perception import metrics, nyu  # noqa: E402

# the mean error falls steeply up to 2 m, coarse bins would make its distribution multimodal
BINS = [0.0, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]
SIGMAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
CORR_SIGMAS = np.arange(0.5, 4.01, 0.25)
TAILS = np.linspace(0.0, 0.3, 61)
SCALE_P = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]


def lag1_corr(sigma):
    """Correlation of neighbouring pixels of white noise blurred with the discrete kernel."""
    if sigma <= 0:
        return 0.0
    k = gaussian_kernel(sigma, "cpu").double()
    return float((k[1:] * k[:-1]).sum() / k.square().sum())


def sigma_for(rho):
    grid = np.linspace(0.05, 4, 400)
    return float(grid[np.argmin([abs(lag1_corr(s) - rho) for s in grid])]) if rho > 0 else 0.0


def neighbour_corr(r):
    """Mean correlation of horizontal and vertical neighbours of standardized residuals [N, H, W],
    NaN outside valid cells."""
    out = []
    for a, b in [(r[:, :, 1:], r[:, :, :-1]), (r[:, 1:], r[:, :-1])]:
        m = np.isfinite(a) & np.isfinite(b)
        out.append(np.corrcoef(a[m], b[m])[0, 1])
    return float(np.mean(out))


def scales(log_ratio):
    return np.array([np.nanmedian(x) for x in log_ratio])


def bin_index(z):
    return np.clip(np.searchsorted(BINS, z, side="right") - 1, 0, len(BINS) - 1)


def central_share(h):
    """Share of |x| < 1 for the unit Gaussian after the Tukey h transform."""
    f = torch.linspace(-8, 8, 16001, dtype=torch.float64)
    w = torch.exp(-(f**2) / 2)
    return float((w * (heavy(f, h).abs() < 1)).sum() / w.sum())


def residual_corr(pred, gt, valid):
    """Neighbour correlation of the log error after the scale of each image."""
    log_ratio = np.where(valid, np.log(pred / gt), np.nan)
    return neighbour_corr(log_ratio - scales(log_ratio)[:, None, None])


def decompose(log_ratio, b):
    """Mean log error per depth bin over all pixels, then the scale of each image around it. In
    NYU near scenes get a larger scale, the noise draws the scale independently of the scene, so
    only means over all pixels keep the error per bin unbiased."""
    m = np.array([np.nanmean(log_ratio[b == i]) for i in range(len(BINS))])
    return scales(log_ratio - m[b]), m


def fit(pred, gt, valid, seed=1):
    b = bin_index(gt)
    dist = edge_distance(torch.as_tensor(gt)).numpy()
    # edge blur: the one that best explains the prediction after scale and mean per bin
    parts, spread = {}, {}
    for sigma in SIGMAS:
        blurred = blur_inverse(torch.as_tensor(gt), sigma).numpy()
        log_ratio = np.where(valid, np.log(pred / blurred), np.nan)
        s, m = decompose(log_ratio, b)
        parts[sigma] = s, m, log_ratio - s[:, None, None] - m[b]
        spread[sigma] = float(np.nanstd(parts[sigma][2]))
    edge_sigma = min(spread, key=spread.get)
    s, m, r = parts[edge_sigma]
    bias = float(np.nanmean(np.where(valid, m[b], np.nan)))

    def per_bin(x):
        return np.array([np.nanstd(x[b == i]) for i in range(len(BINS))])[b]

    # level of the residual per image, the errors are far from Gaussian without it
    level = np.log([np.nanstd(x) for x in r / per_bin(r)])
    level -= level.mean()
    r = r / np.exp(level)[:, None, None]
    z = r / per_bin(r)
    gain = np.array([np.nanstd(z[dist == d]) for d in range(4)])
    gain /= gain[-1]
    r = r / gain[dist]
    std = [float(np.nanstd(r[b == i])) for i in range(len(BINS))]
    # tails from the share of small errors, the part δ1 sees, rather than from the kurtosis
    small = np.nanmean(np.where(np.isfinite(r), np.abs(r / per_bin(r)) < 1, np.nan))
    tail = min(TAILS, key=lambda h: abs(central_share(h) - small))
    params = {
        "bias": bias,
        "scale_p": SCALE_P,
        "scale_q": np.quantile(s - s.mean(), SCALE_P).tolist(),
        "edge_sigma": edge_sigma,
        "edge_gain": gain.tolist(),
        "residual": [list(row) for row in zip(BINS, (m - bias).tolist(), std)],
        "res_tail": float(tail),
        "res_gain_std": float(level.std()),
    }
    # the correlation seen by the check also includes the mean per bin and the edges, so the
    # blur of the field is matched by simulation rather than taken from the residual alone
    target = residual_corr(pred, gt, valid)
    got = {}
    for c in CORR_SIGMAS:
        noisy = simulate({**params, "corr_sigma": float(c)}, gt, 1, seed)
        got[float(c)] = residual_corr(noisy, gt, valid)
    params["corr_sigma"] = min(got, key=lambda c: abs(got[c] - target))
    params["fit"] = {
        "edge_spread": spread,
        "neighbour_corr": target,
        "simulated_corr": got,
        "images": len(pred),
    }
    return params


def simulate(params, gt, samples, seed):
    gen = torch.Generator().manual_seed(seed)
    gt = torch.as_tensor(np.repeat(gt, samples, 0))
    noise = DepthNoise(len(gt), gen, {"mono": 1.0}, mono=params, amp=(1.0, 1.0), signs=(1,))
    noise.reset(torch.arange(len(gt)))
    return noise(gt).numpy()


def summary(pred, gt, valid):
    """Statistics compared by the check: metrics per range, neighbour correlation, edge profile."""
    raw = np.where(valid, gt, np.nan)
    m = metrics.evaluate(list(pred), list(raw), nyu.MAX_DEPTH)
    log_ratio = np.where(valid, np.log(pred / gt), np.nan)
    r = log_ratio - scales(log_ratio)[:, None, None]
    dist = edge_distance(torch.as_tensor(gt)).numpy()
    profile = [float(np.nanstd(r[dist == d])) for d in range(4)]
    return {
        **{k: m[k] for k in m if k.startswith(("abs_rel", "d1"))},
        "neighbour_corr": residual_corr(pred, gt, valid),
        "edge_profile": profile,
        "log_ratio": log_ratio[valid],
    }


def check(real, sim):
    """Thresholds of plan/M3.md, M3.6."""
    rows, ok = [], True
    for lo, hi in metrics.BINS:
        a, b = real[f"abs_rel_{lo}_{hi}"], sim[f"abs_rel_{lo}_{hi}"]
        c, d = real[f"d1_{lo}_{hi}"], sim[f"d1_{lo}_{hi}"]
        good = abs(b - a) <= 0.25 * a and abs(d - c) <= 0.05
        rows.append(
            f"| {lo}-{hi} m | {a:.3f} | {b:.3f} | {c:.3f} | {d:.3f} | {'tak' if good else 'nie'} |"
        )
        ok &= good
    la, lb = sigma_for(real["neighbour_corr"]), sigma_for(sim["neighbour_corr"])
    good = abs(lb - la) <= 0.3 * la
    ok &= good
    ks = stats.ks_2samp(real["log_ratio"], sim["log_ratio"]).statistic
    corr = f"model {real['neighbour_corr']:.3f}, szum {sim['neighbour_corr']:.3f}"
    yes = {True: "tak", False: "nie"}
    lines = [
        "| zakres | AbsRel model | AbsRel szum | δ1 model | δ1 szum | w progu |",
        "|---|---|---|---|---|---|",
        *rows,
        "",
        f"Korelacja sąsiadów: {corr}; σ rozmycia {la:.2f} wobec {lb:.2f} px, w progu: {yes[good]}.",
        f"Statystyka KS log(predykcja / prawda): {ks:.3f}.",
        "Odchylenie residuum w odległości 0, 1, 2, ≥3 px od krawędzi: model "
        + ", ".join(f"{x:.3f}" for x in real["edge_profile"])
        + "; szum "
        + ", ".join(f"{x:.3f}" for x in sim["edge_profile"]),
        "",
        f"Wszystkie progi spełnione: {yes[ok]}.",
    ]
    return "\n".join(lines), ok, ks


def plot(real, sim, pred, gt, noisy, out):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.linspace(-1, 1, 81)
    ax[0].hist(real["log_ratio"], bins, density=True, alpha=0.6, label="model")
    ax[0].hist(sim["log_ratio"], bins, density=True, alpha=0.6, label="szum")
    ax[0].set_xlabel("log(predykcja / prawda)")
    ax[0].legend()
    x = np.arange(len(metrics.BINS))
    keys = [f"abs_rel_{lo}_{hi}" for lo, hi in metrics.BINS]
    ax[1].bar(x - 0.2, [real[k] for k in keys], 0.4, label="model")
    ax[1].bar(x + 0.2, [sim[k] for k in keys], 0.4, label="szum")
    ax[1].set_xticks(x, [f"{lo}-{hi} m" for lo, hi in metrics.BINS])
    ax[1].set_ylabel("AbsRel")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / "check.png", dpi=100)
    plt.close(fig)

    fig, ax = plt.subplots(4, 3, figsize=(9, 9))
    for row, i in enumerate(np.linspace(0, len(gt) - 1, 4).astype(int)):
        for col, img in enumerate([gt[i], pred[i], noisy[i]]):
            ax[row, col].imshow(img, cmap="magma_r", vmin=0, vmax=8)
            ax[row, col].axis("off")
    for col, title in enumerate(["prawda", "model", "szum na prawdzie"]):
        ax[0, col].set_title(title)
    fig.tight_layout()
    fig.savefig(out / "examples.png", dpi=90)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("runs/m3/depth/zero_shot/reduced_r518.npz")
    )
    parser.add_argument("--out", type=Path, default=Path("runs/m3/noise"))
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    data = np.load(args.data)
    raw = data["raw"]
    valid = np.isfinite(raw) & (raw > metrics.MIN_DEPTH) & (raw <= nyu.MAX_DEPTH)
    first, second = nyu.split_scenes(data["index"])
    halves = {k: np.isin(data["index"], idx) for k, idx in {"fit": first, "check": second}.items()}
    pred, gt = data["pred"], data["gt"]

    h = halves["fit"]
    params = fit(pred[h], gt[h], valid[h])
    (args.out / "mono.json").write_text(json.dumps(params, indent=2))

    h = halves["check"]
    noisy = simulate(params, gt[h], args.samples, seed=0)
    real = summary(pred[h], gt[h], valid[h])
    sim = summary(noisy, np.repeat(gt[h], args.samples, 0), np.repeat(valid[h], args.samples, 0))
    text, ok, ks = check(real, sim)
    plot(real, sim, pred[h], gt[h], noisy[:: args.samples], args.out)
    n_fit, n_check = int(halves["fit"].sum()), int(h.sum())
    header = f"Dopasowanie: {n_fit} obrazów, sprawdzenie: {n_check} × {args.samples} próbek.\n\n"
    (args.out / "check.md").write_text(header + text + "\n")
    print(json.dumps({k: v for k, v in params.items() if k != "fit"}, indent=2))
    print(header + text)


if __name__ == "__main__":
    main()
