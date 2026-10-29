"""Depth model on the NYU test set: metrics at several input resolutions, full size and reduced to
the policy image, inference latency, examples (M3.3, M3.4)."""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from dronerl.perception import metrics, nyu  # noqa: E402
from dronerl.perception.depth_model import MODEL_ID, DepthModel, reduce_to_policy  # noqa: E402


def predict(model, rgb, batch=8):
    return np.concatenate(
        [model(rgb[i : i + batch]).cpu().numpy() for i in range(0, len(rgb), batch)]
    )


def reduce(x):
    """Cropped [N, 426, 560] depth -> [N, 26, 35] at the angular pixel size of the policy."""
    return reduce_to_policy(torch.as_tensor(x), nyu.FX).numpy()


def evaluate(pred, data):
    """Metrics of predictions [N, 480, 640] against the filled ground truth, as for NYU."""
    gt, p = nyu.crop(data["depth"]), nyu.crop(pred)
    result = {}
    for name, (a, b) in {"full": (p, gt), "reduced": (reduce(p), reduce(gt))}.items():
        result[name] = metrics.evaluate(a, b, nyu.MAX_DEPTH)
        result[f"{name}_median"] = metrics.evaluate(a, b, nyu.MAX_DEPTH, median_scale=True)
    return result


def latency(model, rgb, device, repeats):
    x = rgb[:1]
    for _ in range(3):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(repeats):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t) / repeats


def latencies(path, res, rgb):
    """Milliseconds per image (batch 1, 480 x 640 uint8 in, preprocessing included)."""
    out = {}
    for name, dtype in {"gpu_fp32": torch.float32, "gpu_fp16": torch.float16}.items():
        out[name] = latency(DepthModel(res, "cuda", dtype, path), rgb, "cuda", 50)
    model = DepthModel(res, "cpu", torch.float32, path)
    threads = torch.get_num_threads()
    for name, n in [("cpu_1_thread", 1), (f"cpu_{threads}_threads", threads)]:
        torch.set_num_threads(n)
        out[name] = latency(model, rgb, "cpu", 5)
    torch.set_num_threads(threads)
    return out


def plot_examples(rgb, gt, pred, path, ids=(0, 100, 300, 500)):
    fig, ax = plt.subplots(len(ids), 4, figsize=(14, 2.6 * len(ids)), squeeze=False)
    for row, i in enumerate(ids):
        err = np.abs(pred[i] - gt[i]) / gt[i]
        ax[row, 0].imshow(rgb[i])
        panels = [(gt[i], "magma_r", 8), (pred[i], "magma_r", 8), (err, "viridis", 0.5)]
        for col, (img, cmap, vmax) in enumerate(panels, start=1):
            im = ax[row, col].imshow(img, cmap=cmap, vmin=0, vmax=vmax)
            fig.colorbar(im, ax=ax[row, col], fraction=0.035)
    for col, title in enumerate(["RGB", "głębia [m]", "predykcja [m]", "błąd względny"]):
        ax[0, col].set_title(title)
    for a in ax.flat:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def error_by_depth(curves, path):
    """Pooled AbsRel in 0.5 m bins of the ground truth, one line per resolution."""
    edges = np.arange(0, 10.5, 0.5)
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, (pred, gt) in curves.items():
        p, g = nyu.crop(pred).ravel(), nyu.crop(gt).ravel()
        idx = np.digitize(g, edges) - 1
        rel = np.abs(p - g) / g
        mean = [
            rel[idx == k].mean() if (idx == k).sum() > 1000 else np.nan
            for k in range(len(edges) - 1)
        ]
        ax.plot(edges[:-1] + 0.25, mean, marker="o", label=label)
    ax.set_xlabel("prawdziwa głębia [m]")
    ax.set_ylabel("AbsRel")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def table(results):
    cols = ["abs_rel", "rmse", "d1", "silog", "abs_rel_0_2", "abs_rel_2_4", "abs_rel_4_10"]
    lines = ["| R | wariant | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for res, r in results.items():
        for variant in ["full", "full_median", "reduced", "reduced_median"]:
            cells = [f"{r[variant][c]:.3f}" for c in cols]
            lines.append(f"| {res} | {variant} | " + " | ".join(cells) + " |")
    lat = [r["latency_ms"] for r in results.values() if "latency_ms" in r]
    if lat:
        names = list(lat[0])
        header = "| R | " + " | ".join(f"{n} [ms]" for n in names) + " |"
        lines += ["", header, "|---|" + "---|" * len(names)]
        for res, r in results.items():
            lines.append(
                f"| {res} | " + " | ".join(f"{r['latency_ms'][n]:.1f}" for n in names) + " |"
            )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID, help="HF id or a fine-tuned model directory")
    parser.add_argument("--res", type=int, nargs="+", default=[518, 364, 252, 196])
    parser.add_argument("--out", type=Path, default=Path("runs/m3/depth/zero_shot"))
    parser.add_argument("--no-latency", action="store_true")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    data = nyu.load(nyu.splits()["test"])
    results, curves = {}, {}
    for res in args.res:
        pred = predict(DepthModel(res, path=args.model), data["rgb"])
        results[res] = evaluate(pred, data)
        if not args.no_latency:
            results[res]["latency_ms"] = latencies(args.model, res, data["rgb"])
        curves[f"R = {res}"] = (pred, data["depth"])
        plot_examples(data["rgb"], data["depth"], pred, args.out / f"examples_r{res}.png")
        # inputs of the noise model fit (M3.6): reduced prediction, filled and raw ground truth
        raw = np.where(data["raw"] > 0, data["raw"], np.nan)
        np.savez_compressed(
            args.out / f"reduced_r{res}.npz",
            index=data["index"],
            pred=reduce(nyu.crop(pred)),
            gt=reduce(nyu.crop(data["depth"])),
            raw=reduce(nyu.crop(raw)),
        )
        print(f"R = {res}: AbsRel {results[res]['full']['abs_rel']:.3f}", flush=True)
    error_by_depth(curves, args.out / "error_by_depth.png")
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2))
    text = table(results)
    (args.out / "metrics.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
