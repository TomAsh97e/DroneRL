"""Monocular depth metrics (Eigen et al. 2014): per-image metrics averaged over images, as usual for
NYU, plus AbsRel and δ1 per depth range pooled over pixels."""

import numpy as np

BINS = ((0, 2), (2, 4), (4, 10))
MIN_DEPTH = 1e-3


def valid(gt, pred, max_depth):
    return np.isfinite(gt) & np.isfinite(pred) & (gt > MIN_DEPTH) & (gt <= max_depth)


def image_metrics(pred, gt):
    """Metrics over 1D arrays of matching pixels."""
    ratio = np.maximum(pred / gt, gt / pred)
    err = np.log(pred) - np.log(gt)
    return {
        "abs_rel": np.mean(np.abs(pred - gt) / gt),
        "sq_rel": np.mean((pred - gt) ** 2 / gt),
        "rmse": np.sqrt(np.mean((pred - gt) ** 2)),
        "rmse_log": np.sqrt(np.mean(err**2)),
        "log10": np.mean(np.abs(np.log10(pred) - np.log10(gt))),
        "d1": np.mean(ratio < 1.25),
        "d2": np.mean(ratio < 1.25**2),
        "d3": np.mean(ratio < 1.25**3),
        "silog": 100 * np.sqrt(max(np.mean(err**2) - np.mean(err) ** 2, 0)),
    }


def evaluate(preds, gts, max_depth=10.0, median_scale=False):
    """preds and gts: sequences of metric depth maps of equal shapes, NaN or 0 in gt is invalid.

    With `median_scale` each prediction is scaled by median(gt) / median(pred) first, which
    removes the global scale error and leaves the error of the shape.
    """
    per_image, pooled = [], []
    for pred, gt in zip(preds, gts):
        pred = np.clip(pred, MIN_DEPTH, max_depth)
        m = valid(gt, pred, max_depth)
        p, g = pred[m], gt[m]
        if median_scale:
            p = np.clip(p * np.median(g) / np.median(p), MIN_DEPTH, max_depth)
        per_image.append(image_metrics(p, g))
        pooled.append((p, g))
    result = {k: float(np.mean([m[k] for m in per_image])) for k in per_image[0]}
    p = np.concatenate([x[0] for x in pooled])
    g = np.concatenate([x[1] for x in pooled])
    for lo, hi in BINS:
        m = (g > lo) & (g <= hi)
        if m.any():
            result[f"abs_rel_{lo}_{hi}"] = float(np.mean(np.abs(p[m] - g[m]) / g[m]))
            result[f"d1_{lo}_{hi}"] = float(np.mean(np.maximum(p[m] / g[m], g[m] / p[m]) < 1.25))
    return result
