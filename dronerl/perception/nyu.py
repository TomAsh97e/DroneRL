"""NYU Depth v2, labeled subset (Silberman et al. 2012): 1449 RGB-D pairs, official split by scenes.

The .mat file is HDF5 (MATLAB v7.3) with transposed arrays, splits.mat is MATLAB v5. `depths` are
filled in by colorization, `rawDepths` keep the holes of the Kinect (0).
"""

from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat

ROOT = Path("runs/data/nyu")
FILE = "nyu_depth_v2_labeled.mat"
FX = 518.86  # focal length of the RGB camera [px], the depths are registered to it
EIGEN_CROP = (slice(45, 471), slice(41, 601))
MAX_DEPTH = 10.0
VAL_IMAGES = 80


def scene_names(root=ROOT):
    with h5py.File(Path(root) / FILE, "r") as f:
        return ["".join(map(chr, f[ref][:].ravel())) for ref in f["scenes"][0]]


def splits(root=ROOT, seed=0):
    """0-based indices of train, val and test. Val takes whole train scenes, so it shares none."""
    s = loadmat(Path(root) / "splits.mat")
    train, test = s["trainNdxs"].ravel() - 1, s["testNdxs"].ravel() - 1
    scenes = scene_names(root)
    counts = Counter(scenes[i] for i in train)
    names = sorted(counts)
    np.random.default_rng(seed).shuffle(names)
    val_scenes, n = set(), 0
    for name in names:
        if n >= VAL_IMAGES:
            break
        val_scenes.add(name)
        n += counts[name]
    in_val = np.array([scenes[i] in val_scenes for i in train])
    return {"train": train[~in_val], "val": train[in_val], "test": test}


def split_scenes(indices, root=ROOT, seed=0):
    """Halves of `indices` with disjoint scenes, e.g. to fit and check the noise model."""
    scenes = scene_names(root)
    names = sorted({scenes[i] for i in indices})
    np.random.default_rng(seed).shuffle(names)
    first = set(names[: len(names) // 2])
    mask = np.array([scenes[i] in first for i in indices])
    return indices[mask], indices[~mask]


def load(indices, root=ROOT):
    """RGB [N, 480, 640, 3] uint8, filled and raw depth [N, 480, 640] in metres."""
    idx = np.sort(indices)  # h5py reads only increasing indices
    with h5py.File(Path(root) / FILE, "r") as f:
        rgb = f["images"][idx].transpose(0, 3, 2, 1)
        depth = f["depths"][idx].transpose(0, 2, 1)
        raw = f["rawDepths"][idx].transpose(0, 2, 1)
    return {
        "index": idx,
        "rgb": np.ascontiguousarray(rgb),
        "depth": np.ascontiguousarray(depth, dtype=np.float32),
        "raw": np.ascontiguousarray(raw, dtype=np.float32),
    }


def crop(x):
    """Eigen crop of [..., 480, 640] arrays, the border is not covered by the depth sensor."""
    return x[..., EIGEN_CROP[0], EIGEN_CROP[1]]
