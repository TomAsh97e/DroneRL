import math
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from scipy.io import savemat

from dronerl import contract
from dronerl.perception import metrics, nyu
from dronerl.perception.depth_model import (
    DepthModel,
    OnnxDepth,
    crop_fov,
    policy_size,
    reduce_depth,
    student_image,
)

M3_REDUCED = Path("runs/m3/depth/zero_shot/reduced_r518.npz")
M3_ONNX = Path("runs/m3/onnx/depth_zero_shot_r518.onnx")


def fake_nyu(root, scenes, h=6, w=8):
    """A file in the layout of the labeled .mat: transposed arrays, scene names as references."""
    n = len(scenes)
    rgb = np.arange(n * h * w * 3, dtype=np.uint8).reshape(n, h, w, 3)
    depth = np.random.default_rng(0).uniform(0.5, 9, (n, h, w)).astype(np.float32)
    with h5py.File(root / nyu.FILE, "w") as f:
        f["images"] = rgb.transpose(0, 3, 2, 1)
        f["depths"] = depth.transpose(0, 2, 1)
        f["rawDepths"] = np.where(depth > 5, 0, depth).transpose(0, 2, 1)
        refs = [
            f.create_dataset(f"names/{i}", data=np.array([ord(c) for c in s], np.uint16)).ref
            for i, s in enumerate(scenes)
        ]
        f.create_dataset("scenes", data=np.array([refs], dtype=h5py.ref_dtype))
    return rgb, depth


def test_nyu_layout_and_splits(tmp_path, monkeypatch):
    scenes = ["a", "a", "b", "c", "c", "c", "d", "e"]
    rgb, depth = fake_nyu(tmp_path, scenes)
    savemat(
        tmp_path / "splits.mat",
        {"trainNdxs": [[1], [2], [3], [4], [5], [6]], "testNdxs": [[7], [8]]},
    )
    assert nyu.scene_names(tmp_path) == scenes
    data = nyu.load(np.array([4, 1]), tmp_path)
    assert data["index"].tolist() == [1, 4]
    np.testing.assert_array_equal(data["rgb"], rgb[[1, 4]])
    np.testing.assert_array_equal(data["depth"], depth[[1, 4]])
    assert (data["raw"][data["depth"] > 5] == 0).all()

    monkeypatch.setattr(nyu, "VAL_IMAGES", 2)
    s = nyu.splits(tmp_path)
    assert sorted([*s["train"], *s["val"]]) == list(range(6)) and s["test"].tolist() == [6, 7]
    # val takes whole scenes
    train_scenes = {scenes[i] for i in s["train"]}
    assert len(s["val"]) >= 2 and not train_scenes & {scenes[i] for i in s["val"]}
    a, b = nyu.split_scenes(np.arange(8), tmp_path)
    assert not {scenes[i] for i in a} & {scenes[i] for i in b} and len(a) + len(b) == 8


def test_eigen_crop():
    x = np.zeros((2, 480, 640))
    assert nyu.crop(x).shape == (2, 426, 560)


def test_metrics_scale_and_median():
    rng = np.random.default_rng(0)
    gts = [rng.uniform(0.5, 9, (20, 30)) for _ in range(3)]
    exact = metrics.evaluate(gts, gts)
    assert exact["abs_rel"] == pytest.approx(0) and exact["d1"] == 1
    assert exact["silog"] == pytest.approx(0)
    scaled = metrics.evaluate([1.1 * g for g in gts], gts)
    assert scaled["abs_rel"] == pytest.approx(0.1, rel=1e-6)
    assert scaled["silog"] == pytest.approx(0, abs=1e-4)  # scale invariant
    assert metrics.evaluate([1.1 * g for g in gts], gts, median_scale=True)["abs_rel"] < 1e-6
    # invalid ground truth is skipped
    holes = [np.where(g > 5, np.nan, g) for g in gts]
    assert metrics.evaluate(gts, holes)["abs_rel"] == pytest.approx(0)
    assert {"abs_rel_0_2", "d1_2_4", "abs_rel_4_10"} <= exact.keys()


def test_reduce_depth():
    d = torch.full((2, 48, 64), 3.0)
    torch.testing.assert_close(reduce_depth(d, (12, 16)), torch.full((2, 12, 16), 3.0))
    # mean of inverse depth over valid pixels only, inf counts as far
    d = torch.tensor([[1.0, 4.0, math.nan, math.inf]])
    out = reduce_depth(d[None], (1, 1))
    assert out.item() == pytest.approx(1 / ((1 + 0.25 + 0) / 3))
    assert reduce_depth(torch.full((1, 4, 4), math.nan), (2, 2)).isnan().all()
    assert reduce_depth(torch.full((1, 4, 4), math.inf), (2, 2)).isinf().all()


def test_policy_size_matches_focal():
    assert policy_size(contract.IMG_H, contract.IMG_W, contract.FOCAL) == (48, 64)
    assert policy_size(426, 560, nyu.FX) == (26, 35)


def test_crop_fov_of_the_gazebo_camera():
    fx = 320 / math.tan(math.radians(50))  # 640 x 480 with HFOV 100°
    x = torch.arange(640.0).expand(3, 480, 640)
    crop = crop_fov(x, fx)
    assert crop.shape == (3, 404, 538)
    assert crop[0, 0].mean() == pytest.approx(319.5)  # centred on the image
    assert crop.shape[-1] / 2 / fx == pytest.approx(math.tan(contract.HFOV / 2), rel=2e-3)
    assert policy_size(*crop.shape[-2:], fx) == (contract.IMG_H, contract.IMG_W)
    assert crop_fov(x, nyu.FX).shape == x.shape  # a narrower camera keeps the whole image


def test_student_image_runs_the_model_on_the_whole_image():
    fx = 320 / math.tan(math.radians(50))
    seen = []

    def depth(rgb):
        seen.append(rgb.shape)
        return torch.full(rgb.shape[:3], 2.0)

    image = student_image(depth, np.zeros((480, 640, 3), np.uint8), fx)
    assert seen == [(1, 480, 640, 3)]
    torch.testing.assert_close(
        image, contract.encode_student_depth(torch.full((contract.IMG_H, contract.IMG_W), 2.0))
    )


@pytest.mark.skipif(
    not (M3_REDUCED.exists() and M3_ONNX.exists() and (nyu.ROOT / nyu.FILE).exists()),
    reason="no NYU or M3 results",
)
@pytest.mark.parametrize("backend", ["onnx", "torch"])
def test_student_image_matches_m3_on_nyu(backend):
    with np.load(M3_REDUCED) as f:
        index, pred = f["index"][:4], f["pred"][:4]
    data = nyu.load(index)
    depth = OnnxDepth(M3_ONNX, device="cpu") if backend == "onnx" else DepthModel(518, "cpu")
    for rgb, ref in zip(data["rgb"], pred):
        image = student_image(depth, rgb, nyu.FX, nyu.crop)
        expected = contract.encode_student_depth(torch.as_tensor(ref))
        torch.testing.assert_close(image, expected, atol=1e-4, rtol=0)


def test_silog_and_augment():
    from dronerl.perception import finetune

    gt = torch.rand(2, 480, 640) * 8 + 0.5
    assert finetune.silog(gt, gt, 0.5).item() == pytest.approx(0, abs=1e-6)
    # with lambda = 1 the loss ignores a global scale
    assert finetune.silog(1.3 * gt, gt, 1.0).item() == pytest.approx(0, abs=1e-3)
    assert finetune.silog(1.3 * gt, gt, 0.5).item() > 0.1

    rgb = torch.randint(0, 256, (2, 480, 640, 3), dtype=torch.uint8)
    depth = torch.arange(480 * 640, dtype=torch.float32).reshape(1, 480, 640).expand(2, -1, -1)
    x, y = finetune.augment(rgb, depth, torch.Generator().manual_seed(0), 0.2)
    assert x.shape == (2, *finetune.CROP, 3) and x.dtype == torch.uint8
    assert y.shape == (2, *finetune.CROP)
    # the crop is a contiguous window, possibly mirrored
    rows = y[0, :, 0] if y[0, 0, 1] > y[0, 0, 0] else y[0, :, -1]
    assert torch.all(rows.diff() == 640)
