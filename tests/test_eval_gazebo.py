import json

import numpy as np
import pytest

from dronerl.perception import eval_gazebo

pytestmark = pytest.mark.skipif(not eval_gazebo.MONO.exists(), reason="no mono noise fit of M3")


def test_against_noise_reports_the_band_of_the_training_noise():
    rng = np.random.default_rng(0)
    gt = rng.uniform(0.5, 9.0, (6, 48, 64)).astype(np.float32)
    pred = gt * np.exp(rng.normal(0.05, 0.1, gt.shape)).astype(np.float32)
    params = json.loads(eval_gazebo.MONO.read_text())
    text, summary = eval_gazebo.against_noise(pred, gt, params, samples=2)
    assert set(summary["band"]) == {"abs_rel_0_2", "abs_rel_2_4", "abs_rel_4_10"}
    for lo, hi in summary["band"].values():
        assert 0 < lo <= hi
    assert abs(summary["median_log_ratio"] - 0.05) < 0.01
    assert "w paśmie" in text


def test_gazebo_profile_is_fitted_on_half_of_the_flights(tmp_path):
    # a floor gradient and a box: random depth would make every pixel an edge
    rng = np.random.default_rng(0)
    gt = np.repeat(np.linspace(0.5, 9.0, 48, dtype=np.float32)[None, :, None], 64, 2)
    gt = np.repeat(gt, 8, 0)
    gt[:, 10:30, 20:40] = rng.uniform(0.6, 3.0, (8, 1, 1))
    pred = gt * np.exp(rng.normal(0.1, 0.1, gt.shape)).astype(np.float32)
    text, summary = eval_gazebo.fit_profile(pred, gt, np.arange(8) // 2, 2, tmp_path)
    params = json.loads((tmp_path / "gazebo.json").read_text())
    assert params["fit"]["images"] == 4
    assert abs(params["bias"] - 0.1) < 0.02
    assert "Dopasowanie: 4 klatek" in text and "fit_check" in summary
