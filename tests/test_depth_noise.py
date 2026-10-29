import math

import pytest
import torch

from dronerl import contract
from dronerl.envs import depth_noise as dn

QUIET = {
    **dn.ACTIVE,
    "k": (0.0, 0.0),
    "holes": (0.0, 0.0),
    "min_range": (0.0, 0.0),
    "max_range": (math.inf, math.inf),
    "flying": (0.0, 0.0),
    "border": 0,
    "scale": 0.0,
}


def scene(n, seed=0):
    """Depth images with edges, a floor gradient and far pixels."""
    g = torch.Generator().manual_seed(seed)
    d = torch.linspace(1, 6, 48)[:, None].expand(n, 48, 64).clone()
    d[:, 10:30, 20:40] = 0.5 + 2 * torch.rand(n, 1, 1, generator=g)
    d[:, :5] = math.inf
    return d


def noise(n, mix, seed=0, **kw):
    out = dn.DepthNoise(n, torch.Generator().manual_seed(seed), mix, **kw)
    out.reset(torch.arange(n))
    return out


def test_none_and_zero_amplitude_are_identity():
    d = scene(4)
    torch.testing.assert_close(noise(4, {"none": 1.0})(d), d)
    torch.testing.assert_close(noise(4, {"mono": 1.0}, amp=(0.0, 0.0))(d), d)
    torch.testing.assert_close(noise(4, {"active": 1.0}, active=QUIET)(d), d)


def test_mono_statistics_match_parameters():
    m = dn.MONO
    n, z = 3000, 2.2  # a flat wall: no edges, one bin of the residual table
    d = torch.full((n, 48, 64), z)
    log_err = torch.log(noise(n, {"mono": 1.0}, amp=(1.0, 1.0), signs=(1,))(d) / d)
    _, mean, std = next(row for row in reversed(m["residual"]) if row[0] <= z)
    per_image = log_err.mean((1, 2)) - m["bias"] - mean
    p = torch.tensor(m["scale_p"])
    q = torch.quantile(per_image, p[1:-1])
    torch.testing.assert_close(q, torch.tensor(m["scale_q"])[1:-1], atol=0.02, rtol=0)
    # the level of the residual varies between frames, its mean square is exp(2σ²)
    res = log_err - log_err.mean((1, 2), keepdim=True)
    expected = std * math.exp(m["res_gain_std"] ** 2)
    assert res.std().item() == pytest.approx(expected, rel=0.05)


def test_gazebo_profile_uses_its_own_fit():
    n, z = 500, 2.2
    d = torch.full((n, 48, 64), z)
    kw = {"amp": (1.0, 1.0), "signs": (1,), "gazebo": {**dn.MONO, "bias": 0.5}}
    mono = torch.log(noise(n, {"mono": 1.0}, **kw)(d) / d).mean()
    gazebo = torch.log(noise(n, {"gazebo": 1.0}, **kw)(d) / d).mean()
    assert (gazebo - mono).item() == pytest.approx(0.5 - dn.MONO["bias"], abs=0.02)


def test_field_is_unit_and_correlated():
    f = dn.field((200, 48, 64), 1.75, torch.Generator().manual_seed(0))
    assert f.std().item() == pytest.approx(1, abs=0.02)
    # the border is as noisy as the centre
    assert f[:, :, 0].std().item() == pytest.approx(1, abs=0.05)
    lag1 = (f[:, :, 1:] * f[:, :, :-1]).mean().item()
    assert lag1 == pytest.approx(math.exp(-1 / (4 * 1.75**2)), abs=0.03)


def test_active_hole_share():
    active = {**QUIET, "holes": (0.05, 0.05)}
    n, d = 500, torch.full((500, 48, 64), 3.0)
    gen = noise(n, {"active": 1.0}, active=active)
    for _ in range(3):  # also after AR steps
        assert gen(d).isnan().float().mean().item() == pytest.approx(0.05, abs=0.01)
    # minimum and maximum range, the border columns
    near = noise(n, {"active": 1.0}, active={**QUIET, "min_range": (1.0, 1.0)})
    assert near(torch.full((n, 4, 4), 0.5)).isnan().all()
    side = noise(n, {"active": 1.0}, active={**QUIET, "border": 4})(d).isnan()
    assert side[:, :, 4:-4].sum() == 0 and side[:, :, 0].any()


def test_mask_is_zero_without_data():
    d = scene(64)
    out = noise(64, {"active": 1.0}, active={**dn.ACTIVE, "holes": (0.1, 0.1)})(d)
    img = contract.encode_student_depth(out)
    missing = out.isnan()
    assert missing.any()
    assert (img[:, 0][missing] == 0).all() and (img[:, 1][missing] == 0).all()
    assert (img[:, 1][~missing] == 1).all()


def test_deterministic():
    mix = {"none": 0.2, "mono": 0.5, "active": 0.3}
    a, b = noise(32, mix, seed=3), noise(32, mix, seed=3)
    for t in range(4):
        d = scene(32, seed=t)
        torch.testing.assert_close(a(d), b(d), equal_nan=True)
        a.reset(torch.tensor([t, 10 + t]))
        b.reset(torch.tensor([t, 10 + t]))
    assert not torch.equal(noise(32, mix, seed=4)(d).nan_to_num(), a(d).nan_to_num())
