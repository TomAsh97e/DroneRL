import numpy as np

from dronerl.eval import sysid


def test_fit_finds_the_parameters_of_the_model():
    true = (0.35, 2.5, 0.08)
    windows = []
    for old, new in ((0.0, 1.0), (1.0, 0.0), (0.0, -0.3), (0.0, 0.5)):
        times = np.arange(-0.2, 2.0, 0.02)
        meas = sysid.respond(times, 0.0, old, old, new, *true)[0]
        windows.append((times, meas, 0.0, old, new))
    grid = np.arange(0.2, 0.5, 0.05), np.arange(1.5, 4.0, 0.5), np.arange(0, 0.15, 0.02)
    f = sysid.fit(windows, *grid)
    np.testing.assert_allclose([f["tau"], f["accel"], f["delay"]], true, atol=1e-6)
    assert f["rms"] < 1e-9


def test_accel_limit_bounds_the_slope():
    times = np.arange(0.0, 1.0, 0.005)
    v = sysid.respond(times, 0.0, 0.0, 0.0, 1.0, 0.01, 2.0, 0.0)[0]
    assert np.diff(v).max() <= 2.0 * sysid.DT + 1e-12
    np.testing.assert_allclose(v[-1], 1.0)


def test_euler_and_motion_in_the_heading_frame():
    # flying north (ENU +y) with the nose north: all of it is forward
    t = np.arange(0.0, 1.0, 0.02)
    yaw = np.pi / 2
    q = np.tile([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)], (len(t), 1))
    gt = np.column_stack([t, np.zeros_like(t), 0.5 * t, np.ones_like(t), q])
    _, vel, _, rate, roll, pitch = sysid.motion(gt)
    np.testing.assert_allclose(vel[:, :, 0], np.tile([0.5, 0.0], (len(t), 1)), atol=1e-9)
    np.testing.assert_allclose(rate, 0, atol=1e-9)
    np.testing.assert_allclose([roll, pitch], 0, atol=1e-9)
