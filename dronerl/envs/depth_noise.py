"""Noise of the student depth image (M3.6): what a real depth sensor gets wrong.

Profiles, drawn per episode:
- `mono`: Depth Anything V2 Small at R = 518, fitted on NYU by `perception/noise_fit.py`,
- `gazebo`: the same model fine-tuned, fitted on Gazebo images by `perception/eval_gazebo.py`,
- `active`: a depth or stereo camera, parametric from the literature (no data to fit yet),
- `none`: perfect depth.

Works on metric depth [E, H, W] at the angular pixel size of the policy, NaN marks missing data.
The noise of a frame is an AR(1) process within the episode, a network or sensor errs similarly
on consecutive frames of the same scene.
"""

import math

import torch
import torch.nn.functional as F

PROFILES = ("none", "mono", "active", "gazebo")

# fitted by noise_fit.py on the fit half of the NYU test set (runs/m3/noise/mono.json)
MONO = {
    "bias": 0.166,  # mean log error; its sign depends on the camera, drawn ± in training
    # log scale of a frame around the bias, as quantiles: in NYU it is skewed towards too far
    "scale_p": [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
    "scale_q": [-0.259, -0.181, -0.142, -0.083, -0.008, 0.07, 0.16, 0.232, 0.332],
    "edge_sigma": 0.5,  # Gaussian blur of inverse depth [px]
    "edge_gain": [1.641, 1.317, 1.152, 1],  # residual std factor at 0, 1, 2, ≥3 px from an edge
    # lower edge [m] of a depth bin, mean and std of the residual log error in it
    "residual": [
        [0.0, 0.269, 0.128],
        [0.75, 0.185, 0.101],
        [1, 0.144, 0.1],
        [1.25, 0.087, 0.08],
        [1.5, 0.043, 0.069],
        [1.75, 0.003, 0.062],
        [2, -0.024, 0.062],
        [2.5, -0.036, 0.062],
        [3, -0.037, 0.066],
        [3.5, -0.029, 0.067],
        [4, -0.034, 0.078],
        [5, -0.035, 0.094],
        [6, -0.041, 0.121],
        [8, -0.141, 0.134],
    ],
    "res_tail": 0.095,  # Tukey h of the residual, the errors are peaked with heavy tails
    "res_gain_std": 0.455,  # log spread of the residual level, some scenes are harder
    "corr_sigma": 1.75,  # spatial correlation of the residual: blur of white noise [px]
}
# Nguyen et al. 2012 (Kinect) and Keselman et al. 2017 (RealSense): error grows with z², holes
# on dark, shiny and distant surfaces, a minimum and maximum range
ACTIVE = {
    "k": (0.001, 0.01),  # relative std k·z [1/m]
    "corr_sigma": 0.75,
    "holes": (0.0, 0.1),  # share of pixels without data
    "hole_sigma": 2.0,
    "min_range": (0.2, 0.4),
    "max_range": (4.0, 8.0),
    "flying": (0.0, 0.3),  # chance of a mixed depth on an edge
    "border": 4,  # up to so many columns without data on each side, a FOV down to ~80°
    "scale": 0.02,  # scale calibration error
}
# fitted by perception/eval_gazebo.py on the frames of M4.10 (runs/m4/noise/gazebo.json): the
# fine-tuned model on Gazebo images: a small bias, a wider spread of the frame scale than mono and
# a stronger correlation of the residual (M4.12)
GAZEBO = {
    "bias": 0.029,
    "scale_p": [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
    "scale_q": [-0.532, -0.285, -0.213, -0.111, 0.012, 0.108, 0.209, 0.346, 0.487],
    "edge_sigma": 0.5,
    "edge_gain": [2.035, 1.16, 1.082, 1.0],
    "residual": [
        [0.0, 0.066, 0.083],
        [0.75, -0.066, 0.085],
        [1.0, -0.062, 0.104],
        [1.25, -0.027, 0.092],
        [1.5, 0.046, 0.084],
        [1.75, 0.017, 0.077],
        [2.0, 0.021, 0.069],
        [2.5, 0.023, 0.067],
        [3.0, 0.028, 0.069],
        [3.5, 0.014, 0.073],
        [4.0, 0.006, 0.082],
        [5.0, -0.005, 0.1],
        [6.0, -0.061, 0.122],
        [8.0, -0.173, 0.149],
    ],
    "res_tail": 0.09,
    "res_gain_std": 0.443,
    "corr_sigma": 2.25,
}
AMP = (0.5, 1.5)  # amplitude of the fitted mono noise, covers the change of domain
RHO = (0.5, 0.95)  # frame to frame correlation within an episode
EDGE = 0.2  # log depth step between neighbours that makes an edge


def gaussian_kernel(sigma, device):
    r = max(1, math.ceil(3 * sigma))
    x = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def blur(x, sigma):
    """Separable Gaussian blur of [E, H, W] with replicated borders; sigma 0 returns x."""
    if sigma <= 0:
        return x
    k = gaussian_kernel(sigma, x.device)
    r = len(k) // 2
    y = F.conv2d(F.pad(x[:, None], (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(y, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))[:, 0]


def blur_inverse(depth, sigma):
    """Blur in inverse depth, as a network smooths depth edges; inf stays far."""
    return 1 / blur(1 / depth, sigma)


def field(shape, sigma, gen):
    """Gaussian field of unit variance, its correlation at lag d is about exp(-d² / 4σ²)."""
    if sigma <= 0:
        return torch.randn(shape, generator=gen, device=gen.device)
    k = gaussian_kernel(sigma, gen.device)
    r = len(k) // 2
    e, h, w = shape
    # drawn with a margin and cropped by the valid convolution, so the borders are stationary too
    x = torch.randn((e, 1, h + 2 * r, w + 2 * r), generator=gen, device=gen.device)
    x = F.conv2d(F.conv2d(x, k.view(1, 1, 1, -1)), k.view(1, 1, -1, 1))
    # variance of white noise under a separable kernel k ⊗ k is (Σ k²)²
    return x[:, 0] / k.square().sum()


def interp(x, xp, fp):
    """Piecewise linear interpolation like np.interp, constant outside xp."""
    i = torch.searchsorted(xp, x.contiguous()).clamp(1, len(xp) - 1)
    w = ((x - xp[i - 1]) / (xp[i] - xp[i - 1])).clamp(0, 1)
    return fp[i - 1] + w * (fp[i] - fp[i - 1])


def heavy(x, h):
    """Tukey h transform of a unit Gaussian, heavier tails for h > 0 and still unit variance."""
    return x * torch.exp(h * x**2 / 2) * (1 - 2 * h) ** 0.75


def edges(depth):
    """Pixels with a log depth step above EDGE to a 4-neighbour, inf (no hit) counts as far."""
    log = torch.log(depth.clamp(max=1e6))
    step = torch.zeros_like(log, dtype=torch.bool)
    dx = (log[:, :, 1:] - log[:, :, :-1]).abs() > EDGE
    dy = (log[:, 1:] - log[:, :-1]).abs() > EDGE
    step[:, :, 1:] |= dx
    step[:, :, :-1] |= dx
    step[:, 1:] |= dy
    step[:, :-1] |= dy
    return step


def edge_distance(depth, cap=3):
    """Chessboard distance [px] of each pixel to the nearest edge, capped at `cap`."""
    m = edges(depth).float()[:, None]
    count = torch.zeros_like(m)
    for _ in range(cap):
        count += m
        m = F.max_pool2d(m, 3, 1, 1)
    return cap - count[:, 0].long()


class DepthNoise:
    """Per-environment noise state; `reset(ids)` at episode starts, then one call per frame.

    `mix` maps profiles to probabilities. `signs` of the scale bias of the fitted profiles: (-1, 1)
    in training, where the camera is unknown, (1,) to reproduce the data in the check of a fit.
    """

    def __init__(
        self, n, gen, mix, mono=MONO, active=ACTIVE, amp=AMP, signs=(-1, 1), gazebo=GAZEBO
    ):
        self.n, self.gen, self.dev = n, gen, gen.device
        self.fitted, self.active, self.amp = {"mono": mono, "gazebo": gazebo}, active, amp
        p = torch.tensor([mix.get(name, 0.0) for name in PROFILES], device=self.dev)
        self.cum = (p / p.sum()).cumsum(0)
        self.signs = torch.tensor(signs, dtype=torch.float32, device=self.dev)
        self.tables = {name: self.table(f) for name, f in self.fitted.items()}
        bias = [self.fitted[name]["bias"] if name in self.fitted else 0.0 for name in PROFILES]
        self.bias = torch.tensor(bias, device=self.dev)
        zeros = torch.zeros(n, device=self.dev)
        self.profile = torch.zeros(n, dtype=torch.long, device=self.dev)
        self.fresh = torch.ones(n, dtype=torch.bool, device=self.dev)
        keys = ["gain", "bias", "rho", "k", "thr", "near", "far", "flying", "scale"]
        self.p = {k: zeros.clone() for k in keys}
        self.left = torch.zeros(n, dtype=torch.long, device=self.dev)
        self.right = torch.zeros(n, dtype=torch.long, device=self.dev)
        self.e = torch.zeros(n, 2, device=self.dev)  # AR states of the mono scale and residual
        self.fields = None  # AR states of the image fields, allocated at the first frame

    def table(self, fitted):
        bins, mean, std = torch.tensor(fitted["residual"], device=self.dev).T.contiguous()
        names = ("edge_gain", "scale_p", "scale_q")
        out = {k: torch.tensor(fitted[k], device=self.dev) for k in names}
        return out | {"bins": bins, "res_mean": mean, "res_std": std}

    def uniform(self, n, lo, hi):
        return lo + (hi - lo) * torch.rand(n, generator=self.gen, device=self.dev)

    def reset(self, ids):
        n, a = len(ids), self.active
        u = torch.rand(n, generator=self.gen, device=self.dev)
        self.profile[ids] = torch.searchsorted(self.cum, u, right=True).clamp(max=len(PROFILES) - 1)
        gain = self.uniform(n, *self.amp)
        sign = self.signs[torch.randint(len(self.signs), (n,), generator=self.gen, device=self.dev)]
        holes = self.uniform(n, *a["holes"])
        self.p["gain"][ids] = gain
        self.p["bias"][ids] = sign * self.bias[self.profile[ids]] * gain
        self.p["rho"][ids] = self.uniform(n, *RHO)
        self.p["k"][ids] = self.uniform(n, *a["k"])
        # a field of unit variance exceeds thr on a share `holes` of the pixels
        self.p["thr"][ids] = math.sqrt(2) * torch.erfinv(1 - 2 * holes)
        self.p["near"][ids] = self.uniform(n, *a["min_range"])
        self.p["far"][ids] = self.uniform(n, *a["max_range"])
        self.p["flying"][ids] = self.uniform(n, *a["flying"])
        self.p["scale"][ids] = self.uniform(n, -a["scale"], a["scale"])
        cols = (n,)
        self.left[ids] = torch.randint(a["border"] + 1, cols, generator=self.gen, device=self.dev)
        self.right[ids] = torch.randint(a["border"] + 1, cols, generator=self.gen, device=self.dev)
        self.fresh[ids] = True

    def step_states(self, rows, shape):
        """Advance the AR(1) states of `rows`; a fresh episode starts from the stationary
        distribution."""
        # field -> (profile that uses it, its correlation)
        sigmas = {name: (name, f["corr_sigma"]) for name, f in self.fitted.items()}
        sigmas["active"] = ("active", self.active["corr_sigma"])
        sigmas["holes"] = ("active", self.active["hole_sigma"])
        if self.fields is None or self.fields["mono"].shape[1:] != shape[1:]:
            self.fields = {k: torch.zeros(self.n, *shape[1:], device=self.dev) for k in sigmas}
            self.fresh[:] = True
        rho = torch.where(self.fresh[rows], 0.0, self.p["rho"][rows])
        keep, new = rho[:, None, None], torch.sqrt(1 - rho**2)[:, None, None]
        w = torch.randn(len(rows), 2, generator=self.gen, device=self.dev)
        self.e[rows] = keep[:, :, 0] * self.e[rows] + new[:, :, 0] * w
        # a field only for the envs whose profile uses it, the profile is fixed per episode
        prof = self.profile[rows]
        for k, (name, sigma) in sigmas.items():
            sel = prof == PROFILES.index(name)
            r, f = rows[sel], field((int(sel.sum()), *shape[1:]), sigma, self.gen)
            self.fields[k][r] = keep[sel] * self.fields[k][r] + new[sel] * f
        self.fresh[rows] = False

    def fitted_depth(self, depth, rows, name):
        m, t, g = self.fitted[name], self.tables[name], self.p["gain"][rows, None, None]
        b = (torch.bucketize(depth.clamp(max=1e6), t["bins"], right=True) - 1).clamp(min=0)
        e = self.e[rows, :, None, None]
        gain = t["edge_gain"][edge_distance(depth, len(t["edge_gain"]) - 1)]
        f = heavy(self.fields[name][rows], m["res_tail"])
        res = t["res_mean"][b] + t["res_std"][b] * gain * torch.exp(m["res_gain_std"] * e[:, 1]) * f
        # the Gaussian AR state through the quantiles keeps the correlation in time
        u = 0.5 * (1 + torch.erf(e[:, 0] / math.sqrt(2)))
        scale = interp(u, t["scale_p"], t["scale_q"])
        log_err = self.p["bias"][rows, None, None] + g * (scale + res)
        # the edge blur grows with the amplitude up to the fitted one, zero amplitude is identity
        inv = 1 / depth
        inv = inv + g.clamp(max=1) * (blur(inv, m["edge_sigma"]) - inv)
        return torch.exp(log_err) / inv

    def active_depth(self, depth, rows):
        p = {k: v[rows, None, None] for k, v in self.p.items()}
        noise = self.fields["active"][rows]
        z = depth * torch.exp(p["scale"] + p["k"] * depth.clamp(max=1e6) * noise)
        u = torch.rand(depth.shape, generator=self.gen, device=self.dev)
        z = torch.where(edges(depth) & (u < p["flying"]), blur_inverse(depth, 1.0), z)
        cols = torch.arange(depth.shape[-1], device=self.dev)
        left, right = self.left[rows, None], depth.shape[-1] - self.right[rows, None]
        border = (cols < left) | (cols >= right)
        holes = self.fields["holes"][rows] > p["thr"]
        missing = holes | (depth < p["near"]) | (depth > p["far"])
        return torch.where(missing | border[:, None, :], torch.nan, z)

    def __call__(self, depth, rows=None):
        """Metric depth [N, H, W] of envs `rows` (all by default) -> noisy metric depth, NaN where
        the sensor gives no data."""
        rows = torch.arange(self.n, device=self.dev) if rows is None else rows
        self.step_states(rows, depth.shape)
        prof, out = self.profile[rows], depth.clone()
        for name in PROFILES[1:]:
            sel = prof == PROFILES.index(name)
            if name in self.fitted:
                out[sel] = self.fitted_depth(depth[sel], rows[sel], name)
            else:
                out[sel] = self.active_depth(depth[sel], rows[sel])
        return out
