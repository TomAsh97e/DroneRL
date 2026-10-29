"""Depth from one RGB image and its reduction to the image of the policy.

`reduce_depth` is shared by the evaluation here, the noise model fit and later `depth_node` (M4),
so the policy sees the same kind of image in simulation and on the drone. `student_image` is
the whole path from a camera image to the input of the student.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from dronerl import contract


def reduce_depth(depth, size):
    """Metric depth [..., H, W] -> [..., h, w] by averaging inverse depth over the valid pixels.

    NaN and values <= 0 are invalid; a cell without valid pixels is NaN. The average is taken in
    inverse depth, as the policy sees it, so one near pixel is not washed out by far ones.
    """
    shape = depth.shape
    d = depth.reshape(-1, 1, *shape[-2:]).float()
    ok = d > 0  # false for NaN; inf is a valid "far" reading with inverse depth 0
    inv = torch.where(ok, 1 / d, 0)
    count = F.adaptive_avg_pool2d(ok.float(), size)
    mean = F.adaptive_avg_pool2d(inv, size) / count
    out = torch.where(count > 0, 1 / mean, torch.nan)
    return out.reshape(*shape[:-2], *out.shape[-2:])


def policy_size(height, width, fx):
    """Size of an image of focal length fx [px] after reduction to the angular pixel size of the
    policy camera (focal contract.FOCAL at 64 x 48)."""
    scale = contract.FOCAL / fx
    return round(height * scale), round(width * scale)


def crop_fov(image, fx, hfov=contract.HFOV):
    """Central crop [..., h, w] of an image with focal length fx [px] to the horizontal field of
    view hfov and the aspect ratio of the policy image. Even sides keep the crop centred, a side
    of a narrower camera stays whole."""
    w = 2 * round(fx * math.tan(hfov / 2))
    h = 2 * round(w * contract.IMG_H / contract.IMG_W / 2)
    h, w = min(h, image.shape[-2]), min(w, image.shape[-1])
    top, left = (image.shape[-2] - h) // 2, (image.shape[-1] - w) // 2
    return image[..., top : top + h, left : left + w]


def reduce_to_policy(depth, fx):
    """Metric depth [..., H, W] of a camera with focal length fx [px] -> the angular pixel size of
    the policy camera."""
    return reduce_depth(depth, policy_size(*depth.shape[-2:], fx))


MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
PATCH = 14


def input_size(h, w, res):
    """Model input for an h x w image: shorter side res, aspect ratio kept, multiples of 14."""
    s = res / min(h, w)
    return round(h * s / PATCH) * PATCH, round(w * s / PATCH) * PATCH


def preprocess(rgb, size, device="cpu"):
    """uint8 [N, H, W, 3] -> normalized [N, 3, *size], as the DPTImageProcessor (bicubic, ImageNet
    normalization), but in torch."""
    x = torch.as_tensor(rgb, device=device).permute(0, 3, 1, 2).float() / 255
    x = F.interpolate(x, size, mode="bicubic", align_corners=False)
    mean = torch.tensor(MEAN, device=device)[:, None, None]
    std = torch.tensor(STD, device=device)[:, None, None]
    return (x - mean) / std


class DepthModel:
    """Depth Anything V2 Small, metric indoor variant, with the shorter input side resized to `res`.

    Preprocessing follows its DPTImageProcessor (aspect ratio kept, sides multiples of 14, bicubic,
    ImageNet normalization), but in torch on the model's device.
    """

    def __init__(self, res=518, device="cuda", dtype=torch.float32, path=MODEL_ID):
        from transformers import AutoModelForDepthEstimation

        self.res, self.device, self.dtype = res, torch.device(device), dtype
        self.net = AutoModelForDepthEstimation.from_pretrained(path).to(self.device, dtype).eval()

    def input_size(self, h, w):
        return input_size(h, w, self.res)

    def preprocess(self, rgb):
        """uint8 [N, H, W, 3] -> normalized [N, 3, h, w] at the model resolution."""
        size = self.input_size(*rgb.shape[1:3])
        return preprocess(rgb, size, self.device).to(self.dtype)

    def forward(self, x, size):
        """Metric depth [N, *size] from preprocessed images, differentiable for fine-tuning."""
        out = self.net(pixel_values=x).predicted_depth[:, None]
        return F.interpolate(out.float(), size, mode="bilinear", align_corners=False)[:, 0]

    @torch.no_grad()
    def __call__(self, rgb):
        return self.forward(self.preprocess(rgb), rgb.shape[1:3])


class OnnxDepth:
    """The depth model exported by deploy/export_onnx.py, used like DepthModel by depth_node.

    The export fixes the input size, so one file serves images of one size and aspect ratio.
    """

    def __init__(self, path, res=518, device="cuda"):
        import onnxruntime as ort

        providers, opts = ["CPUExecutionProvider"], ort.SessionOptions()
        if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            ort.preload_dlls()  # CUDA and cuDNN of the pip wheels
            providers.insert(0, "CUDAExecutionProvider")
            # the CPU only feeds the GPU, spinning threads would take it from the simulation
            opts.intra_op_num_threads = 1
            opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(str(path), opts, providers=providers)
        self.res = res
        self.size = tuple(self.session.get_inputs()[0].shape[-2:])

    def __call__(self, rgb):
        size = input_size(*rgb.shape[1:3], self.res)
        if size != self.size:
            raise ValueError(f"the images need input {size}, the model takes {self.size}")
        x = preprocess(rgb, size).numpy()
        out = [self.session.run(None, {"image": x[i : i + 1]})[0] for i in range(len(x))]
        out = torch.from_numpy(np.concatenate(out))[:, None]
        return F.interpolate(out, rgb.shape[1:3], mode="bilinear", align_corners=False)[:, 0]


def student_image(depth, rgb, fx, crop=None):
    """Student image [2, h, w] from one RGB image uint8 [H, W, 3] of a camera with focal length
    fx [px]: metric depth of the whole image from `depth` (DepthModel or OnnxDepth), then the crop
    (`crop_fov` by default), reduction and the encoding of the contract.

    The model sees the whole image, as in the evaluation of M3 and the noise model fit: on a
    magnified crop it puts everything 8-12% closer (NYU, M4.6).
    """
    return student_depth(depth(torch.as_tensor(rgb)[None])[0].cpu(), fx, crop)


def student_depth(depth, fx, crop=None):
    """Student image from metric depth [H, W] of a camera with focal length fx [px], of a model or
    of a depth camera (NaN missing, inf no hit)."""
    depth = crop_fov(depth, fx) if crop is None else crop(depth)
    return contract.encode_student_depth(reduce_to_policy(depth, fx))
