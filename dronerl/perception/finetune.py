"""Fine-tuning of the depth model on NYU train (M3.4).

SILog loss as in the metric fine-tuning of Depth Anything V2. Variant `frozen` trains only the
neck and head, `full` also the DINOv2 encoder with a 10x smaller learning rate. Random crops keep
the focal length, so the metric scale stays consistent; a resize would change it.
"""

import csv
import json
import math
from dataclasses import dataclass

import numpy as np
import torch

from dronerl.perception import eval_depth, nyu
from dronerl.perception.depth_model import DepthModel
from dronerl.utils.run import make_run_dir, parse_args, set_seed

CROP = (416, 544)


@dataclass
class Config:
    exp: str = "m3/depth_ft"
    variant: str = "full"
    env_id: str = "nyu"
    log_dir: str = "runs"
    device: str = "cuda"
    seed: int = 0
    res: int = 518  # chosen in M3.3
    epochs: int = 15
    batch: int = 8
    lr: float = 5e-5
    encoder_lr_ratio: float = 0.1  # 0 freezes the encoder
    weight_decay: float = 0.01
    warmup: int = 100
    silog_lambda: float = 0.5
    jitter: float = 0.2
    bf16: bool = True


def silog(pred, gt, lam):
    m = (gt > 1e-3) & (gt <= nyu.MAX_DEPTH)
    g = torch.log(pred[m].clamp(min=1e-3)) - torch.log(gt[m])
    return torch.sqrt((g**2).mean() - lam * g.mean() ** 2)


def augment(rgb, depth, gen, jitter):
    """Random crop, horizontal flip and colour jitter of uint8 [N, H, W, 3] on the GPU."""
    n, h, w = depth.shape
    y = torch.randint(0, h - CROP[0] + 1, (1,), generator=gen).item()
    x = torch.randint(0, w - CROP[1] + 1, (1,), generator=gen).item()
    rows, cols = slice(y, y + CROP[0]), slice(x, x + CROP[1])
    rgb, depth = rgb[:, rows, cols], depth[:, rows, cols]
    if torch.rand(1, generator=gen).item() < 0.5:
        rgb, depth = rgb.flip(2), depth.flip(2)
    scale = 1 + jitter * (2 * torch.rand(n, 1, 1, 3, generator=gen) - 1)
    offset = 255 * jitter / 2 * (2 * torch.rand(n, 1, 1, 1, generator=gen) - 1)
    rgb = (rgb.float() * scale.to(rgb.device) + offset.to(rgb.device)).clamp(0, 255)
    return rgb.to(torch.uint8), depth


def validate(model, data):
    model.net.eval()
    pred = eval_depth.predict(model, data["rgb"])
    return eval_depth.evaluate(pred, data)["full"]


def train(cfg: Config):
    set_seed(cfg.seed)
    run_dir = make_run_dir(cfg)
    dev = torch.device(cfg.device)
    split = nyu.splits()
    data = nyu.load(split["train"])
    val = nyu.load(split["val"])
    # kept in RAM, at R = 518 the GPU memory goes to the activations
    rgb, depth = torch.as_tensor(data["rgb"]), torch.as_tensor(data["depth"])
    # outside the Eigen crop the filled depth is extrapolated, 0 drops it from the loss
    outside = torch.ones_like(depth[0], dtype=torch.bool)
    outside[nyu.EIGEN_CROP] = False
    depth[:, outside] = 0

    model = DepthModel(cfg.res, cfg.device)
    encoder = list(model.net.backbone.parameters())
    rest = [p for n, p in model.net.named_parameters() if not n.startswith("backbone.")]
    groups = [{"params": rest, "lr": cfg.lr}]
    if cfg.encoder_lr_ratio > 0:
        groups.append({"params": encoder, "lr": cfg.lr * cfg.encoder_lr_ratio})
    else:
        for p in encoder:
            p.requires_grad_(False)
    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * (len(rgb) // cfg.batch)

    def schedule(step):
        return min(1, (step + 1) / cfg.warmup) * 0.5 * (1 + math.cos(math.pi * step / steps))

    lr_sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
    gen = torch.Generator().manual_seed(cfg.seed)
    best, rows = math.inf, []
    for epoch in range(cfg.epochs):
        model.net.train()
        order = torch.randperm(len(rgb), generator=gen)
        losses = []
        for i in range(0, len(order) - cfg.batch + 1, cfg.batch):
            idx = order[i : i + cfg.batch]
            x, y = augment(rgb[idx].to(dev), depth[idx].to(dev), gen, cfg.jitter)
            with torch.autocast(dev.type, torch.bfloat16, enabled=cfg.bf16):
                pred = model.forward(model.preprocess(x), CROP)
            loss = silog(pred, y, cfg.silog_lambda)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            lr_sched.step()
            losses.append(loss.item())
        m = validate(model, val)
        train_loss = float(np.mean(losses))
        rows.append({"epoch": epoch + 1, "train_silog": train_loss, **m})
        print(f"epoch {epoch + 1}: loss {train_loss:.4f}, AbsRel {m['abs_rel']:.4f}", flush=True)
        if m["abs_rel"] < best:
            best = m["abs_rel"]
            model.net.save_pretrained(run_dir / "checkpoints" / "best")
    with (run_dir / "eval.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    best_model = DepthModel(cfg.res, cfg.device, path=run_dir / "checkpoints" / "best")
    test = nyu.load(split["test"])
    result = eval_depth.evaluate(eval_depth.predict(best_model, test["rgb"]), test)
    result["best_epoch"] = int(np.argmin([r["abs_rel"] for r in rows]) + 1)
    (run_dir / "final.json").write_text(json.dumps(result, indent=2))
    print(f"test AbsRel {result['full']['abs_rel']:.4f}, epoch {result['best_epoch']}, {run_dir}")
    return run_dir


def main(argv=None):
    train(parse_args(Config, argv))


if __name__ == "__main__":
    main()
