import argparse
import dataclasses
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _cast(value, typ):
    if typ is bool:
        return value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes")
    if typ is int:
        # PyYAML reads "1e6" as a string, so go through float
        return int(float(value))
    if typ is float:
        return float(value)
    if typ is str:
        return str(value)
    return yaml.safe_load(value) if isinstance(value, str) else value


def load_config(cls, path=None, overrides=()):
    cfg = cls()
    types = {f.name: f.type for f in dataclasses.fields(cls)}
    items = list((yaml.safe_load(Path(path).read_text()) or {}).items()) if path else []
    items += [item.split("=", 1) for item in overrides]
    for key, value in items:
        if key not in types:
            raise KeyError(f"unknown config key: {key}")
        setattr(cfg, key, _cast(value, types[key]))
    return cfg


def parse_args(cls, argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--set", nargs="+", action="extend", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    cfg = load_config(cls, args.config, args.set)
    if args.seed is not None:
        cfg.seed = args.seed
    return cfg


def git_info():
    def git(*args):
        res = subprocess.run(["git", *args], capture_output=True, text=True)
        return res.stdout.strip() if res.returncode == 0 else None

    commit = git("rev-parse", "HEAD") or "none"
    dirty = bool(git("status", "--porcelain"))
    return f"commit: {commit}\ndirty: {str(dirty).lower()}\n"


def make_run_dir(cfg) -> Path:
    """Creates <log_dir>/<exp>/<variant>_<env>_s<seed>_<time>/ with the config and git state."""
    env = cfg.env_id.split(":")[-1]  # "module:Env-v0" ids name the module only for the import
    name = f"{cfg.variant}_{env}_s{cfg.seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    base = Path(cfg.log_dir) / cfg.exp
    path, i = base / name, 1
    while path.exists():
        path, i = base / f"{name}_{i}", i + 1
    (path / "checkpoints").mkdir(parents=True)
    (path / "config.yaml").write_text(yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False))
    (path / "git.txt").write_text(git_info())
    return path
