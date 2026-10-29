"""Steps per second of the GPU navigation environment with random actions and resets (M2.3).

With --student also the observations of the student (M3.8).
"""

import argparse
import json
import time
from pathlib import Path

import torch

from dronerl.envs.nav_gpu import STUDENT, NavEnv, load_maps


def measure(maps, num_envs, steps, device="cuda", student=None):
    # the densest level with the curriculum mix, as at the end of training
    env = NavEnv(maps, num_envs, device, level=5, mix=0.2, randomize=True, student=student)
    env.reset()
    gen = torch.Generator(device).manual_seed(0)
    actions = [torch.rand(num_envs, 3, generator=gen, device=device) * 2 - 1 for _ in range(16)]
    for a in actions:  # warm-up
        env.step(a)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    resets = torch.zeros((), device=device)
    start = time.perf_counter()
    for i in range(steps):
        _, _, terminated, truncated, _ = env.step(actions[i % len(actions)])
        resets += (terminated | truncated).sum()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "envs": num_envs,
        "steps_per_s": round(num_envs * steps / elapsed),
        "ms_per_step": round(1000 * elapsed / steps, 2),
        "resets_per_step": round(resets.item() / steps, 1),
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, nargs="+", default=[1024, 4096, 16384])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--out", type=Path, default=Path("runs/m2/throughput"))
    parser.add_argument("--student", action="store_true")
    args = parser.parse_args(argv)
    maps = load_maps("train")
    student = STUDENT if args.student else None
    results = [measure(maps, n, args.steps, student=student) for n in args.envs]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "env.json").write_text(json.dumps(results, indent=2))
    lines = [
        "| środowiska | kroki/s | ms/krok | resety/krok | pamięć GPU [GB] |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['envs']} | {r['steps_per_s']} | {r['ms_per_step']} | {r['resets_per_step']} "
        f"| {r['peak_memory_gb']} |"
        for r in results
    ]
    text = f"{torch.cuda.get_device_name()}\n\n" + "\n".join(lines)
    (args.out / "env.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
