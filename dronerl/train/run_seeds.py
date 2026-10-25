"""Runs one config for several seeds, each in its own process."""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

MODULES = {
    "dqn": "dronerl.agents.dqn",
    "ppo": "dronerl.agents.ppo",
    "ppo_gpu": "dronerl.agents.ppo_gpu",
    "rsl_rl": "dronerl.train.rsl_rl_baseline",
    "sb3": "dronerl.train.sb3_baseline",
}


def run(module, config, seed, overrides):
    cmd = [sys.executable, "-m", module, "--config", config, "--seed", str(seed)]
    if overrides:
        cmd += ["--set", *overrides]
    # Small networks gain nothing from extra threads, and parallel runs would oversubscribe the CPU
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    return seed, subprocess.run(cmd, capture_output=True, text=True, env=env)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("algo", choices=MODULES)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--variant")
    parser.add_argument("--set", nargs="+", action="extend", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    overrides = args.set + ([f"variant={args.variant}"] if args.variant else [])
    module, failed = MODULES[args.algo], 0
    with ThreadPoolExecutor(args.jobs) as pool:
        futures = [pool.submit(run, module, args.config, s, overrides) for s in args.seeds]
        for future in as_completed(futures):
            seed, res = future.result()
            if res.returncode == 0:
                last = res.stdout.strip().splitlines()[-1:] or [""]
                print(f"seed {seed}: ok | {last[0]}", flush=True)
            else:
                failed += 1
                print(f"seed {seed}: failed\n{res.stderr[-3000:]}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
