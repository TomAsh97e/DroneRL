import csv
import json
import time
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Training metrics go to TensorBoard, evaluation results also to eval.csv and final.json."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.tb = SummaryWriter(str(run_dir / "tb"))
        self.fields = None
        self.start = time.time()

    def scalar(self, tag, value, step):
        self.tb.add_scalar(tag, value, step)

    def eval(self, step, returns, **extra):
        row = {
            "step": step,
            "time": round(time.time() - self.start, 1),
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            **extra,
        }
        for key, value in row.items():
            if key not in ("step", "time"):
                self.tb.add_scalar(f"eval/{key}", value, step)
        with (self.run_dir / "eval.csv").open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if self.fields is None:
                writer.writeheader()
                self.fields = list(row)
            writer.writerow(row)
        sr = f", SR {row['success_rate']:.2f}" if "success_rate" in row else ""
        mean, std = row["mean_return"], row["std_return"]
        print(f"step {step}: eval {mean:.1f} ± {std:.1f}{sr}", flush=True)
        return row

    def final(self, step, returns, **extra):
        """Result of the final evaluation of the checkpoint saved at `step`."""
        result = {
            "step": int(step),
            "episodes": len(returns),
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            **extra,
        }
        (self.run_dir / "final.json").write_text(json.dumps(result, indent=2))
        sr = f", SR {result['success_rate']:.2f}" if "success_rate" in result else ""
        print(
            f"final {result['mean_return']:.1f} ± {result['std_return']:.1f}{sr}"
            f" (step {step}) {self.run_dir}",
            flush=True,
        )
        return result

    def close(self):
        self.tb.close()
