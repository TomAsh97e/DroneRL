"""Hyperparameter search for ppo_gpu.py with Optuna (M2.8).

Objective: mean validation SR over the three levels, averaged over the last two evaluations to
reduce noise. The ‖Δa‖ penalty stays fixed, the objective does not reward smoothness.
"""

import argparse
import dataclasses
import gc
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import optuna  # noqa: E402
import torch  # noqa: E402
from optuna.visualization.matplotlib import plot_optimization_history  # noqa: E402

from dronerl.agents import ppo_gpu  # noqa: E402
from dronerl.utils.run import load_config  # noqa: E402

SPACE = {
    "learning_rate": (1e-4, 1e-3),
    "ent_coef": (0.0, 0.01),
    "gamma": [0.98, 0.99, 0.995],
    "gae_lambda": (0.9, 0.98),
    "clip_coef": [0.1, 0.2, 0.3],
    "num_steps": [16, 32, 64],
    "num_minibatches": [2, 4, 8],
    "update_epochs": [3, 5, 8],
    "init_std": [0.3, 0.5, 1.0],
    "prox_w": (0.0, 0.3),
}


def sample(trial):
    params = {}
    for name, space in SPACE.items():
        if isinstance(space, list):
            params[name] = trial.suggest_categorical(name, space)
        else:
            params[name] = trial.suggest_float(name, *space, log=name == "learning_rate")
    return params


def objective(trial, base, steps, evals):
    # a pruned trial leaves its tensors in the frames of the exception, 8 GB hold only one run
    gc.collect()
    torch.cuda.empty_cache()
    params = sample(trial)
    variant = f"trial{trial.number:03d}"
    cfg = dataclasses.replace(base, total_steps=steps, eval_every=steps // evals, variant=variant)
    cfg = dataclasses.replace(cfg, **params)
    history = []

    def report(step, row):
        history.append(row["success_rate"])
        trial.report(row["success_rate"], step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    run_dir = ppo_gpu.train(cfg, on_eval=report)
    trial.set_user_attr("run_dir", str(run_dir))
    return sum(history[-2:]) / len(history[-2:])


def summary(study, out, top=10):
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    done.sort(key=lambda t: -t.value)
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    names = list(SPACE)
    lines = [
        f"Triale: {len(study.trials)}, ukończone: {len(done)}, przycięte: {len(pruned)}",
        "",
        "| trial | SR walidacja | " + " | ".join(names) + " |",
        "|---|---|" + "---|" * len(names),
    ]
    for t in done[:top]:
        cells = [f"{t.params[n]:.3g}" for n in names]
        lines.append(f"| {t.number} | {t.value:.3f} | " + " | ".join(cells) + " |")
    if len(done) > 1:
        importance = optuna.importance.get_param_importances(study)
        lines += ["", "| parametr | ważność |", "|---|---|"]
        lines += [f"| {k} | {v:.3f} |" for k, v in importance.items()]
        plot_optimization_history(study)
        plt.gcf().set_size_inches(8, 5)
        plt.tight_layout()
        plt.savefig(out / "history.png", dpi=120)
        plt.close("all")
    trials = [
        {
            "number": t.number,
            "state": t.state.name,
            "value": t.value,
            "params": t.params,
            "intermediate": t.intermediate_values,
            **t.user_attrs,
        }
        for t in study.trials
    ]
    (out / "trials.json").write_text(json.dumps(trials, indent=2))
    text = "\n".join(lines)
    (out / "summary.md").write_text(text + "\n")
    print(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="the variant chosen in M2.7")
    parser.add_argument("--steps", type=int, required=True, help="environment steps per trial")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--evals", type=int, default=10, help="validations per trial")
    parser.add_argument("--out", type=Path, default=Path("runs/m2/optuna"))
    parser.add_argument("--study", default="m2.8")
    args = parser.parse_args(argv)

    base = dataclasses.replace(load_config(ppo_gpu.Config, args.config), exp="m2/m2.8")
    args.out.mkdir(parents=True, exist_ok=True)
    # early SR says little, the curriculum starts on easy levels
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=int(0.3 * args.steps))
    study = optuna.create_study(
        study_name=args.study,
        storage=f"sqlite:///{args.out / 'study.db'}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=0),
        pruner=pruner,
        load_if_exists=True,
    )
    left = args.trials - sum(t.state.is_finished() for t in study.trials)
    if not study.trials:
        # the starting configuration as trial 0, a reference under the same budget
        study.enqueue_trial({k: getattr(base, k) for k in SPACE})
    if left > 0:
        study.optimize(
            lambda t: objective(t, base, args.steps, args.evals),
            n_trials=left,
            timeout=args.hours * 3600,
        )
    summary(study, args.out)


if __name__ == "__main__":
    main()
