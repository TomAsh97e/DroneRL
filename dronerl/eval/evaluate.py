import numpy as np

# Each evaluation replays the same episodes, so results are comparable across runs and algorithms
EVAL_SEED = 10_000
# Periodic evaluation picks the best checkpoint, so it must not see the test episodes
VAL_SEED = 20_000


def score(row):
    """Checkpoints are ranked by success rate when the environment reports it, then by return.

    Return alone misleads with potential-based shaping: summed without discounting it pays for
    hovering away from the goal until the time limit.
    """
    return row.get("success_rate", 0.0), row["mean_return"]


def evaluate(env, act, episodes: int, seed: int = EVAL_SEED):
    """Plays `episodes` episodes with policy `act(obs) -> action`.

    Returns (returns, extra), where extra holds the mean episode length and, for environments
    reporting `info["is_success"]`, the success rate.
    """
    returns, lengths, successes = [], [], []
    for i in range(episodes):
        obs, info = env.reset(seed=seed + i)
        done, ret, length = False, 0.0, 0
        while not done:
            obs, reward, terminated, truncated, info = env.step(act(obs))
            ret += float(reward)
            length += 1
            done = terminated or truncated
        returns.append(ret)
        lengths.append(length)
        successes.append(info.get("is_success"))
    extra = {"mean_length": float(np.mean(lengths))}
    if None not in successes:
        extra["success_rate"] = float(np.mean(successes))
    return np.array(returns), extra
