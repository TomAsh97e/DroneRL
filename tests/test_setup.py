import importlib

import gymnasium as gym
import pytest
import torch

MODULES = ["dronerl", "stable_baselines3", "tensorboard", "yaml", "matplotlib"]


@pytest.mark.parametrize("module", MODULES)
def test_import(module):
    importlib.import_module(module)


@pytest.mark.parametrize(
    "env_id, kwargs",
    [("CartPole-v1", {}), ("Pendulum-v1", {}), ("LunarLander-v3", {"continuous": True})],
)
def test_make_env(env_id, kwargs):
    env = gym.make(env_id, **kwargs)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    obs, *_ = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    env.close()


def test_torch():
    x = torch.ones(3, 3)
    assert torch.equal(x @ x, torch.full((3, 3), 3.0))
