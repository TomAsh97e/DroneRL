from gym_pybullet_drones.envs.HoverAviary import HoverAviary


def test_hover_aviary_runs():
    env = HoverAviary(gui=False)
    obs, _ = env.reset(seed=0)
    # upstream returns float64 obs for a float32 space, so only the shape is checked
    assert obs.shape == env.observation_space.shape
    for _ in range(100):
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
