from gymnasium.envs.registration import register

# gym.make("dronerl.envs:GoalReach-v0") imports this module first, also in subprocess workers
register(id="GoalReach-v0", entry_point="dronerl.envs.goal_reach:GoalReachAviary")
