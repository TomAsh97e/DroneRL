# DroneRL

DroneRL is a learning project in which a drone learns to choose its own flight path with reinforcement learning, first in simulation and later on a real quadcopter with basic sensors and an RGB camera. The current stage trains a teacher policy with PPO in a GPU-parallel navigation environment with depth images, privileged lidar and procedurally generated obstacle maps, compared against an A* baseline. The roadmap and milestones are in `PLAN.md` and the detailed task plans in `plan/` (both in Polish). Install the dependencies with `uv sync` and run the tests with `uv run pytest`.
