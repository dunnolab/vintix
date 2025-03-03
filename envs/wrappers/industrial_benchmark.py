import gymnasium
import industrial_benchmark_python.IBGym as IBGym
import numpy as np
from typing import Optional
import os

TASKS = [
    'industrial-benchmark-0-v1',
    'industrial-benchmark-5-v1',
    'industrial-benchmark-10-v1',
    'industrial-benchmark-15-v1',
    'industrial-benchmark-20-v1',
    'industrial-benchmark-25-v1',
    'industrial-benchmark-30-v1',
    'industrial-benchmark-35-v1',
    'industrial-benchmark-40-v1',
    'industrial-benchmark-45-v1',
    'industrial-benchmark-50-v1',
    'industrial-benchmark-55-v1',
    'industrial-benchmark-60-v1',
    'industrial-benchmark-65-v1',
    'industrial-benchmark-70-v1',
    'industrial-benchmark-75-v1',
    'industrial-benchmark-80-v1',
    'industrial-benchmark-85-v1',
    'industrial-benchmark-90-v1',
    'industrial-benchmark-95-v1',
    'industrial-benchmark-100-v1',
]

class IBGymnasiumWrapper(gymnasium.Env):
    def __init__(
            self, 
            setpoint
        ):

        self.env = IBGym.IBGym(
            setpoint = setpoint,
            reward_type = "classic",
            action_type = "continuous",
            observation_type = "classic",
            reset_after_timesteps = 250
        )

        self.action_space = gymnasium.spaces.Box(np.array([-1, -1, -1]), np.array([+1, +1, +1]))

        single_low = np.array([0, 0, 0, 0, 0, 0])
        single_high = np.array([100, 100, 100, 100, 1000, 1000])

        self.observation_space = gymnasium.spaces.Box(low=single_low, high=single_high)

    def reset(self, seed=0, **kwargs):
        """Reset env"""
        obs, inf = self.env.reset(seed=seed), {}
        return obs, inf

    def step(self, action: np.ndarray):
        """Make step"""
        obs, rew, term, inf = self.env.step(action)
        trunc = False
        return obs, rew, term, trunc, inf

def get_gym_env(
        task_name: str,
        seed: Optional[int] = None
) -> gymnasium.Env:
    """Get Industrial Benchmark Gym Environment

    Get Industrial Benchmark Gym Environment

    Args:
        task_name: string name for task
        seed: optional seed for reproducibility

    Returns:
        gymnasium.Env: Gym wrapped Industrial Benchmark environment
    """
    
    setpoint = int(task_name.split('-')[2])
    return IBGymnasiumWrapper(setpoint)
