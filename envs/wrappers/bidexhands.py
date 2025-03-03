import os
from typing import Any, Dict, Optional, Tuple

import gym
import numpy as np
from bidexhands.utils.config import get_args, parse_sim_params, load_cfg
from bidexhands.utils.parse_task import parse_task
from bidexhands.utils.process_marl import get_AgentIndex
from bidexhands.tasks.hand_base.base_task import BaseTask
import torch

TASKS = [
    # 'ShadowHandOver', - unavailable due to current architecture
    'ShadowHandCatchUnderarm',
    'ShadowHandTwoCatchUnderarm',
    'ShadowHandCatchAbreast',
    'ShadowHandLiftUnderarm',
    'ShadowHandCatchOver2Underarm',
    'ShadowHandDoorCloseInward',
    'ShadowHandDoorCloseOutward',
    'ShadowHandDoorOpenInward',
    'ShadowHandDoorOpenOutward',
    'ShadowHandBottleCap',
    'ShadowHandPushBlock',
    'ShadowHandSwingCup',
    'ShadowHandGraspAndPlace',
    'ShadowHandScissors',
    'ShadowHandSwitch',
    'ShadowHandPen',
    'ShadowHandReOrientation',
    'ShadowHandKettle',
    'ShadowHandBlockStack'
]


class BiDexGymEnv(gym.Env):
    """Gym wrapper for BiDex-Hands

    Args:
        env: bidexhands task instance
    """

    def __init__(self, env: BaseTask):
        self.env = env
        assert self.env.num_agents == 1
        assert self.env.num_envs == 1
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Any]:
        """Reset Environment

        Args:
            seed: optional seed for reproducibility
            options: placeholder to follow gym API

        Returns:
            Tuple[np.ndarray, Any]: observation and info tuple
        """
        obs = self.env.reset()
        obs = obs.squeeze().detach().cpu().numpy()
        return obs, None

    def step(
            self,
            action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Environment Step

        Environment Step

        Args:
            action: numpy array with selected action

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]: MDP
                timestamp
        """
        actions = torch.from_numpy(action.astype(np.float32))
        actions = actions.unsqueeze(0)
        observations, rewards, terminations, info = self.env.step(
            actions=actions)
        obs = observations.squeeze().detach().cpu().numpy()
        rews = rewards.squeeze().detach().cpu().numpy().item()
        terminals = bool(terminations.squeeze().detach().cpu().numpy().item())
        return obs, rews, terminals, False, info

    def close(self) -> None:
        """Close Environment

        Destroy IsaacGym simulation as it handles only one
        simulation per Python process
        """
        self.env.task.gym.destroy_sim(self.env.task.sim)


def get_gym_env(
        task_name: str,
        seed: Optional[int] = None
) -> BiDexGymEnv:
    """Get BiDex-Hands Gym Environment

    Get BiDex-Hands Gym Environment

    Args:
        task_name: string name for task
        seed: optional seed for reproducibility

    Returns:
        BiDexGymEnv: Gym wrapped BiDex-Hands Environment
    """
    # Locate bidexhands installation
    import bidexhands
    bidexhands_path = os.path.dirname(bidexhands.__file__)
    # Get arguments for CPU installation
    args = get_args(task_name=task_name, algo='ppo')
    args.sim_device = 'cpu'
    args.sim_device_type = 'cpu'
    args.device = 'cpu'
    args.pipeline = 'CPU'
    args.headless = True
    args.use_gpu = False
    args.use_gpu_pipeline = False
    args.num_envs = 1
    if seed is not None:
        args.seed = seed
    # Retrieve environment
    prev_cwd = os.getcwd()
    os.chdir(bidexhands_path)
    cfg, cfg_train, logdir = load_cfg(args)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    agent_index = get_AgentIndex(cfg)
    task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)
    os.chdir(prev_cwd)
    return BiDexGymEnv(env)
