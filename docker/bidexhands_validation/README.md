# Docker Image for Bi-DexHands Domain

## Installation Phases
1) Installing PyTorch and flash_attn
2) [Patching](isaacgym_torch_utils.patch) IsaacGym `torch_utils.py` file in order to be compatible with
`numpy>=1.24.4`
3) Installing IsaacGym
4) [Patching](bidexhands.patch) and installing Bi-DexHands

## Initializing Bi-DexHands Gym environment

In newly created environment run these commands:
```python
from envs.wrappers.bidexhands import get_gym_env, TASKS

print(f"Available tasks are:", *TASKS, sep='\n')

# Initialize env
env = get_gym_env(task_name="ShadowHandCatchUnderarm")

# Don`t forget to .close before initializing new env!
env.close()
```

## Notes

This Docker image supports only CPU-based environments due to the lack of
Vulcan installation.




