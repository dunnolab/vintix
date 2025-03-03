# Docker Image for Industrial Benchmark domain

## Installation Phases
1) Installing PyTorch and flash_attn
2) [Patching](industrial_benchmark_ids.patch) Industrial Benchmark `IDS.py` file in order to be compatible with
`numpy>=1.24.4`
3) [Patching](industrial_benchmark_ibgym.patch) and installing Industrial Benchmark

## Initializing Industrial Behcmark Gymnasium environment

In newly created environment run these commands:
```python
from envs.wrappers.industrial_benchmark import get_gym_env, TASKS

print(f"Available tasks are:", *TASKS, sep='\n')

# Initialize env
env = get_gym_env(task_name="industrial-benchmark-0-v1")
```
