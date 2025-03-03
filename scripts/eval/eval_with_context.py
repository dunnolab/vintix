import torch
import metaworld
import gymnasium as gym
import numpy as np
from vintix import Vintix


PATH_TO_CHECKPOINT = "/path/to/checkpoint"
model = Vintix()
model.load_model(PATH_TO_CHECKPOINT)
model.to(torch.device('cuda'))
model.eval()

task_name = "Humanoid-v4"
# task_name = "shelf-place-v2"
env = gym.make(task_name)

# collect prompt
# you can collect any prompt you want
# there are a simple example of using model with prompt
prompt_len = 5000
cur_len = 0
prompt_obs = []
prompt_acs = []
prompt_rews = []
prompt_steps = []

while cur_len < prompt_len:
    observation, info = env.reset()
    done = False
    step = 0
    while not done and cur_len < prompt_len:
        prompt_obs.append(observation)
        prompt_steps.append(step)
        action = env.action_space.sample()
        observation, reward, termined, truncated, info = env.step(action)
        prompt_acs.append(action)
        prompt_rews.append(reward)
        done = termined or truncated
        step += 1
        cur_len += 1

prompt = (prompt_obs, prompt_acs, prompt_rews, prompt_steps)
model.reset_model(task_name,
                  use_cache=True,
                  torch_dtype=torch.float16,
                  prompt=prompt)
max_episodes = 50
episode_rewards = []
for episode in range(max_episodes):
    cur_ep_rews = []
    observation, info = env.reset()
    reward = None
    done = False
    while not done:
        action = model.get_next_action(observation=observation,
                                       prev_reward=reward)
        observation, reward, termined, truncated, info = env.step(action)

        done = termined or truncated
        cur_ep_rews.append(reward)
    episode_rewards.append(sum(cur_ep_rews))
print(f"Rewards per episode for {task_name}: {episode_rewards}")