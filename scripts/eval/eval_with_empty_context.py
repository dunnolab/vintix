import torch
import metaworld
import gymnasium as gym
from vintix import Vintix


PATH_TO_CHECKPOINT = "/path/to/checkpoint"
model = Vintix()
model.load_model(PATH_TO_CHECKPOINT)
model.to(torch.device('cuda'))
model.eval()

task_name = "shelf-place-v2"
env = gym.make(task_name)
model.reset_model(task_name,
                  use_cache=True,
                  torch_dtype=torch.float16)
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
