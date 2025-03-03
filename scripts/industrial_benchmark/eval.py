import numpy as np
from envs.wrappers.industrial_benchmark import get_gym_env, TASKS

print(f"Available tasks are:", *TASKS, sep='\n')

for t in sorted(TASKS):
    print(f'Processing env {t}')
    env = get_gym_env(task_name=t)
    print(f'{t}: obs space: {env.observation_space}, acs space: {env.action_space}')
    n_tries = 20
    ep_lens, ep_rews = [], []
    for _ in range(n_tries):
        done = False
        nsteps, tot_rew = 0, 0
        obs, info = env.reset()
        while not done:
            acs = env.action_space.sample()
            obs, rews, term, trunc, info = env.step(action=acs)
            done = term or trunc
            nsteps += 1
            tot_rew += rews
        ep_lens.append(nsteps)
        ep_rews.append(tot_rew)
    print(f'{t}: rand. ep. len.: {np.mean(ep_lens)}, rand. rews: {np.mean(ep_rews)}')
    env.close()
