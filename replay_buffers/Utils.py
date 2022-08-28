import numpy as np
import gym


def get_data_spec(env, priority=None):
    state = env.reset()
    if isinstance(env.action_space, gym.spaces.Discrete):
        action = np.random.randint(env.action_space.n)
    else:
        "Implement method to sample non-discrete action"
    next_state, reward, done, _, _ = env.step(action)
    specs = {"state": [type(state), np.shape(state)],
             "action": [type(action), np.shape(action)],
             "reward": [type(reward), np.shape(reward)],
             "next_state": [type(next_state), np.shape(next_state)],
             "done": [type(done), np.shape(done)],
             "priority": [type(priority), np.shape(priority)]}
    return specs
