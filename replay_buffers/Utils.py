import numpy as np
import gym


def get_data_spec(env, continuous=True, priority=None):
    state = env.reset()
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    specs = {"state": [type(state), np.shape(state)],
             "action": [type(action), np.shape(action)],
             "reward": [type(reward), np.shape(reward)],
             "next_state": [type(next_state), np.shape(next_state)],
             "done": [type(done), np.shape(done)],
             "priority": [type(priority), np.shape(priority)]}
    return specs
