import numpy as np


class metrics:
    def __init__(self, env, policy):
        self.env = env
        self.policy = policy

    def AverageReturn(self, n_episodes=10):
        returns = []
        for episode in range(n_episodes):
            state = self.env.reset()
            ep_return = 0
            done = False
            while not done:
                action = self.policy(state)
                next_state, reward, done, _ = self.env.step(action)
                ep_return += reward
                state = next_state
            returns.append(ep_return)
        return np.mean(returns)

    def AverageEpisodeLength(self, n_episodes=10):
        all_lengths = []
        for episode in range(n_episodes):
            state = self.env.reset()
            ep_length = 0
            done = False
            while not done:
                action = self.policy(state)
                next_state, reward, done, _ = self.env.step(action)
                ep_length += 1
                state = next_state
            all_lengths.append(ep_length)
        return np.mean(all_lengths)
