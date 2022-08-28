import numpy as np
from replay_buffers.Trajectory import Trajectory

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
            traj = Trajectory(state, None, None, None, False)
            while not done:
                action = self.policy(traj)
                next_state, reward, done, _, _ = self.env.step(action)
                traj = Trajectory(state, action, reward, next_state, done)
                ep_return += reward
            returns.append(ep_return)
        return np.mean(returns)

    def AverageEpisodeLength(self, n_episodes=10):
        all_lengths = []
        for episode in range(n_episodes):
            state = self.env.reset()
            ep_length = 0
            done = False
            traj = Trajectory(state, None, None, None, False)
            while not done:
                action = self.policy(traj)
                next_state, reward, done, _, _ = self.env.step(action)
                traj = Trajectory(state, action, reward, next_state, done)
                ep_length += 1
                state = next_state
            all_lengths.append(ep_length)
        return np.mean(all_lengths)
