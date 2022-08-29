import numpy as np


class AverageEpisodeLengthObserver:
    def __init__(self, buffer_size=10):
        self.buffer_size = buffer_size
        self.buffer = []
        self.step_count = 0
        self.buffer_episodes = 0

    def call(self, traj):
        if traj.done:
            self.step_count += 1
            self.buffer.append(self.step_count)
            self.step_count = 0
            if len(self.buffer) > self.buffer_size:
                del self.buffer[0]
        else:
            self.step_count += 1

    def result(self):
        return np.mean(self.buffer)


class AverageReturnObserver:
    def __init__(self, buffer_size=10):
        self.buffer_size = buffer_size
        self.buffer = []
        self.episode_buffer = []
        self.buffer_episodes = 0

    def call(self, traj):
        if traj.done:
            self.episode_buffer.append(traj.reward)
            self.buffer.append(self.episode_buffer)
            self.episode_buffer = []
            if len(self.buffer) > self.buffer_size:
                del self.buffer[0]
        else:
            self.episode_buffer.append(traj.reward)

    def result(self):
        returns = []
        for episode in self.buffer:
            returns.append(np.sum(episode))
        return np.mean(returns)


class MinReturnObserver:
    def __init__(self, buffer_size=10):
        self.buffer_size = buffer_size
        self.buffer = []
        self.episode_buffer = []
        self.buffer_episodes = 0

    def call(self, traj):
        if traj.done:
            self.episode_buffer.append(traj.reward)
            self.buffer.append(self.episode_buffer)
            self.episode_buffer = []
            if len(self.buffer) > self.buffer_size:
                del self.buffer[0]
        else:
            self.episode_buffer.append(traj.reward)

    def result(self):
        returns = []
        for episode in self.buffer:
            returns.append(np.sum(episode))
        return np.min(returns)


class MaxReturnObserver:
    def __init__(self, buffer_size=10):
        self.buffer_size = buffer_size
        self.buffer = []
        self.episode_buffer = []
        self.buffer_episodes = 0

    def call(self, traj):
        if traj.done:
            self.episode_buffer.append(traj.reward)
            self.buffer.append(self.episode_buffer)
            self.episode_buffer = []
            if len(self.buffer) > self.buffer_size:
                del self.buffer[0]
        else:
            self.episode_buffer.append(traj.reward)

    def result(self):
        returns = []
        for episode in self.buffer:
            returns.append(np.sum(episode))
        return np.max(returns)
