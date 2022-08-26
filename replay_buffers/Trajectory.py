import numpy as np


class Trajectory:
    def __init__(self, state, action, reward, next_state, done, priority=None):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.priority = priority

    def data_spec(self):
        specs = {"state": [type(self.state), np.shape(self.state)],
                 "action": [type(self.action), np.shape(self.action)],
                 "reward": [type(self.reward), np.shape(self.reward)],
                 "next_state": [type(self.next_state), np.shape(self.next_state)],
                 "done": [type(self.done), np.shape(self.done)],
                 "priority": [type(self.priority), np.shape(self.priority)]}
        return specs
