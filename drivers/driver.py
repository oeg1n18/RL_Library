import numpy as np


class driver:
    def __init__(self, env, policy, replay_buffer):
        self.env = env
        self.policy = policy
        self.replay_buffer = replay_buffer

    def collect(self, n_episodes):
        for episode in range(n_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.policy(state)
                next_state, reward, done, _ = self.env.step(action)
                self.replay_buffer.add_trajectory(state, action, reward, next_state, done)
                state = next_state
        return self.replay_buffer
