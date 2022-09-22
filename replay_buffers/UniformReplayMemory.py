import random
import numpy as np


class ReplayMemory:
    def __init__(self, data_spec, batch_size, max_length):
        self.data_spec = data_spec
        self.batch_size = batch_size
        self.max_length = max_length
        self.memory = []
        self.index = 0
        self.full_flag = False

    def add_trajectory(self, trajectory):
        if self.index < self.max_length:
            self.memory.append(trajectory)
            self.index += 1
        else:
            self.index = 0
            self.memory[self.index] = trajectory

    def sample_experience(self, N=1, use_priority=None):
        if use_priority:
            assert False, "Have not implemented priority sampling yet"
        else:
            if self.index > self.batch_size or self.full_flag:
                self.full_flag = True
                return random.sample(self.memory, self.batch_size)
            else:
                return random.sample(self.memory, self.index)

    def sample_all_episodes(self):
        return self.memory

    def clear_experiences(self):
        self.memory = []
        self.index = 0


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.values = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states), np.array(self.actions), np.array(self.probs), np.array(self.values), np.array(self.rewards), np.array(self.dones), batches

    def store_memory(self, state, action, probs, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []



