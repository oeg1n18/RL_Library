import random


class ReplayMemory:
    def __init__(self, data_spec, batch_size, max_length):
        self.data_spec = data_spec
        self.batch_size = batch_size
        self.max_length = max_length
        self.memory = []
        self.index = 0

    def add_trajectory(self, trajectory):
        assert self.data_spec == trajectory.data_spec(), "The trajectory type does not match replay buffer"
        if self.index < self.max_length:
            self.memory.append(trajectory)
            self.index += 1
        else:
            self.index = 0
            self.memory[self.index] = trajectory

    def sample_experience(self, use_priority=None):
        if use_priority:
            return random.sample(self.memory, self.batch_size)
        else:
            assert False, "Have not implemented priority sampling yet"

    def clear_experiences(self):
        self.memory = []
        self.index = 0
