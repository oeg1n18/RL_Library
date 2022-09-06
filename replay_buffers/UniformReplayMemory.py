import random


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
