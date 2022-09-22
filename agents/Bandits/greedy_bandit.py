import numpy as np


class greedy_bandit:
    def __init__(self, env, epsilon_decay = 1/1000):
        self.env = env
        self.reward_space = []
        for _ in range(env.action_space.n):
            self.reward_space.append([0.0])
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay

    def get_action(self):
        self.epsilon -= self.epsilon_decay
        if np.random.rand() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            mean_rewards = []
            for list in self.reward_space:
                if list:
                    mean_rewards.append(np.mean(list))
                else:
                    mean_rewards.append(0.0)
            action = np.argmax(mean_rewards)
        return action

    def update(self, action, reward):
        self.reward_space[action].append(reward)


