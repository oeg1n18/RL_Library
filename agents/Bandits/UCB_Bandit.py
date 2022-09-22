import numpy as np


class UCB_Bandit:
    def __init__(self, env):
        self.env = env
        self.reward_space = []
        for _ in range(env.action_space.n):
            self.reward_space.append([])
        self.t = 0

    def get_action(self):
        if self.t < self.env.action_space.n:
            action = self.t
        else:
            mean_rewards = []
            ucb = []
            for list in self.reward_space:
                mean_rewards.append(np.mean(list))
                ucb.append(np.sqrt((2*np.log(self.t))/len(list)))
            action = np.argmax(np.array(mean_rewards) + 3*np.array(ucb))
        self.t += 1
        return action

    def update(self, action, reward):
        self.reward_space[action].append(reward)


