import numpy as np
import tensorflow_probability as tfp
from scipy.stats import norm
import matplotlib.pyplot as plt

class GaussianThompson:
    def __init__(self, action, mu_0, std_0, posterior_std=1.0):
        self.t_0 = 1/(std_0**2)
        self.mu_0 = mu_0
        self.action = action
        self.posterior_std = posterior_std
        self.all_rewards = []


    def sample_priori(self):
        std = np.sqrt(1/self.t_0)
        dist = tfp.distributions.Normal(self.mu_0, std)
        return dist.sample().numpy()

    def update(self, reward):
        self.all_rewards.append(reward)
        n = len(self.all_rewards)
        mu_num = self.t_0*self.mu_0 + self.posterior_std * np.sum(self.all_rewards)
        mu_den = self.t_0 + n * self.posterior_std
        self.mu_0 = mu_num/mu_den
        self.t_0 = self.t_0 + n * self.posterior_std




class ThompsonBandit(GaussianThompson):
    def __init__(self, env, priori_mu=0.0001, priori_std=100, posterior_std=2.5):
        self.env = env
        self.reward_dists = [GaussianThompson(action, priori_mu, priori_std, posterior_std=posterior_std) for action in range(env.action_space.n)]

    def get_action(self):
        samples = [self.reward_dists[i].sample_priori() for i in range(self.env.action_space.n)]
        return np.argmax(samples)

    def update_priori(self, action, reward):
        self.reward_dists[action].update(reward)

    def plot_best_distributions(self, num_dists=5):
        mask = []
        for dist in self.reward_dists:
            if len(dist.all_rewards) > 0:
                mask.append(True)
            else:
                mask.append(False)

        samples = [self.reward_dists[i].sample_priori() for i in range(self.env.action_space.n)]
        ordered_samples = np.argsort(samples)
        idxs = []
        i = len(ordered_samples)-1
        while len(idxs) < num_dists:
            if len(self.reward_dists[ordered_samples[i]].all_rewards) > 0:
                idxs.append(ordered_samples[i])
            i -= 1

        lowest_mean = np.inf
        highest_mean = 0
        for i in idxs:
            if self.reward_dists[i].mu_0 > highest_mean:
                highest_mean = self.reward_dists[i].mu_0
            if self.reward_dists[i].mu_0 < lowest_mean:
                lowest_mean = self.reward_dists[i].mu_0
        x_low = lowest_mean-15
        x_high = highest_mean + 15
        x = np.linspace(x_low, x_high, 1000)
        dists = [self.reward_dists[i] for i in idxs]
        for i, dist in enumerate(dists):
            label = "Action Distribution " + str(idxs[i])
            plt.fill_between(x, norm.pdf(x, dist.mu_0, 1/dist.t_0), np.zeros(x.size)-0.02, alpha=0.8, label=label)
        plt.xlabel("Reward")
        plt.ylabel("Probability")
        plt.title("Action distribution")
        plt.legend()
        plt.show()


