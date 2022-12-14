from drivers.openai_drivers import driver
from replay_buffers.UniformReplayMemory import ReplayMemory
from agents.ReinforceAgent import ReinforceAgent
from Networks.ReinforceNet import CreateReinfoceNetwork
from observers.metrics import AverageReturnObserver
import random

random.seed(10)
import gym
import numpy as np

from replay_buffers.Utils import get_data_spec


def policy(experience):
    return np.random.randint(2)


env = gym.make("CartPole-v1")

data_spec = get_data_spec(env)

replay_buffer = ReplayMemory(data_spec, 32, 50000)

policy_network = CreateReinfoceNetwork(env.observation_space, env.action_space, (50,))

agent = ReinforceAgent(policy_network, learning_rate=0.01)

replay_buffer = driver(env, policy, replay_buffer, 100)

observers = [AverageReturnObserver()]


for training_step in range(1000):
    replay_buffer.clear_experiences()
    replay_buffer, observers = driver(env, agent.policy, replay_buffer, 5, observers=observers)
    experiences = replay_buffer.sample_all_episodes()
    agent.train(experiences)
    print("Training Step: ", training_step, " Average Return: ", observers[0].result())

agent.save("saved_agents/REINFORCE")
agent.load("saved_agents/REINFORCE")