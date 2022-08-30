from drivers.value_driver import driver
from replay_buffers.UniformReplayMemory import ReplayMemory
from agents.Reinforce import ReinforceAgent
from Networks.ReinforceNet import CreateReinfoceNetwork
from observers.metrics import AverageReturnObserver

import gym
import numpy as np

from replay_buffers.Utils import get_data_spec


def policy(experience):
    return np.random.randint(2)


env = gym.make("CartPole-v1", new_step_api=True)

data_spec = get_data_spec(env)

replay_buffer = ReplayMemory(data_spec, 32, 50000)

policy_network = CreateReinfoceNetwork(env.observation_space, env.action_space, (5,))

agent = ReinforceAgent(policy_network)

replay_buffer = driver(env, policy, replay_buffer, 100)

observers = [AverageReturnObserver()]

for training_step in range(200):
    replay_buffer.clear_experiences()
    replay_buffer, observers = driver(env, policy, replay_buffer, 16, observers=observers)
    experiences = replay_buffer.sample_all_episodes()
    agent.train(experiences)
    print("Training Step: ", training_step, " Average Return: ", observers[0].result())

