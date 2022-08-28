from Networks.Qnet import CreateQNetwork
from replay_buffers.Trajectory import Trajectory
from agents.DQNAgent import DQNAgent
from replay_buffers.UniformReplayMemory import ReplayMemory
from drivers.driver import driver
from replay_buffers.Utils import get_data_spec
from metrics.metrics import metrics
import gym
import numpy as np

env = gym.make("CartPole-v1", new_step_api=True)

qnet = CreateQNetwork(env.observation_space, env.action_space, (10, 5))

data_spec = get_data_spec(env)

replay_memory = ReplayMemory(data_spec, 128, 50000)

agent = DQNAgent(env.observation_space, env.action_space, qnet)

metric = metrics(env, agent.policy)

for step in range(1000):
    agent.epsilon -= 0.01
    replay_memory = driver(env, agent.collect_policy, replay_memory, 16)
    experiences = replay_memory.sample_experience()
    agent.train(experiences)
    metrics.policy = agent.policy
    print(step, metric.AverageReturn())