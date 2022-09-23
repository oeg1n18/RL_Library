
from Networks.QNet import CreateQNetwork
from replay_buffers.Trajectory import Trajectory
from agents.SACAgent import SACAgent
from replay_buffers.UniformReplayMemory import ReplayMemory
from drivers.openai_drivers import driver
from replay_buffers.Utils import get_data_spec
from observers.metrics import AverageReturnObserver
from observers.metrics import AverageEpisodeLengthObserver
import gym
import pybullet
import pybullet_envs
import numpy as np
import time



env = gym.make("InvertedPendulumBulletEnv-v0")

agent = SACAgent(env, n_actions=2, actor_lr=0.01, critic_lr=0.02)

all_observers = [AverageReturnObserver(), AverageEpisodeLengthObserver()]

for step in range(2000):
    agent.replay_buffer, all_observers = driver(env, agent.policy, agent.replay_buffer, 1, observers=all_observers)
    experiences = agent.replay_buffer.sample_experience()
    agent.train(experiences)
    print("Average Return: ", all_observers[0].result(), " Average Steps: ", all_observers[1].result())


agent.save("saved_agents/DDQN")
agent.load("saved_agents/DDQN")
