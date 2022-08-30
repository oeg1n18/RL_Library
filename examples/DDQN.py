from Networks.QNet import CreateQNetwork
from replay_buffers.Trajectory import Trajectory
from agents.DDQNAgent import DDQNAgent
from replay_buffers.UniformReplayMemory import ReplayMemory
from drivers.value_driver import driver
from replay_buffers.Utils import get_data_spec
from observers.metrics import AverageReturnObserver
from observers.metrics import AverageEpisodeLengthObserver
import gym
import numpy as np
import copy

env = gym.make("CartPole-v1", new_step_api=True)

qnet = CreateQNetwork(env.observation_space, env.action_space, (10, 5))

data_spec = get_data_spec(env)

replay_memory = ReplayMemory(data_spec, 256, 1000)

agent = DDQNAgent(env.observation_space, env.action_space, qnet)


def random_policy(traj):
    return np.random.randint(2)


replay_memory = driver(env, random_policy, replay_memory, 500)


all_observers = [AverageReturnObserver(), AverageEpisodeLengthObserver()]
av_return = 0
metric_freq = 100
for step in range(1000):
    agent.epsilon -= 1 / 1000
    replay_memory, all_observers = driver(env, agent.collect_policy, replay_memory, 1, observers=all_observers)
    experiences = replay_memory.sample_experience()
    agent.train(experiences)
    print("Average Return: ", all_observers[0].result(), " Average Steps: ", all_observers[1].result(), " Epsilon: ", agent.epsilon)


agent.save("saved_agents/DDQN")