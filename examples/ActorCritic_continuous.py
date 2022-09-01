from agents.ActorCriticAgent import ActorCriticAgent
from drivers.value_driver import ac_driver
from Networks.ActorCriticNet import ActorCriticNetwork
from observers.metrics import AverageReturnObserver

import warnings
import numpy as np
import gym


warnings.filterwarnings("ignore")
np.random.seed(10)


env = gym.make("Pendulum-v1")

actor_critic = ActorCriticNetwork()

agent = ActorCriticAgent(env, actor_critic, learning_rate=0.001)

observers = [AverageReturnObserver()]

for training_step in range(200):
    agent, observers = ac_driver(env, agent, 1, observers=observers)
    print("Training Step: ", training_step, " Average Return: ", observers[0].result())
