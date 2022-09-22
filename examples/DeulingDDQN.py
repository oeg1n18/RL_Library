import gym
import numpy as np
from agents.DuelingDDQNAgent import DuelingDDQNAgent
from drivers.openai_drivers import driver
from observers.metrics import AverageReturnObserver, AverageEpisodeLengthObserver


env = gym.make('CartPole-v1')

agent = DuelingDDQNAgent(env, lr=0.001, df=0.99,
                         epsilon=1.0, batch_size=64, epsilon_dec=1/1800, replace=25)


observers=[AverageReturnObserver(), AverageEpisodeLengthObserver()]

for episode in range(2000):
    agent.replay_buffer, observers = driver(env, agent.policy, agent.replay_buffer, 1, observers=observers)
    experience = agent.replay_buffer.sample_experience()
    agent.train(experience)
    print("Step: ", episode, " epsilon: ", agent.epsilon, " AverageReturn: ", observers[0].result())
