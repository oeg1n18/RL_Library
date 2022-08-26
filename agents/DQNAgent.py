import numpy as np
import tensorflow as tf
from tensorflow import keras
from replay_buffers.UniformReplayMemory import *

class DQNAgent:
    def __init__(self, observation_space, action_space, qnet, learning_rate=0.01, df=0.99, epsilon_decay=0.005):
        self.epsilon = None
        self.learning_rate = learning_rate
        self.df = df
        self.epsilon_decay = epsilon_decay
        self.episode = 0
        self.qnet = qnet
        self.observation_space = observation_space
        self.action_space = action_space

    def collect_policy(self, trajectory):
        if trajectory.done:
            self.episode += 1
            self.epsilon = np.exp(-self.epsilon_decay*self.episode)
        if np.rnadon.rand() < self.epsilon:
            try:
                action = np.random.randint(self.action_space.n)
            except:
                print("Action space must have attribute action_space.n")
        else:
            qvalues = self.Qnet(np.expand_dims(trajectory.state, axis=0))
            action = np.argmax(qvalues)
        return int(qvalues)

    def policy(self, trajectory):
        qvalues = self.Qnet(np.expand_dims(trajectory.state, axis=0))[0]
        action = np.argmax(qvalues)
        return int(qvalues)






