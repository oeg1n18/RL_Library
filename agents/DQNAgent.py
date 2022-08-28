import numpy as np
import tensorflow as tf
from tensorflow import keras
from replay_buffers.UniformReplayMemory import *
from copy import copy


class DQNAgent:
    def __init__(self, observation_space, action_space, qnet, learning_rate=0.01, df=0.99, epsilon_decay=0.005):
        self.epsilon = 1.0
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
            self.epsilon = np.exp(-self.epsilon_decay * self.episode)
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_space.n)
        else:
            qvalues = self.qnet(np.expand_dims(trajectory.state, axis=0))
            action = np.argmax(qvalues)
        return int(action)

    def policy(self, trajectory):
        qvalues = self.qnet(np.expand_dims(trajectory.state, axis=0))
        print(qvalues)
        action = np.argmax(qvalues)
        return int(action)

    def train(self, experiences):
        states = []
        actions = []
        rewards = []
        next_states = []
        for trajectory in experiences:
            states.append(trajectory.state)
            actions.append(trajectory.action)
            rewards.append(trajectory.reward)
            next_states.append(trajectory.next_state)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        current_qvalues = self.qnet(states).numpy()
        next_qvalues = self.qnet(next_states).numpy()
        target_qvalues = copy(current_qvalues)
        for t in range(current_qvalues.shape[0]):
            target_qvalues[t, actions[t]] = rewards[t] + self.df * np.max(next_qvalues[t, :])
        self.qnet.fit(states, target_qvalues, verbose=False)
