import numpy as np
import tensorflow as tf
from tensorflow import keras
from replay_buffers.UniformReplayMemory import *
from copy import copy
import os


class DDQNAgent:
    def __init__(self, observation_space, action_space, qnet, learning_rate=0.01, df=0.99, epsilon_decay=0.005,
                 weight_update_freq=7):
        tf.keras.backend.set_floatx('float64')
        self.epsilon = 1.0
        self.learning_rate = learning_rate
        self.df = df
        self.epsilon_decay = epsilon_decay
        self.episode = 0
        self.qnet = qnet
        self.target_qnet = copy(qnet)
        self.observation_space = observation_space
        self.action_space = action_space
        self.n_train = 0
        self.weight_update_freq = weight_update_freq
        self.check_weights_model = copy(qnet)

    def collect_policy(self, trajectory):
        if trajectory.done:
            self.episode += 1
            self.epsilon = np.exp(-self.epsilon_decay * self.episode)
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_space.n)
        else:
            qvalues = self.qnet(np.expand_dims(trajectory.state, axis=0)).numpy()
            action = np.argmax(qvalues)
        return int(action)

    def policy(self, trajectory):
        qvalues = self.qnet(np.expand_dims(trajectory.state, axis=0)).numpy()
        action = self.argmax(qvalues)
        return int(action)

    def save(self, dir):
        if not os.path.isdir(dir):
            os.mkdir(dir)
        actor_dir = os.path.join(dir, "actor_net")
        target_dir = os.path.join(dir, "target_net")
        if not os.path.isdir(actor_dir):
            os.mkdir(actor_dir)
        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)
        self.qnet.save(actor_dir)
        self.target_qnet.save(target_dir)

    def load(self, dir):
        actor_dir = os.path.join(dir, "actor_net")
        target_dir = os.path.join(dir, "target_net")
        self.qnet = keras.models.load_model(actor_dir)
        self.target_qnet = keras.models.load_model(target_dir)

    def train(self, experiences):
        self.n_train += 1
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for trajectory in experiences:
            states.append(trajectory.state)
            actions.append(trajectory.action)
            rewards.append(trajectory.reward)
            dones.append(trajectory.done)
            next_states.append(trajectory.next_state)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        current_qvalues = self.qnet(states).numpy()
        next_qvalues = self.qnet(next_states).numpy()
        target_actions = self.argmax(next_qvalues)
        next_target_qvalues = self.target_qnet(next_states).numpy()
        target_qvalues = copy(current_qvalues)
        for t in range(current_qvalues.shape[0]):
            target_qvalues[t, actions[t]] = rewards[t] + (1 - dones[t])*self.df * next_target_qvalues[t, int(target_actions[t])]

        self.qnet.fit(states.astype(np.float32), target_qvalues.astype(np.float32), verbose=False)
        if self.n_train % self.weight_update_freq == 0:
            self.target_qnet.set_weights(self.qnet.get_weights())

    def argmax(self, values, n_actions=2):
        upper_values = np.argmax(values, axis=1)
        lower_values = np.argmin(values, axis=1)
        for index, value in enumerate(upper_values):
            if value == lower_values[index]:
                upper_values[index] = np.random.randint(n_actions)
        return upper_values

