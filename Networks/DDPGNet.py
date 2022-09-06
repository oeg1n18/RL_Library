import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense


class CreateCriticNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=512, fc2_dims=512):
        super(CreateCriticNetwork, self).__init__()
        self.n_actions = n_actions
        self.fc1_dims = 512
        self.fc2_dims = 512

        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        action_value = self.fc1(tf.concat([state, action], 1))
        action_value = self.fc2(action_value)
        q_value = self.q(action_value)
        return q_value

class CreateActorNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=512, fc2_dims=512):
        super(CreateActorNetwork, self).__init__()
        self.n_actions = n_actions
        self.fc1_dims = 512
        self.fc2_dims = 512

        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.mu = Dense(1, activation='tanh')

    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)
        mu = self.mu(prob)
        return mu


