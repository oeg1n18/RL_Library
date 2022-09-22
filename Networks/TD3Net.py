import numpy as np
import tensorflow as tf


class CriticNetwork(tf.keras.Model):
    def __init__(self, fc1_dims, fc2_dims, n_actions):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = tf.keras.layers.Dense(fc1_dims, activation="relu")
        self.fc2 = tf.keras.layers.Dense(fc2_dims, activation="relu")
        self.q = tf.keras.layer.Dense(1, activation=None)

    def call(self, state, action):
        action_value = self.fc1(tf.concat([state, action], axis=1))
        action_value = self.fc2(action_value)
        q = self.q(action_value)
        return q

class ActorNetwork(tf.keras.Model):
    def __init__(self, fc1_dims, fc2_dims, n_actions):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = tf.keras.layers.Dense(fc1_dims, activation="relu")
        self.fc2 = tf.keras.layers.Dense(fc2_dims, activation="relu")
        self.mean = tf.keras.layers.Dense(self.n_actions, activation="tanh")

    def call(self, state):
        mean = self.fc1(state)
        mean = self.fc2(mean)
        mean = self.mean(mean)
        return mean





