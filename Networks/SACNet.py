
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp

class CreateActorNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=256, fc2_dims=256):
        super(CreateActorNetwork, self).__init__()
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.fc1 = tf.keras.layers.Dense(fc1_dims, activation="relu")
        self.fc2 = tf.keras.layers.Dense(fc2_dims, activation="relu")
        self.mean = tf.keras.layers.Dense(n_actions, activation=None)
        self.stddev = tf.keras.layers.Dense(n_actions, activation="sigmoid")

    def call(self, state):
        probs = self.fc1(state)
        probs = self.fc2(probs)
        mean = self.mean(probs)
        stddev = self.mean(probs)
        return mean, stddev

    def sample_action(self, state):
        mean, stddev = self.call(state)
        dist = tfp.distributions.Normal(mean, stddev)
        action = dist.sample()
        return action, dist.log_prob(action)



class CreateCriticNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=256, fc2_dims=256):
        super(CreateCriticNetwork, self).__init__()
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.fc1 = tf.keras.layers.Dense(fc1_dims, activation="relu")
        self.fc2 = tf.keras.layers.Dense(fc2_dims, activation="relu")
        self.v = tf.keras.layers.Dense(1, activation=None)

    def call(self, state, action):
        state_values = self.fc1(tf.concat([state, action], 1))
        state_values = self.fc2(state_values)
        value = self.v(state_values)
        return value


class CreateValueNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=256, fc2_dims=256):
        super(CreateValueNetwork, self).__init__()
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.fc1 = tf.keras.layers.Dense(fc1_dims, activation="relu")
        self.fc2 = tf.keras.layers.Dense(fc2_dims, activation="relu")
        self.v = tf.keras.layers.Dense(1, activation=None)

    def call(self, state):
        state_values = self.fc1(state)
        state_values = self.fc2(state_values)
        value = self.v(state_values)
        return value







