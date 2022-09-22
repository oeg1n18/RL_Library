import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense


class ActorCriticNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=1024, fc2_dims=512):
        super(ActorCriticNetwork, self).__init__()
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.fc1 = Dense(self.fc1_dims, activation="relu")
        self.fc2 = Dense(self.fc2_dims, activation="relu")
        self.value = Dense(1, activation=None)
        self.policy = Dense(n_actions, activation="softmax")

    def call(self, state):
        value = self.fc1(state)
        value = self.fc2(value)
        state_value = self.value(value)
        action_probs = self.policy(value)
        return action_probs, state_value


class CriticNetwork(tf.keras.Model):
    def __init__(self, fc1_dims, fc2_dims):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.fc1 = tf.keras.layers.Dense(fc1_dims, activation="relu")
        self.fc2 = tf.keras.layers.Dense(fc2_dims, activation="relu")
        self.q = tf.keras.layers.Dense(1, activation=None)

    def call(self, state):
        action_value = self.fc1(state)
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
        self.mean = tf.keras.layers.Dense(self.n_actions, activation="softmax")

    def call(self, state):
        probs = self.fc1(state)
        probs = self.fc2(probs)
        probs = self.mean(probs)
        return probs


