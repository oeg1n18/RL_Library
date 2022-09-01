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


