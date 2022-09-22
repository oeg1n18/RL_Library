import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense


class CreateActorNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=256, fc2_dims=256):
        super(CreateActorNetwork, self).__init__()
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.fc3 = Dense(n_actions, activation="softmax")

    def call(self, state):
        action_value = self.fc1(state)
        action_value = self.fc2(action_value)
        logits = self.fc3(action_value)
        return logits

class CreateCriticNetwork(keras.Model):
    def __init__(self, fc1_dims=256, fc2_dims=256):
        super(CreateCriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.fc3 = Dense(1, activation=None)

    def call(self, state):
        value = self.fc1(state)
        value = self.fc2(value)
        value = self.fc3(value)
        return value

