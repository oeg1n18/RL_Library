import tensorflow as tf
from tensorflow import keras



class CreateDuelingDDQNNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims, fc2_dims):
        super(CreateDuelingDDQNNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(fc1_dims, activation="relu")
        self.dense2 = tf.keras.layers.Dense(fc2_dims, activation="relu")
        self.V = tf.keras.layers.Dense(1, activation=None)
        self.A = tf.keras.layers.Dense(n_actions, activation=None)


    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        V = self.V(x)
        A = self.A(x)

        Q = V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True))

        return Q

    def advantage(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        A = self.A(x)
        return A