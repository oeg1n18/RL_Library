from copy import copy

import numpy as np
import tensorflow as tf
from tensorflow import keras


class Reinforce_base:
    def __init__(self, discount_factor):
        self.df = discount_factor

    def prepare_returns(self, experiences):
        discounted_returns = np.zeros(len(experiences))
        for index, trajectory in enumerate(experiences):
            discounted_return = 0
            for future_index_step, future_trajectory in enumerate(experiences[index:]):
                discounted_return += (self.df ** future_index_step) * future_trajectory.reward
                if future_trajectory.done:
                    discounted_returns[index] = discounted_return
                    break
        mean = discounted_returns.mean()
        std = discounted_returns.std()
        final_returns = [((G - mean) / std) for G in discounted_returns]
        return list(final_returns)


class ReinforceAgent(Reinforce_base):
    def __init__(self, policynet, learning_rate=0.05, discount_factor=0.95):
        super().__init__(discount_factor)
        self.policynet = policynet
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.optimizer = keras.optimizers.Nadam(learning_rate=learning_rate)

    def policy(self, trajectory):
        logits = self.policynet(np.expand_dims(trajectory.state, axis=0)).numpy()
        return int(np.argmax(logits))

    def train(self, experiences):
        all_returns = self.prepare_returns(experiences)
        all_grads = []
        for trajectory in experiences:
            with tf.GradientTape() as tape:
                tape.watch(self.policynet.trainable_variables)
                logits = self.policynet(np.expand_dims(trajectory.state, axis=0))
                target_logits = np.zeros(logits.shape)
                target_logits[0, trajectory.action] = 1.0
                target_logits = tf.convert_to_tensor(target_logits)
                loss = self.loss_fn(target_logits, logits)
            all_grads.append(tape.gradient(loss, self.policynet.trainable_variables))

        all_mean_grads = []
        for var_index in range(len(self.policynet.trainable_variables)):
            mean_grads = tf.reduce_mean([final_return * all_grads[step][var_index]
                                         for step, final_return in enumerate(all_returns)], axis=0)
            all_mean_grads.append(mean_grads)
        self.optimizer.apply_gradients(zip(all_mean_grads, self.policynet.trainable_variables))

    def save(self, dir):
        self.policynet.save(dir)

    def load(self, dir):
        self.policynet = keras.models.load_model(dir)
