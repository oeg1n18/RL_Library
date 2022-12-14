from copy import copy

import numpy as np
import tensorflow as tf
from tensorflow import keras
import os


class Reinforce_base:
    def __init__(self, discount_factor):
        self.df = discount_factor

    def prepare_returns(self, experiences):
        discounted_returns = np.zeros(len(experiences))
        for index, trajectory in enumerate(experiences):
            future_df = self.df
            discounted_return = 0

            for future_index_step, future_trajectory in enumerate(experiences[index:]):
                discounted_return += self.df ** future_index_step * future_trajectory.reward
                if future_trajectory.done:
                    discounted_returns[index] = discounted_return
                    break
        return discounted_returns


class ReinforceBaselineAgent(Reinforce_base):
    def __init__(self, policynet, value_network, learning_rate=0.001, discount_factor=0.95):
        super().__init__(discount_factor)
        self.policynet = policynet
        self.value_network = value_network
        self.policynet.compile(optimizer=keras.optimizers.Nadam(learning_rate=learning_rate))
        self.value_network.compile(loss='mse', optimizer=keras.optimizers.Nadam(learning_rate=learning_rate))
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()

    def policy(self, trajectory):
        logits = self.policynet(np.expand_dims(trajectory.state, axis=0)).numpy()
        return int(np.argmax(logits))

    def train(self, experiences):
        all_returns = self.prepare_returns(experiences)
        all_states = []
        all_grads = []
        for trajectory in experiences:
            all_states.append(trajectory.state)
            with tf.GradientTape() as tape:
                tape.watch(self.policynet.trainable_variables)
                logits = self.policynet(np.expand_dims(trajectory.state, axis=0))
                target_logits = np.zeros(logits.shape)
                target_logits[0, trajectory.action] = 1.0
                target_logits = tf.convert_to_tensor(target_logits, dtype=tf.float32)
                loss = self.loss_fn(target_logits, tf.cast(logits, tf.float32))
            all_grads.append(tape.gradient(loss, self.policynet.trainable_variables))

        all_values = self.value_network(np.array(all_states)).numpy()[0]
        baseline_error = all_returns - all_values
        all_mean_grads = []
        for var_index in range(len(self.policynet.trainable_variables)):
            mean_grads = tf.reduce_mean([error * all_grads[step][var_index]
                                         for step, error in enumerate(baseline_error)], axis=0)
            all_mean_grads.append(mean_grads)
        self.value_network.fit(np.array(all_states), all_returns, verbose=False)
        self.policynet.optimizer.apply_gradients(zip(all_mean_grads, self.policynet.trainable_variables))

    def save(self, dir):
        if not os.path.isdir(dir):
            os.mkdir(dir)
        actor_dir = os.path.join(dir, "policy_net")
        baseline_dir = os.path.join(dir, "baseline_net")
        if not os.path.isdir(actor_dir):
            os.mkdir(actor_dir)
        if not os.path.isdir(baseline_dir):
            os.mkdir(baseline_dir)
        self.policynet.save(actor_dir)
        self.value_network.save(baseline_dir)

    def load(self, dir):
        actor_dir = os.path.join(dir, "policy_net")
        baseline_dir = os.path.join(dir, "baseline_net")
        self.policy = keras.models.load_model(actor_dir)
        self.value_network = keras.models.load_model(baseline_dir)

