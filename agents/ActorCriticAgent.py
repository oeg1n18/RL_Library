from copy import copy

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import os


class ActorCriticBase:
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

class ActorCriticAgent(ActorCriticBase):
    def __init__(self, actor_critic, learning_rate=0.001, discount_factor=0.99):
        super().__init__(discount_factor)
        self.actor_critic = actor_critic
        self.actor_loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.critic_loss_fn = tf.keras.losses.Huber()
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.actor_critic.compile(optimizer=self.optimizer)

    def policy(self, trajectory):
        probs, _ = self.actor_critic(tf.convert_to_tensor([trajectory.state]))
        action_probs = tfp.distributions.Categorical(probs=probs)
        action = action_probs.sample()
        return action.numpy()[0]


    def train_step(self, trajectory):
        state = tf.convert_to_tensor([trajectory.state], dtype=tf.float32)
        next_state = tf.convert_to_tensor([trajectory.next_state], dtype=tf.float32)
        reward = tf.convert_to_tensor(trajectory.reward, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.actor_critic.trainable_variables)
            logits, value = self.actor_critic(state)
            _, next_value = self.actor_critic(next_state)
            state_value = tf.squeeze(value)
            next_state_value = tf.squeeze(next_value)
            action_probs = tfp.distributions.Categorical(probs=logits)
            log_prob = action_probs.log_prob(trajectory.action)

            td_error = reward + self.df * next_state_value * (1-int(trajectory.done)) - state_value
            actor_loss = -log_prob * td_error
            actor_loss = tf.squeeze(actor_loss)
            critic_loss = td_error**2

            total_loss = actor_loss + critic_loss
        grads = tape.gradient(total_loss, self.actor_critic.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.actor_critic.trainable_variables))

    def save(self, dir):
        path = os.path.join(dir, "actor_critic")
        self.actor_critic.save(path)

    def load(self, dir):
        path = os.path.join(dir, "actor_critic")
        self.actor_critic = keras.models.load_model(path)

