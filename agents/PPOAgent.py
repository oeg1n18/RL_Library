import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import os
import numpy as np

from replay_buffers.UniformReplayMemory import PPOMemory
from replay_buffers.Utils import get_data_spec
from replay_buffers.Trajectory import Trajectory


from Networks.PPONet import CreateActorNetwork, CreateCriticNetwork

class PPOAgent:
    def __init__(self, env, df=0.99, lr=0.0003,
                 gae_lambda=0.95, policy_clip=0.2, batch_size=64,
                 n_epochs=10):
        self.df = df
        self.env = env
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = CreateActorNetwork(env.action_space.n)
        self.critic = CreateCriticNetwork()
        self.actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
        self.critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

        self.replay_buffer = PPOMemory(batch_size)

    def add_trajectory(self, state, action, probs, value, reward, done):
        self.replay_buffer.store_memory(state, action, probs, value, reward, done)

    def save(self, dir):
        actor_path = os.path.join(dir, "actor")
        critic_path = os.path.join(dir, "critic")
        self.actor.save(actor_path)
        self.critic.save(critic_path)

    def load(self, dir):
        actor_path = os.path.join(dir, "actor")
        critic_path = os.path.join(dir, "critic")
        self.actor = tf.keras.models.load_model(actor_path)
        self.critic = tf.keras.models.load_model(critic_path)

    def policy(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        probs = self.actor(state)
        dist = tfp.distributions.Categorical(probs)
        action = dist.sample()
        value = self.critic(state)
        action = action.numpy()[0]
        value = value.numpy()[0]
        return action, probs, value

    def train(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.replay_buffer.generate_batches()
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.df * values[k+1] * (1-int(dones_arr[k])) - values[k])
                    discount *= self.df * self.gae_lambda
                advantage[t] = a_t

            for batch in batches:
                with tf.GradientTape(persistent=True) as tape:
                    states = tf.convert_to_tensor(state_arr[batch])
                    old_probs = tf.convert_to_tensor(action_arr[batch])
                    actions = tf.convert_to_tensor(action_arr[batch])

                    probs = self.actor(states)
                    dist = tfp.distributions.Categorical(probs)
                    new_probs = dist.log_prob(actions)

                    critic_value = self.critic(states)
                    critic_value = tf.squeeze(critic_value, 1)
                    prob_ratio = tf.math.exp(new_probs - tf.cast(old_probs, tf.float32))
                    weighted_probs = advantage[batch] * prob_ratio
                    clipped_probs = tf.clip_by_value(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)
                    weighted_clipped_probs = clipped_probs * advantage[batch]
                    actor_loss = -tf.math.minimum(weighted_probs, weighted_clipped_probs)
                    actor_loss = tf.math.reduce_mean(actor_loss)
                    returns = advantage[batch] + values[batch]
                    critic_loss = keras.losses.MSE(critic_value, returns)
                actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
                critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
                self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
                self.critic.optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        self.replay_buffer.clear_memory()







