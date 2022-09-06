import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from Networks.DDPGNet import CreateCriticNetwork, CreateActorNetwork
from replay_buffers.Trajectory import Trajectory

class DDPGAgent:
    def __init__(self, input_dims, actor_lr=0.001, critic_lr=0.002,
                 env=None, df=0.99, n_actions=2, network_update_constant=0.005,
                 fc1=400, fc2=300, noise=0.1, tau=0.005):
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.env = env
        self.df = df
        self.tau = tau
        self.n_actions = n_actions
        self.network_update_constant = network_update_constant
        self.fc1 = fc1
        self.fc2 = fc2
        self.noise = noise
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]

        self.actor = CreateActorNetwork(n_actions=n_actions)
        self.target_actor = CreateActorNetwork(n_actions=n_actions)
        self.critic = CreateCriticNetwork(n_actions=n_actions)
        self.target_critic = CreateCriticNetwork(n_actions=n_actions)

        self.actor.compile(optimizer=Adam(learning_rate=actor_lr))
        self.critic.compile(optimizer=Adam(learning_rate=critic_lr))
        self.target_actor.compile(optimizer=Adam(learning_rate=actor_lr))
        self.target_critic.compile(optimizer=Adam(learning_rate=critic_lr))

        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight*tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)
        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight*tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)

    def policy(self, trajectory, evaluate=False):
        state = tf.convert_to_tensor([trajectory.state], dtype=tf.float32)
        actions = self.actor(state)
        if not evaluate:
            actions += tf.random.normal(shape=[self.n_actions], mean=0.0, stddev=self.noise)
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)
        return actions[0]


    def save(self, dir):
        if not os.path.isdir(dir):
            os.mkdir(dir)
        actor_path = os.path.join(dir, "actor.h5")
        target_actor_path = os.path.join(dir, "target_actor.h5")
        critic_path = os.path.join(dir, "critic.h5")
        target_critic_path = os.path.join(dir, "target_critic.h5")
        self.actor.save(actor_path)
        self.target_actor.save(target_actor_path)
        self.critic.save(critic_path)
        self.target_critic.save(target_critic_path)

    def load(self, dir):
        if not os.path.isdir(dir):
            os.mkdir(dir)
        actor_path = os.path.join(dir, "actor.h5")
        target_actor_path = os.path.join(dir, "target_actor.h5")
        critic_path = os.path.join(dir, "critic.h5")
        target_critic_path = os.path.join(dir, "target_critic.h5")
        self.actor = keras.models.load_model(actor_path)
        self.target_actor = keras.models.load_model(target_actor_path)
        self.critic = keras.models.load_model(critic_path)
        self.target_critic = keras.models.load_model(target_critic_path)

    def train(self, experience):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for trajectory in experience:
            states.append(trajectory.state)
            actions.append(trajectory.action)
            rewards.append(trajectory.reward)
            next_states.append(trajectory.next_state)
            dones.append(trajectory.done)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state)
            next_critic_value = tf.squeeze(self.target_critic(next_states, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            target = rewards + self.df * next_critic_value*(1-dones)
            critic_loss = keras.losses.MSE(target, critic_value)
        critic_network_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)
        actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))

        self.update_network_parameters()





