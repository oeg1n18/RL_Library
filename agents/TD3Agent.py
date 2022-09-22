
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.optimizers import Adam

from replay_buffers.UniformReplayMemory import ReplayMemory
from replay_buffers.Trajectory import Trajectory
from replay_buffers.Utils import get_data_spec
from Networks.TD3Net import ActorNetwork, CriticNetwork



class Agent:
    def __init__(self, actor_lr, critic_lr, tau, env, df=0.99,
                 update_actor_interval=2, warmup=1000, n_actions=2,
                 max_size=1000000, layer1_size=400, layer2_size=300,
                 batch_size=300, noise=0.1):
        self.df = df
        self.tau = tau
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        self.batch_size = batch_size
        self.time_step = 0
        self.train_step_counter = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval
        self.noise = noise

        self.replay_buffer = ReplayMemory(get_data_spec(env), batch_size, max_size)

        self.actor = ActorNetwork(layer1_size, layer2_size, n_actions=n_actions)
        self.target_actor = ActorNetwork(layer1_size, layer2_size, n_actions=n_actions)
        self.critic_1 = CriticNetwork(layer1_size, layer2_size)
        self.critic_2 = CriticNetwork(layer1_size, layer2_size)
        self.target_critic_1 = CriticNetwork(layer1_size, layer2_size)
        self.target_critic_2 = CriticNetwork(layer1_size, layer2_size)

        self.actor.compile(optimizer=Adam(learning_rate=actor_lr), loss="mean_squared_error")
        self.target_actor.compile(optimizer=Adam(learning_rate=actor_lr), loss="mean_squared_error")
        self.critic_1.compile(optimizer=Adam(learning_rate=critic_lr), loss="mean_squared_error")
        self.critic_2.compile(optimizer=Adam(learning_rate=critic_lr), loss="mean_squared_error")
        self.target_critic_1.compile(optimizer=Adam(learning_rate=critic_lr), loss="mean_squared_error")
        self.target_critic_2.compile(optimizer=Adam(learning_rate=critic_lr), loss="mean_squared_error")


    def policy(self, traj):
        if self.time_step < self.warmup:
            mu = np.random.normal(scale=self.noise, size=(self.n_actions))
        else:
            state = tf.convert_to_tensor([traj.state], dtype=tf.float32)
            mean = self.actor(state)[0]
        mean_prime = mu + np.random.normal(scale=self.noise)

        mean_prime = tf.clip_by_value(mean_prime, self.min_actions, self.max_action)

        self.time_step += 1

        return mean_prime

    def add_trajectory(self, state, action, reward, next_state, done):
        traj = Trajectory(state, action, reward, next_state, done)
        self.replay_buffer.add_trajectory(traj)

    def train(self, experience):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for traj in experience:
            states.append(traj.state)
            actions.append(traj.action)
            rewards.append(traj.reward)
            next_states.append(traj.next_state)
            dones.append(traj.done)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones)
        dones = tf.cast(dones, tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            target_actions = self.target_actor(next_states)
            target_actions = target_actions + tf.clip_by_value(np.random.normal(scale=0.2), -0.5, 0.5)

            q1_next = self.target_critic_1(next_states, target_actions)
            q2_next = self.target_critic_2(next_states, target_actions)

            q1_next = tf.squeeze(q1_next, 1)
            q2_next = tf.squeeze(q2_next, 1)

            q1 = tf.squeeze(self.critic_1(states, actions), 1)
            q2 = tf.squeeze(self.critic_2(states, actions), 1)

            critic_value_next = tf.math.minimum(q1_next, q2_next)

            target = rewards + self.df * critic_value_next * (1-dones)

            critic_1_loss = keras.losses.MSE(target, q1)
            critic_2_loss = keras.losses.MSE(target, q2)

        critic_1_gradient = tf.gradient(critic_1_loss, self.critic_1.trainable_variables)
        critic_2_gradient = tf.gradient(critic_2_loss, self.critic_2.trainable_variables)
        self.critic_1.optimizer.apply_gradients(zip(critic_1_gradient, self.critic_1.trainables_variables))
        self.critic_2.optimizer.apply_gradients(zip(critic_2_gradient, self.critic_2.trainable_variables))

        self.train_step_counter += 1

        with tf.GradientTape() as tape:
            new_actions = self.actor(states)
            critic_1_value = self.critic_1(states, new_actions)
            actor_loss = -tf.math.reduce_mean(critic_1_value)
        actor_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_gradient, self.actor.trainable_variables))

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if not tau:
            tau = self.tau

        new_target_weights = []
        weights = self.critic_1.weights
        for i, weight in enumerate(weights):
            new_target_weights.append(tau*weight + self.target_critic_1.weights[i]*(1-tau))
        self.target_critic_1.set_weights(new_target_weights)

        new_target_weights = []
        weights = self.critic_2.weights
        for i, weight in enumerate(weights):
            new_target_weights.append(tau * weight + self.target_critic_2.weights[i] * (1 - tau))
        self.target_critic_2.set_weights(new_target_weights)

        new_target_weights = []
        weights = self.actor.weights
        for i, weight in enumerate(weights):
            new_target_weights.append(tau * weight + self.target_actor.weights[i] * (1 - tau))
        self.target_actor.set_weights(new_target_weights)









