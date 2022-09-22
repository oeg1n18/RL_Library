import tensorflow as tf
import numpy as np
import os
from keras.optimizers import Adam

from replay_buffers.UniformReplayMemory import ReplayMemory
from replay_buffers.Utils import get_data_spec
from Networks.DuelingDDQNNet import CreateDuelingDDQNNetwork
from replay_buffers.Trajectory import Trajectory


class DuelingDDQNAgent():
    def __init__(self, env, lr, df, epsilon, batch_size, epsilon_dec=1e-3, eps_end=0.01, mem_size=1000000, fc1_dims=128, fc2_dims=128, replace=100):
        self.action_space = env.action_space
        self.df = df
        self.epsilon = epsilon
        self.eps_decay = epsilon_dec
        self.eps_min = eps_end
        self.replace = replace
        self.batch_size = batch_size
        self.env = env
        self.train_step_counter = 0

        self.learn_step_counter = 0
        self.replay_buffer = ReplayMemory(get_data_spec(env), batch_size, mem_size)
        self.q_eval = CreateDuelingDDQNNetwork(env.action_space.n, fc1_dims, fc2_dims)
        self.q_next = CreateDuelingDDQNNetwork(env.action_space.n, fc1_dims, fc2_dims)

        self.q_eval.compile(optimizer=Adam(learning_rate=lr), loss="mean_squared_error")
        self.q_next.compile(optimizer=Adam(learning_rate=lr), loss="mean_squared_error")

    def store_trajectory(self, state, action, next_state, done):
        traj = Trajectory(state, action, next_state, done)
        self.replay_buffer.add_trajectory(traj)


    def replace_target_network(self):
        if self.train_step_counter % self.replace == 0:
            self.q_next.set_weights(self.q_eval.weights)

    def policy(self, traj):
        state = tf.convert_to_tensor([traj.state], dtype=tf.float32)
        if np.random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            state = np.array([state])
            actions = tf.squeeze(self.q_eval.advantage(state)).numpy()
            action = tf.math.argmax(actions).numpy()
        return action

    def greedy_policy(self, state):
        state = np.array([state])
        actions = self.q_eval.advantage(state)
        action = tf.math.argmax(actions, axis=1).numpy()[0]
        return action

    def save(self, dir):
        q_eval_path = os.path.join(dir, "q_eval")
        q_next_path = os.path.join(dir, "q_next")
        self.q_eval.save(q_eval_path)
        self.q_next.save(q_next_path)

    def load(self, dir):
        q_eval_path = os.path.join(dir, "q_eval")
        q_next_path = os.path.join(dir, "q_next")
        self.q_eval = tf.keras.models.load_model(q_eval_path)
        self.q_next = tf.keras.models.load_model(q_next_path)


    def train(self, experiences):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for traj in experiences:
            states.append(traj.state)
            actions.append(traj.action)
            rewards.append(traj.reward)
            next_states.append(traj.next_state)
            dones.append(traj.done)
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        q_pred = self.q_eval(tf.convert_to_tensor(states))
        q_next = self.q_next(tf.convert_to_tensor(next_states))

        q_target = q_pred.numpy()
        max_actions = tf.math.argmax(self.q_eval(tf.convert_to_tensor(next_states)), axis=1)
        for i, terminal in enumerate(dones):
            q_target[i, int(actions[i])] = rewards[i] + self.df * q_next[i, int(max_actions[i])]*(1.0-terminal)
        self.q_eval.train_on_batch(states, q_target)

        self.epsilon = self.epsilon - self.eps_decay if self.epsilon > self.eps_min else self.eps_min
        self.train_step_counter += 1
        self.replace_target_network()