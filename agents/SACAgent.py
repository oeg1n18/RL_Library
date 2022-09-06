import tensorflow as tf
from tensorflow import keras
from Networks.SACNet import CreateActorNetwork, CreateCriticNetwork, CreateValueNetwork
from replay_buffers.UniformReplayMemory import ReplayMemory
from replay_buffers.Utils import get_data_spec
from replay_buffers.Trajectory import Trajectory
import os
import gym

class SACAgent:
    def __init__(self, env, actor_lr=0.0003, critic_lr=0.0003, input_dims=[8], n_actions=1,
                 layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2, tau=0.05, df=0.99):
        self.env = env
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.input_dims = input_dims
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.batch_size = batch_size
        self.reward_scale = reward_scale
        self.n_actions = n_actions
        self.tau = tau
        self.df = df

        self.replay_buffer = ReplayMemory(get_data_spec(env), batch_size, 1000000)

        self.actor = CreateActorNetwork(n_actions=n_actions)
        self.critic_1 = CreateCriticNetwork(n_actions=n_actions)
        self.critic_2 = CreateCriticNetwork(n_actions=n_actions)
        self.value = CreateValueNetwork(n_actions=n_actions)
        self.target_value = CreateValueNetwork(n_actions=n_actions)
        self.actor.compile(optimizer=keras.optimizers.Adam(learning_rate=actor_lr))
        self.critic_1.compile(optimizer=keras.optimizers.Adam(learning_rate=critic_lr))
        self.critic_2.compile(optimizer=keras.optimizers.Adam(learning_rate=critic_lr))
        self.value.compile(optimizer=keras.optimizers.Adam(learning_rate=critic_lr))
        self.target_value.compile(optimizer=keras.optimizers.Adam(learning_rate=critic_lr))

        self.update_network_parameters(tau=1)


    def policy(self, traj):
        state = tf.convert_to_tensor([traj.state])
        action, _ = self.actor.sample_action(state)
        return action[0]

    def add_trajectory(self, traj):
        self.replay_buffer.add_trajectory(traj)

    def update_network_parameters(self, tau=None):
        if not tau:
            tau = self.tau
        new_target_weights = []
        weights = self.value.weights
        for i, weight in enumerate(weights):
            new_target_weights = tau*weight + (1-tau)*self.target_value.weights[i]
        self.target_value.set_weights(new_target_weights)

    def save(self, dir):
        actor_path = os.path.join(dir, "actor")
        critic1_path = os.path.join(dir, "critic_1")
        critic2_path = os.path.join(dir, "critic_2")
        value_path = os.path.join(dir, "value")
        target_value_path = os.path.join(dir, "target_value")
        self.actor.save(actor_path)
        self.critic_1.save(critic1_path)
        self.critic_2.save(critic2_path)
        self.value.save(value_path)
        self.target_value.save(target_value_path)

    def load(self, dir):
        actor_path = os.path.join(dir, "actor")
        critic1_path = os.path.join(dir, "critic_1")
        critic2_path = os.path.join(dir, "critic_2")
        value_path = os.path.join(dir, "value")
        target_value_path = os.path.join(dir, "target_value")
        self.actor = keras.models.load_model(actor_path)
        self.critic_1 = keras.models.load_model(critic1_path)
        self.critic_2 = keras.models.load_model(critic2_path)
        self.value = keras.models.load_model(value_path)
        self.target_value = keras.models.load_model(target_value_path)

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
            value = self.value(states)
            next_value = self.target_value(next_states)

            new_actions, log_probs = self.actor.sample_action(states)
            q1_new_policy = self.critic_1(states, new_actions)
            q2_new_policy = self.critic_2(states, new_actions)
            critic_value = tf.math.minimum(q1_new_policy, q2_new_policy)

            value_target = critic_value - log_probs
            value_loss = 0.5 * keras.losses.MSE(value, value_target)

            actor_loss = log_probs - critic_value
            actor_loss = tf.reduce_mean(actor_loss)

            q_hat = self.reward_scale * rewards + self.df * next_value * (1-dones)
            q1_old_policy = self.critic_1(states, actions)
            q2_old_policy = self.critic_2(states, actions)
            critic_1_loss = 0.5 * keras.losses.MSE(q1_old_policy, q_hat)
            critic_2_loss = 0.5 * keras.losses.MSE(q2_old_policy, q_hat)

            critic_loss = critic_1_loss + critic_2_loss
        critic_1_grads = tape.gradient(critic_loss, self.critic_1.trainable_variables)
        critic_2_grads = tape.gradient(critic_loss, self.critic_2.trainable_variables)
        value_grads = tape.gradient(value_loss, self.value.trainable_weights)
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_weights)
        self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_weights))
        self.value.optimizer.apply_gradients(zip(value_grads, self.value.trainable_weights))
        self.critic_1.optimizer.apply_gradients(zip(critic_1_grads, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(zip(critic_2_grads, self.critic_2.trainable_variables))

