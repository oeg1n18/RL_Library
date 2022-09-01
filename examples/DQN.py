from Networks.QNet import CreateQNetwork
from agents.DQNAgent import DQNAgent
from replay_buffers.UniformReplayMemory import ReplayMemory
from drivers.value_driver import driver
from replay_buffers.Utils import get_data_spec
import gym
from replay_buffers.Trajectory import Trajectory
from observers.metrics import AverageReturnObserver
from observers.metrics import AverageEpisodeLengthObserver
import numpy as np
import time

env = gym.make("CartPole-v1")

qnet = CreateQNetwork(env.observation_space, env.action_space, (10, 5))

data_spec = get_data_spec(env)

replay_memory = ReplayMemory(data_spec, 256, 50000)

agent = DQNAgent(env.observation_space, env.action_space, qnet)

all_observers = [AverageReturnObserver(), AverageEpisodeLengthObserver()]


def random_policy(traj):
    return np.random.randint(2)


replay_memory = driver(env, random_policy, replay_memory, 10)

for step in range(1000):
    agent.epsilon -= 1 / 1000
    replay_memory, all_observers = driver(env, agent.collect_policy, replay_memory, 1, observers=all_observers)
    experiences = replay_memory.sample_experience()
    agent.train(experiences)
    print("Average Return: ", all_observers[0].result(), " Average Steps: ", all_observers[1].result(), " Epsilon: ",
          agent.epsilon)

# Evaluation
done = False
eval_env = gym.make("CartPole-v1")
state= eval_env.reset()
traj = Trajectory(state, None, None, None, done)
while not done:
    action = agent.policy(traj)
    next_state, reward, done, _ = eval_env.step(action)
    traj = Trajectory(state, action, reward, next_state, done)
    eval_env.render()
    time.sleep(0.1)
    state = next_state



agent.save("saved_agents/DQN")