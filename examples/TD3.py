
from Networks.QNet import CreateQNetwork
from replay_buffers.Trajectory import Trajectory
from replay_buffers.UniformReplayMemory import ReplayMemory
from replay_buffers.Utils import get_data_spec

from drivers.openai_drivers import td3_driver
from agents.TD3Agent import Td3Agent
from observers.metrics import AverageReturnObserver
from observers.metrics import AverageEpisodeLengthObserver
import gym
import numpy as np
import time



env = gym.make("Pendulum-v1")

agent = Td3Agent(env=env, n_actions=env.action_space.shape[0])

all_observers = [AverageReturnObserver(), AverageEpisodeLengthObserver()]

for step in range(1000):
    replay_memory, all_observers = td3_driver(env, agent, 1, observers=all_observers)
    print("Average Return: ", all_observers[0].result(), " Average Steps: ", all_observers[1].result())

# Evaluation
done = False
eval_env = gym.make("Pendulum-v0")
state= eval_env.reset()
traj = Trajectory(state, None, None, None, done)
while not done:
    action = agent.policy(traj)
    next_state, reward, done, _ = eval_env.step(action)
    traj = Trajectory(state, action, reward, next_state, done)
    eval_env.render()
    time.sleep(0.1)
    state = next_state


agent.save("saved_agents/TD3")
