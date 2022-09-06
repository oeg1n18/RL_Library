
from Networks.QNet import CreateQNetwork
from replay_buffers.Trajectory import Trajectory
from replay_buffers.UniformReplayMemory import ReplayMemory
from replay_buffers.Utils import get_data_spec

from drivers.openai_drivers import ddpg_driver
from agents.DDPGAgent import DDPGAgent
from observers.metrics import AverageReturnObserver
from observers.metrics import AverageEpisodeLengthObserver
import gym
import numpy as np
import time



env = gym.make("Pendulum-v1")

agent = DDPGAgent(input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0])

replay_memory = ReplayMemory(get_data_spec(env), 64, 1000000)

all_observers = [AverageReturnObserver(), AverageEpisodeLengthObserver()]

for step in range(1000):
    replay_memory, all_observers = ddpg_driver(env, agent, replay_memory, 1, observers=all_observers)
    print("Average Return: ", all_observers[0].result(), " Average Steps: ", all_observers[1].result())

# Evaluation
done = False
eval_env = gym.make("Pendulum-v0")
state= eval_env.reset()
traj = Trajectory(state, None, None, None, done)
while not done:
    action = agent.policy(traj, evaluate=True)
    next_state, reward, done, _ = eval_env.step(action)
    traj = Trajectory(state, action, reward, next_state, done)
    eval_env.render()
    time.sleep(0.1)
    state = next_state


agent.save("saved_agents/DDPG")
