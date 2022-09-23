
import gym

from agents.PPOAgent import PPOAgent
from drivers.openai_drivers import ppo_driver
from observers.metrics import AverageReturnObserver


env = gym.make("CartPole-v0")

agent = PPOAgent(n_actions=env.action_space.n, batch_size=5, alpha=0.0003, n_epochs=4, input_dims=env.observation_space.shape)

observers = [AverageReturnObserver()]

for batch_episode in range(300):
    agent, observers = ppo_driver(env, agent, 10, observers=observers)
    print("Episode: ", batch_episode * 10, " Average Return: ", observers[0].result())