import numpy as np
import tensorflow as tf
from tensorflow import keras
from replay_buffers.UniformReplayMemory import *

def DQNAgent:
    def __init__(self, qnet, replay_mem, learning_rate=0.01, df=0.99, epsilon_decay=0.005):
        self.replay_mem = replay_mem
        self.learning_rate = learning_rate
        self.df = df
        self.epsilon_decay = epsilon_decay
        self.episode = 0
        self.qnet = qnet

    def collect_policy(self, trajectory):
        if trajectory.done == True:
            self.episode += 1
            self.epsilon ==

        self.epsilon =

    def policy():

    def train(self, experiences):