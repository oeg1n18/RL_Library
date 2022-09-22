from replay_buffers.Trajectory import Trajectory
import numpy as np


def driver(env, policy, replay_buffer, n_episodes, observers=None):
    all_returns = []
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        traj = Trajectory(state, None, None, None, done)
        returns = 0
        while True:
            action = policy(traj)
            next_state, reward, done, _ = env.step(action)
            traj = Trajectory(state, action, reward, next_state, done)
            replay_buffer.add_trajectory(traj)
            if observers:
                for observer in observers:
                    observer.call(traj)
            returns += 1
            if traj.done:
                break
            state = next_state
        all_returns.append(returns)
    if observers:
        return replay_buffer, observers
    else:
        return replay_buffer


def ac_driver(env, agent, n_episodes, observers=None):
    all_returns = []
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        traj = Trajectory(state, None, None, None, done)
        returns = 0
        while True:
            action = agent.policy(traj)
            next_state, reward, done, _ = env.step(action)
            traj = Trajectory(state, action, reward, next_state, done)
            agent.train_step(traj)
            if observers:
                for observer in observers:
                    observer.call(traj)
            returns += 1
            if traj.done:
                break
            state = next_state
        all_returns.append(returns)
    if observers:
        return agent, observers
    else:
        return agent




def ddpg_driver(env, agent, replay_buffer, n_episodes, observers=None):
    all_returns = []
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        traj = Trajectory(state, None, None, None, done)
        returns = 0
        while True:
            action = agent.policy(traj)
            next_state, reward, done, _ = env.step(action)
            traj = Trajectory(state, action, reward, next_state, done)
            replay_buffer.add_trajectory(traj)
            experiences = replay_buffer.sample_experience()
            agent.train(experiences)
            if observers:
                for observer in observers:
                    observer.call(traj)
            returns += 1
            if traj.done:
                break
            state = next_state
        all_returns.append(returns)
    if observers:
        return replay_buffer, observers
    else:
        return replay_buffer


def ppo_driver(env, agent, n_episodes, observers=None):
    train_iters = 0
    n_steps = 0
    N = 20
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.policy(state)
            next_state, reward, done, info = env.step(action)
            traj = Trajectory(state, action, reward, next_state, done)
            agent.add_trajectory(state, action, prob, val, reward, done)
            if observers:
                for observer in observers:
                    observer.call(traj)
            if n_steps % N == 0:
                agent.train()
                train_iters += 1
            state = next_state
        if observers:
            return agent, observers
        else:
            return agent
