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
            next_state, reward, done, _, _ = env.step(action)
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
