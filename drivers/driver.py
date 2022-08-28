
from replay_buffers.Trajectory import Trajectory


def driver(env, policy, replay_buffer, n_episodes):
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        traj = Trajectory(state, None, None, None, done)
        while True:
            action = policy(traj)
            next_state, reward, done, _, _ = env.step(action)
            traj = Trajectory(state, action, reward, next_state, done)
            replay_buffer.add_trajectory(traj)
            if traj.done:
                break
            state = next_state
    return replay_buffer
