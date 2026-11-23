import sys
import time
import numpy as np
import mujoco.viewer
from sb3_contrib import RecurrentPPO

from env.maze_hier_env import MazeHierEnv

ckpt = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 9999

model = RecurrentPPO.load(ckpt)
env = MazeHierEnv(rows=5, cols=5, pool_size=1, base_seed=seed, noise_enabled=False, render_realtime=True)
obs, _ = env.reset()

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    env.viewer = viewer
    lstm_states = None
    ep_start = np.ones((1,), dtype=bool)
    done = False

    while viewer.is_running() and not done:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=ep_start, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        ep_start = np.zeros((1,), dtype=bool)
        done = term or trunc

    status = "Goal reached" if info["is_success"] else "Timed out"
    print(f"{status} in {env.step_count} cell moves | collision moves: {info['collisions']} | physics tick hits: {info['tick_collisions']}")
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.05)