import os
import sys
import time
import numpy as np
import mujoco.viewer
from sb3_contrib import RecurrentPPO
from env.maze_hier_env import MazeHierEnv


def run_smooth_episode(env, model, obs, max_moves: int = 120):
    lstm_states = None
    ep_start = np.ones((1,), dtype=bool)

    def peek_policy(cur_obs):
        nonlocal lstm_states, ep_start
        act, lstm_states = model.predict(
            cur_obs, state=lstm_states, episode_start=ep_start, deterministic=True
        )
        ep_start = np.zeros((1,), dtype=bool)
        return int(act)

    action = peek_policy(obs)
    carry = 0.0
    stopped = True
    moves = 0
    chained_straights = 0
    t0 = env.data.time
    act_names = ["F", "L", "R", "B"]

    for _ in range(max_moves):
        if env.viewer is not None and not env.viewer.is_running():
            break

        obs_now = env._get_obs()
        if obs_now[action] < 0.5:
            if not stopped:
                for _ in range(15):
                    env._apply_wheel_vel(0.0, 0.0)
                    env._sim_tick()
                stopped = True
                carry = 0.0
            action = peek_policy(obs_now)
            continue

        if action != 0:
            env.turn_in_place(action)

        res = env.drive_cell_lookahead(peek_fn=peek_policy, carry=carry)
        moves += 1
        carry = res["carry"]
        stopped = res["stopped"]
        if not stopped:
            chained_straights += 1

        nxt = act_names[res["next_action"]] if res["next_action"] is not None else "-"
        print(f"  move {moves:2d}: act={act_names[action]} -> next={nxt} | {'CHAIN' if not stopped else 'STOP '}")

        if res["at_goal"]:
            sim_dt = env.data.time - t0
            print(
                f"Goal reached in {moves} moves ({chained_straights} non-stop straight chains) "
                f"| sim time: {sim_dt:.2f}s | hits: {env.wall_hits}"
            )
            return True

        action = res["next_action"] if res["next_action"] is not None else peek_policy(env._get_obs())

    print(f"Max moves ({max_moves}) reached without goal.")
    return False


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/maze_5x5_hier_best.zip"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 9999
    smooth = "--stop-and-go" not in sys.argv

    model = RecurrentPPO.load(ckpt)
    env = MazeHierEnv(rows=5, cols=5, pool_size=1, base_seed=seed, noise_enabled=False, render_realtime=True)
    obs, _ = env.reset(seed=seed)  # Reset ONCE before launching viewer

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        env.viewer = viewer
        if smooth:
            run_smooth_episode(env, model, obs)
        else:
            lstm_states, ep_start, done = None, np.ones((1,), dtype=bool), False
            while not done and viewer.is_running():
                act, lstm_states = model.predict(obs, state=lstm_states, episode_start=ep_start, deterministic=True)
                obs, _, term, trunc, info = env.step(act)
                ep_start = np.zeros((1,), dtype=bool)
                done = term or trunc
        time.sleep(1.5)
        env.viewer = None  # Detach before viewer closes