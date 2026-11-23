import argparse
import gc
import multiprocessing as mp
import time
import numpy as np
import torch
import mujoco.viewer
from sb3_contrib import RecurrentPPO

from env.maze_hier_env import MazeHierEnv


def run_episode(model, seed, rows, cols, max_steps, noise, headless, hold):
    env = MazeHierEnv(
        rows=rows,
        cols=cols,
        pool_size=1,
        max_steps=max_steps,
        base_seed=seed,
        noise_enabled=noise,
        render_realtime=not headless,
    )
    obs, _ = env.reset(seed=seed)
    lstm_states = None
    ep_start = np.ones((1,), dtype=bool)
    done = False
    info = {"is_success": False, "collisions": 0, "tick_collisions": 0}

    try:
        if headless:
            while not done:
                action, lstm_states = model.predict(obs, state=lstm_states, episode_start=ep_start, deterministic=True)
                obs, _, term, trunc, info = env.step(action)
                ep_start = np.zeros((1,), dtype=bool)
                done = term or trunc
        else:
            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                env.viewer = viewer
                while viewer.is_running() and not done:
                    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=ep_start, deterministic=True)
                    obs, _, term, trunc, info = env.step(action)
                    ep_start = np.zeros((1,), dtype=bool)
                    done = term or trunc

                t_end = time.time() + hold
                while viewer.is_running() and (hold < 0 or time.time() < t_end):
                    viewer.sync()
                    time.sleep(0.05)
    finally:
        steps = env.step_count
        env.viewer = None
        del env
        gc.collect()

    return info["is_success"], steps, info["collisions"], info["tick_collisions"]


def _viewer_worker(ckpt, seed, rows, cols, max_steps, noise, hold, q):
    torch.set_num_threads(1)
    model = RecurrentPPO.load(ckpt)
    q.put(run_episode(model, seed, rows, cols, max_steps, noise, False, hold))


def run_viewer_isolated(ckpt, seed, rows, cols, max_steps, noise, hold):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=_viewer_worker, args=(ckpt, seed, rows, cols, max_steps, noise, hold, q))
    proc.start()
    proc.join()
    if not q.empty():
        return q.get()
    return False, 0, 0, 0


if __name__ == "__main__":
    torch.set_num_threads(1)

    p = argparse.ArgumentParser()
    p.add_argument("ckpt", nargs="?", default="checkpoints/maze_5x5_hier_best.zip")
    p.add_argument("--rows", type=int, default=5)
    p.add_argument("--cols", type=int, default=5)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--noise", action="store_true")
    p.add_argument("--headless", action="store_true")
    args = p.parse_args()

    max_steps = args.max_steps or max(80, args.rows * args.cols * 4)
    rng = np.random.default_rng()

    if args.seed is not None:
        seeds = [args.seed + i for i in range(args.episodes)]
    else:
        seeds = [int(rng.integers(20_000, 999_999)) for _ in range(args.episodes)]

    model = RecurrentPPO.load(args.ckpt) if (args.headless or args.episodes == 1) else None
    solved, zero_col, moves_ok, ticks_ok, failed = 0, 0, [], [], []
    hold = -1.0 if args.episodes == 1 else 1.5

    for i, s in enumerate(seeds, 1):
        if not args.headless and args.episodes > 1:
            ok, steps, col_moves, tick_hits = run_viewer_isolated(
                args.ckpt, s, args.rows, args.cols, max_steps, args.noise, hold
            )
        else:
            ok, steps, col_moves, tick_hits = run_episode(
                model, s, args.rows, args.cols, max_steps, args.noise, args.headless, hold
            )

        tag = "SOLVED" if ok else "TIMEOUT"
        if not args.headless or not ok:
            print(f"[{i}/{args.episodes}] seed={s} ({args.rows}x{args.cols}) | {tag} in {steps} moves | col_moves={col_moves} | tick_hits={tick_hits}")
        if ok:
            solved += 1
            moves_ok.append(steps)
            ticks_ok.append(tick_hits)
            if tick_hits == 0:
                zero_col += 1
        else:
            failed.append(s)

    print(f"\nResult ({args.rows}x{args.cols}, noise={args.noise}): Solved {solved}/{args.episodes} | Zero-col {zero_col}/{args.episodes}")
    if moves_ok:
        print(f"Avg moves (solved): {np.mean(moves_ok):.1f} | Avg tick hits: {np.mean(ticks_ok):.2f}")
    if failed:
        print(f"Failed seeds: {failed[:15]}")