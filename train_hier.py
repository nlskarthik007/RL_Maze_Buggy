import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import RecurrentPPO

from env.maze_hier_env import MazeHierEnv


def test_primitives():
    solved, zero_col = 0, 0
    for s in range(9800, 9805):
        env = MazeHierEnv(rows=5, cols=5, pool_size=1, base_seed=s, noise_enabled=False)
        obs, _ = env.reset()
        done = False
        while not done:
            # Simple Right-Hand Rule sanity check on open_dirs (obs[0:4] = F, L, R, B)
            for a in (2, 0, 1, 3):
                if obs[a] > 0.5:
                    action = a
                    break
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        if info["is_success"]:
            solved += 1
            if info["tick_collisions"] == 0:
                zero_col += 1
    print(f"[Primitive Sanity Check] RHR Solved: {solved}/5 | Zero-collision: {zero_col}/5")


def eval_unseen(model, n=20):
    solved, zero_col, steps_list, ticks_list = 0, 0, [], []
    for s in range(9900, 9900 + n):
        env = MazeHierEnv(rows=5, cols=5, pool_size=1, base_seed=s, noise_enabled=False)
        obs, _ = env.reset()
        lstm_states = None
        ep_start = np.ones((1,), dtype=bool)
        done = False
        while not done:
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=ep_start, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            ep_start = np.zeros((1,), dtype=bool)
            done = term or trunc
        if info["is_success"]:
            solved += 1
            steps_list.append(env.step_count)
            ticks_list.append(info["tick_collisions"])
            if info["tick_collisions"] == 0:
                zero_col += 1
    avg_s = np.mean(steps_list) if steps_list else 0.0
    avg_c = np.mean(ticks_list) if ticks_list else 0.0
    print(f"\n==> [Unseen 5x5 Eval] Solved: {solved}/{n} | Zero-col: {zero_col}/{n} | Avg moves: {avg_s:.1f} | Avg tick hits: {avg_c:.1f}\n")
    return solved, zero_col


class UnseenEvalCallback(BaseCallback):
    def __init__(self, eval_every: int = 10_000, save_path: str = "checkpoints/maze_5x5_hier_best"):
        super().__init__()
        self.eval_every = eval_every
        self.save_path = save_path
        self.next_eval = eval_every
        self.best = (-1, -1)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.eval_every
            score = eval_unseen(self.model, n=20)
            if score > self.best:
                self.best = score
                self.model.save(self.save_path)
                print(f"[NEW BEST @ {self.num_timesteps} steps -> {self.save_path}.zip]\n")
        return True


def make_env(rank):
    def _init():
        return MazeHierEnv(rows=5, cols=5, pool_size=40, max_steps=80, base_seed=3000 + rank * 100)
    return _init


if __name__ == "__main__":
    test_primitives()

    env = VecMonitor(SubprocVecEnv([make_env(i) for i in range(8)]), info_keywords=("is_success",))
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=3e-4,
        n_steps=64,
        batch_size=128,
        n_epochs=6,
        gamma=0.98,
        ent_coef=0.02,
        target_kl=0.02,
        policy_kwargs=dict(lstm_hidden_size=128, n_lstm_layers=1, net_arch=dict(pi=[64], vf=[64])),
        verbose=1,
    )

    cb = UnseenEvalCallback(eval_every=10_000, save_path="checkpoints/maze_5x5_hier_best")
    try:
        model.learn(total_timesteps=120_000, callback=cb)
        model.save("checkpoints/maze_5x5_hier_final")
    except KeyboardInterrupt:
        out = f"checkpoints/maze_5x5_hier_{model.num_timesteps}_steps"
        model.save(out)
        print(f"\n[Interrupted — saved to {out}.zip]")