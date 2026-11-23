from collections import deque
import time
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from sim.scene_builder import build_maze_scene
from sim.pid_controller import DifferentialPID

DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 0=E, 1=N, 2=W, 3=S
ACTION_DH = [0, 1, -1, 2]                  # 0=Fwd, 1=Left, 2=Right, 3=Back
ACTION_PROX = [0, 1, 2, 5]                 # indices into prox_names


def wrap_angle(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


class MazeHierEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        buggy_xml_path: str = "assets/my_buggy.xml",
        rows: int = 5,
        cols: int = 5,
        cell_size: float = 0.30,
        pool_size: int = 40,
        max_steps: int = 80,
        noise_enabled: bool = True,
        base_seed: int = 0,
        viewer=None,
        render_realtime: bool = False,
    ):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.max_steps = max_steps
        self.noise_enabled = noise_enabled
        self.max_range = 1.2
        self.dt = 0.005
        self.viewer = viewer
        self.render_realtime = render_realtime

        self.prox_names = [
            "bot/rf_front",
            "bot/rf_left",
            "bot/rf_right",
            "bot/rf_fl",
            "bot/rf_fr",
            "bot/rf_back",
        ]

        self.pool = []
        for i in range(pool_size):
            model, data, maze = build_maze_scene(
                buggy_xml_path=buggy_xml_path,
                rows=rows,
                cols=cols,
                cell_size=cell_size,
                seed=base_seed + i,
            )
            wall_ids = {
                g for g in range(model.ngeom)
                if (model.geom(g).name or "").startswith("wall_")
            }
            bot_ids = {
                g for g in range(model.ngeom)
                if (model.geom(g).name or "").startswith("bot/")
            }
            bfs_dist = self._build_bfs(model, data, maze, wall_ids)
            self.pool.append((model, data, maze, wall_ids, bot_ids, bfs_dist))

        self.sensor_off = self._measure_sensor_offsets(*self.pool[0][:2])
        half_clear = 0.5 * (self.cell_size - 0.02)
        self.front_center = max(0.035, half_clear - self.sensor_off[0])
        self.side_center = max(0.035, half_clear - 0.5 * (self.sensor_off[1] + self.sensor_off[2]))
        self.open_thresh = np.array([
            half_clear - self.sensor_off[i] + 0.07 for i in ACTION_PROX
        ], dtype=np.float32)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(20,), dtype=np.float32)

        self.rng = np.random.default_rng(base_seed)
        self.pid = DifferentialPID(dt=self.dt)
        self.model = None
        self.data = None
        self.maze = None
        self.wall_ids = set()
        self.bot_ids = set()
        self.bfs_dist = None
        self._tick = 0

    def _measure_sensor_offsets(self, model, data):
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        bpos = data.body("bot/buggy").xpos[:2]
        offs = []
        for name in self.prox_names:
            sid = int(model.sensor_objid[model.sensor(name).id])
            offs.append(float(np.linalg.norm(data.site_xpos[sid][:2] - bpos)))
        return offs

    def _build_bfs(self, model, data, maze, wall_ids):
        mujoco.mj_resetData(model, data)
        free_jnts = np.where(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]
        if len(free_jnts) > 0:
            data.qpos[int(model.jnt_qposadr[free_jnts[0]]) + 2] = 5.0
        mujoco.mj_forward(model, data)

        adj = {(r, c): [] for r in range(self.rows) for c in range(self.cols)}
        geomid = np.zeros(1, dtype=np.int32)
        for r in range(self.rows):
            for c in range(self.cols):
                pnt = np.array([(c + 0.5) * self.cell_size, (r + 0.5) * self.cell_size, 0.04])
                for dr, dc in DIRS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        vec = np.array([float(dc), float(dr), 0.0])
                        d = mujoco.mj_ray(model, data, pnt, vec, None, 1, -1, geomid)
                        if not (0.0 <= d < self.cell_size and int(geomid[0]) in wall_ids):
                            adj[(r, c)].append((nr, nc))
        mujoco.mj_resetData(model, data)

        goal_cell = self._clamped_cell(*maze["goal_xy"])
        bfs = np.full((self.rows, self.cols), 999, dtype=np.int32)
        bfs[goal_cell] = 0
        q = deque([goal_cell])
        while q:
            r, c = q.popleft()
            for nr, nc in adj[(r, c)]:
                if bfs[nr, nc] > bfs[r, c] + 1:
                    bfs[nr, nc] = bfs[r, c] + 1
                    q.append((nr, nc))
        return bfs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        idx = int(self.rng.integers(len(self.pool)))
        self.model, self.data, self.maze, self.wall_ids, self.bot_ids, self.bfs_dist = self.pool[idx]

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        sx, sy = self.maze["start_xy"]
        self.pid.reset(x=sx, y=sy, theta=0.0)

        self.odom_r, self.odom_c = self._clamped_cell(sx, sy)
        self.odom_h = 0
        self.step_count = 0
        self.collision_steps = 0
        self.tick_collisions = 0
        self._tick = 0

        start_true = self._clamped_cell(*self._true_xy())
        self.min_bfs = int(self.bfs_dist[start_true])
        self.visited_true = {start_true}
        self.visit_counts = {(self.odom_r, self.odom_c): 1}
        self.goal_cell = self._clamped_cell(*self.maze["goal_xy"])

        return self._get_obs(), {}

    def _sim_tick(self, v_ref: float, w_ref: float):
        enc_vl = float(self.data.sensor("bot/enc_left_vel").data[0])
        enc_vr = float(self.data.sensor("bot/enc_right_vel").data[0])
        gyro_z = float(self.data.sensor("bot/gyro").data[2])

        if self.noise_enabled:
            enc_vl += float(self.rng.normal(0.0, 0.05))
            enc_vr += float(self.rng.normal(0.0, 0.05))
            gyro_z += float(self.rng.normal(0.0, 0.01))

        wl, wr, v_m, w_m = self.pid.step(v_ref, w_ref, enc_vl, enc_vr, gyro_z)
        self.data.actuator("bot/left_vel").ctrl[0] = wl
        self.data.actuator("bot/right_vel").ctrl[0] = wr
        mujoco.mj_step(self.model, self.data)

        hit = self._check_wall_collision()
        if hit:
            self.tick_collisions += 1

        self._tick += 1
        if self.viewer is not None and self._tick % 4 == 0:
            self.viewer.sync()
            if self.render_realtime:
                time.sleep(0.012)

        return v_m, w_m, hit

    def _clear_pid_integrators(self):
        self.pid.int_v = 0.0
        self.pid.int_w = 0.0
        self.pid.prev_err_v = 0.0
        self.pid.prev_err_w = 0.0

    def _execute_move(self, action: int):
        new_h = (self.odom_h + ACTION_DH[action]) % 4
        target_theta = wrap_angle(new_h * (np.pi / 2.0))
        hit_any = False

        # Phase A: In-place turn to exact cardinal heading
        if action != 0:
            self._clear_pid_integrators()
            for _ in range(320):
                err = wrap_angle(target_theta - self.pid.theta)
                if action == 3 and abs(err) > 2.75:
                    err = abs(err)
                w_ref = float(np.clip(4.0 * err, -2.5, 2.5))
                _, w_m, hit = self._sim_tick(0.0, w_ref)
                hit_any = hit_any or hit
                if abs(err) < 0.025 and abs(w_m) < 0.15:
                    break

        # Phase B: Drive 1 cell straight with corridor centering + front-wall alignment
        self._clear_pid_integrators()
        dist_traveled = 0.0
        side_wall_max = self.side_center + 0.045

        for _ in range(360):
            prox = self._read_prox(noisy=False)
            rf_f, rf_l, rf_r, rf_fl, rf_fr = prox[0], prox[1], prox[2], prox[3], prox[4]

            if dist_traveled > 0.16 and rf_f < self.front_center + 0.065:
                rem = rf_f - self.front_center
            else:
                rem = self.cell_size - dist_traveled

            if rem <= 0.005:
                break

            v_ref = float(np.clip(3.2 * rem, 0.06, 0.34))
            yaw_err = wrap_angle(target_theta - self.pid.theta)

            if rf_l < side_wall_max and rf_r < side_wall_max:
                e_lat = rf_l - rf_r
            elif rf_l < side_wall_max:
                e_lat = rf_l - self.side_center
            elif rf_r < side_wall_max:
                e_lat = self.side_center - rf_r
            else:
                e_lat = 0.0

            w_ref = 3.5 * yaw_err + 10.0 * e_lat
            if rf_fl < 0.036:
                w_ref -= 1.2 * (0.036 - rf_fl) / 0.036
            if rf_fr < 0.036:
                w_ref += 1.2 * (0.036 - rf_fr) / 0.036
            w_ref = float(np.clip(w_ref, -2.0, 2.0))

            v_m, _, hit = self._sim_tick(v_ref, w_ref)
            hit_any = hit_any or hit
            dist_traveled += max(0.0, v_m) * self.dt

        # Brief settle
        for _ in range(12):
            _, _, hit = self._sim_tick(0.0, 0.0)
            hit_any = hit_any or hit

        dr, dc = DIRS[new_h]
        self.odom_r = int(np.clip(self.odom_r + dr, 0, self.rows - 1))
        self.odom_c = int(np.clip(self.odom_c + dc, 0, self.cols - 1))
        self.odom_h = new_h
        self.pid.x = (self.odom_c + 0.5) * self.cell_size
        self.pid.y = (self.odom_r + 0.5) * self.cell_size
        return hit_any

    def step(self, action):
        action = int(action)
        self.step_count += 1

        prox = self._read_prox(noisy=False)
        is_open = prox[ACTION_PROX[action]] > self.open_thresh[action]

        reward = -0.1
        terminated = False

        if not is_open:
            reward -= 1.0
        else:
            hit = self._execute_move(action)
            if hit:
                self.collision_steps += 1
                reward -= 2.0

            cell_odom = (self.odom_r, self.odom_c)
            prev_cnt = self.visit_counts.get(cell_odom, 0)
            self.visit_counts[cell_odom] = prev_cnt + 1

            true_cell = self._clamped_cell(*self._true_xy())
            if true_cell not in self.visited_true:
                self.visited_true.add(true_cell)
                reward += 1.5
            else:
                reward -= 0.25

            curr_bfs = int(self.bfs_dist[true_cell])
            if curr_bfs < self.min_bfs:
                reward += 2.0 * float(self.min_bfs - curr_bfs)
                self.min_bfs = curr_bfs

            tx, ty = self._true_xy()
            gx, gy = self.maze["goal_xy"]
            if true_cell == self.goal_cell or float(np.hypot(gx - tx, gy - ty)) < 0.12:
                reward += 30.0
                if self.tick_collisions == 0:
                    reward += 10.0
                terminated = True

        truncated = self.step_count >= self.max_steps
        info = {
            "is_success": terminated,
            "collisions": self.collision_steps,
            "tick_collisions": self.tick_collisions,
            "cells_visited": len(self.visited_true),
        }
        return self._get_obs(), reward, terminated, truncated, info

    def _read_prox(self, noisy: bool = True):
        vals = []
        for name in self.prox_names:
            d = float(self.data.sensor(name).data[0])
            d = self.max_range if d < 0 else min(d, self.max_range)
            if noisy and self.noise_enabled:
                d = float(np.clip(d + self.rng.normal(0.0, 0.008), 0.0, self.max_range))
            vals.append(d)
        return np.array(vals, dtype=np.float32)

    def _get_obs(self):
        prox = self._read_prox(noisy=True)
        card_prox = prox[ACTION_PROX]
        open_dirs = (card_prox > self.open_thresh).astype(np.float32)
        prox_norm = card_prox / self.max_range

        adj_visited = []
        adj_counts = []
        for dh in ACTION_DH:
            h = (self.odom_h + dh) % 4
            dr, dc = DIRS[h]
            nc = (self.odom_r + dr, self.odom_c + dc)
            cnt = self.visit_counts.get(nc, 0)
            adj_visited.append(1.0 if cnt > 0 else 0.0)
            adj_counts.append(min(cnt, 4) / 4.0)

        gx, gy = self.maze["goal_xy"]
        dx_w = gx - self.pid.x
        dy_w = gy - self.pid.y
        c, s = np.cos(self.pid.theta), np.sin(self.pid.theta)
        max_diag = np.hypot(self.cols * self.cell_size, self.rows * self.cell_size)
        goal_local = np.clip([(c * dx_w + s * dy_w) / max_diag, (-s * dx_w + c * dy_w) / max_diag], -1.0, 1.0)

        return np.concatenate([
            open_dirs,
            prox_norm,
            adj_visited,
            adj_counts,
            goal_local,
            [s, c],
        ]).astype(np.float32)

    def _true_xy(self):
        pos = self.data.body("bot/buggy").xpos
        return float(pos[0]), float(pos[1])

    def _clamped_cell(self, x: float, y: float):
        r = int(np.clip(np.floor(y / self.cell_size), 0, self.rows - 1))
        c = int(np.clip(np.floor(x / self.cell_size), 0, self.cols - 1))
        return (r, c)

    def _check_wall_collision(self):
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            if (g1 in self.wall_ids and g2 in self.bot_ids) or (
                g2 in self.wall_ids and g1 in self.bot_ids
            ):
                return True
        return False