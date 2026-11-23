import numpy as np


class DifferentialPID:
    def __init__(
        self,
        wheel_radius: float = 0.03,
        track_width: float = 0.12,
        dt: float = 0.005,
        max_wheel_rad_s: float = 30.0,
    ):
        self.r = wheel_radius
        self.L = track_width
        self.dt = dt
        self.max_w = max_wheel_rad_s

        # Gains (feedforward + PID correction)
        self.kp_v, self.ki_v, self.kd_v = 1.8, 0.4, 0.02
        self.kp_w, self.ki_w, self.kd_w = 1.5, 0.3, 0.02
        self.reset()

    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        self.int_v = 0.0
        self.int_w = 0.0
        self.prev_err_v = 0.0
        self.prev_err_w = 0.0
        self.x = x
        self.y = y
        self.theta = theta

    def step(self, v_ref: float, w_ref: float, enc_vel_l: float, enc_vel_r: float, gyro_z: float):
        # Measured chassis speeds
        v_meas = self.r * 0.5 * (enc_vel_l + enc_vel_r)
        w_meas = gyro_z

        # Dead-reckoning odometry update (encoders for distance, gyro for heading)
        self.theta = np.arctan2(
            np.sin(self.theta + w_meas * self.dt),
            np.cos(self.theta + w_meas * self.dt),
        )
        self.x += v_meas * np.cos(self.theta) * self.dt
        self.y += v_meas * np.sin(self.theta) * self.dt

        # Linear PID
        err_v = v_ref - v_meas
        self.int_v = np.clip(self.int_v + err_v * self.dt, -0.5, 0.5)
        der_v = (err_v - self.prev_err_v) / self.dt
        self.prev_err_v = err_v
        v_cmd = v_ref + self.kp_v * err_v + self.ki_v * self.int_v + self.kd_v * der_v

        # Angular PID
        err_w = w_ref - w_meas
        self.int_w = np.clip(self.int_w + err_w * self.dt, -1.5, 1.5)
        der_w = (err_w - self.prev_err_w) / self.dt
        self.prev_err_w = err_w
        w_cmd = w_ref + self.kp_w * err_w + self.ki_w * self.int_w + self.kd_w * der_w

        # Differential drive inverse kinematics
        w_left = np.clip((v_cmd - 0.5 * self.L * w_cmd) / self.r, -self.max_w, self.max_w)
        w_right = np.clip((v_cmd + 0.5 * self.L * w_cmd) / self.r, -self.max_w, self.max_w)
        return float(w_left), float(w_right), v_meas, w_meas