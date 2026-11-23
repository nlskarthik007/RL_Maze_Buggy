<div align="center">

<img src="assets/hero.svg" alt="Maze Hier-RL Animated Banner" width="100%" />

<br />

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=16&duration=2500&pause=800&color=38BDF8&center=true&vCenter=true&width=760&lines=94%25%2B+Solve+Rate+%E2%80%A2+0+Wall+Collision+Tick+Hits;200+Hz+Deterministic+PID+%2B+RecurrentPPO+(LSTM);20-Dim+Local+Sensors+%E2%80%A2+Zero+Global+Map+Input;341+KB+Actor+%E2%80%A2+Zero-Shot+Grid+Scaling+%E2%80%A2+ESP32-S3+Ready" alt="Typing Animation" />

<p>
  <img src="https://img.shields.io/badge/MuJoCo-3.1%2B-0284C7?style=flat-square" />
  <img src="https://img.shields.io/badge/PyTorch-RecurrentPPO-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Control-200Hz%20PID-10B981?style=flat-square" />
  <img src="https://img.shields.io/badge/Target-ESP32--S3-8B5CF6?style=flat-square&logo=espressif&logoColor=white" />
</p>

</div>

---

## Quantitative Highlights

<div align="center">
  <img src="assets/metrics.svg" alt="Animated System Telemetry Dashboard" width="100%" />
</div>

<br />

| Category | Specification | Measured Value |
| :--- | :--- | :--- |
| **Navigation Performance** | Procedural 5×5 Solve Rate / Avg Moves | **94%+** / **~21 moves** (`10` moves on unseen test) |
| **Safety** | Wall Collision Tick Hits | **0** |
| **Training Efficiency** | Timesteps / CPU Wall-Clock Time | **120,000 steps** / **~19 min** (`8` envs, `320` mazes) |
| **Policy Interface** | Observation / Action Space | **20-dim** local vector / **Discrete(4)** (`F, L, R, B`) |
| **Low-Level Control** | PID Loop Rate / Cell Stride | **200 Hz** (`5 ms` tick) / **0.30 m** cell step |
| **Robot Kinematics** | Chassis / Wheels / Track Width | **120 × 96 mm** / **r = 0.03 m** / **L = 0.12 m** |
| **Sensor Suite** | Laser Rangefinders / Cutoff | **6× ToF** (`4` cardinal + `2` diagonal) / **1.2 m** |
| **Edge Deployment** | LSTM Actor Size / Latency | **~341 KB** (`85.3k` FP32 params) / **~3 ms/cell** on ESP32-S3 |

---

## Architecture

The system decouples **strategic maze exploration** from **precise physical motion** into a two-tier hierarchy. Separating decision-making from actuation eliminates wall-grazing failures, accelerates RL convergence, and allows the same policy to scale zero-shot to larger grids or deploy directly to embedded hardware.

<div align="center">
  <img src="assets/architecture.svg" alt="Animated Hierarchical RL + PID Architecture Flowchart" width="100%" />
</div>

<br />

- **Local Perception (`20-Dim`):** Never receives the global wall map. At each cell, the robot builds a local vector from 4 cardinal rangefinders (`1.2 m` cutoff), binary open-corridor flags, odometry-tracked visited cell counts, a body-frame goal bearing vector, and its gyro heading.
- **High-Level Policy (`RecurrentPPO`):** A 128-unit LSTM evaluates the local state once per cell and outputs a relative `Discrete(4)` command (**Forward, Left, Right, or Back**), using its hidden state and breadcrumb memory to backtrack out of dead ends.
- **Low-Level Controller (`200 Hz PID`):** Filters blocked directions before motion starts. Valid moves run in two deterministic phases: an **in-place 90°/180° gyro-locked turn**, followed by a **`0.30 m` corridor drive** with side-wall centering and front-wall distance re-zeroing to cancel encoder drift.
- **Physics & Hardware Transfer:** Low-friction skids lifted `0.5 mm` (`z = -0.0195 m`) and side guards use contact priority tuning (`priority="1"`) so floor friction never brakes the drive wheels—mirroring ball casters on a physical **ESP32-S3 + VL53L0X + N20 encoder** build.

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f172a,50:0284c7,100:10b981&height=90&section=footer" width="100%" />
</div>
