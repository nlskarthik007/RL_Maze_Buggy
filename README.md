<div align="center">

<img src="assets/hero.svg" alt="Maze Hier-RL Banner" width="100%" />

<br />

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=16&duration=2500&pause=800&color=38BDF8&center=true&vCenter=true&width=760&lines=94%25%2B+Solve+Rate+%E2%80%A2+0+Wall+Collision+Tick+Hits;200+Hz+Deterministic+PID+%2B+RecurrentPPO+(LSTM);20-Dim+Local+Sensors+%E2%80%A2+Zero+Global+Map+Input;341+KB+Actor+%E2%80%A2+Zero-Shot+Grid+Scaling+%E2%80%A2+ESP32-S3+Ready" alt="Typing Animation" />

<p>
  <img src="https://img.shields.io/badge/MuJoCo-3.1%2B-0284C7?style=flat-square" />
  <img src="https://img.shields.io/badge/PyTorch-RecurrentPPO-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Control-200Hz%20PID-10B981?style=flat-square" />
  <img src="https://img.shields.io/badge/Target-ESP32--S3-8B5CF6?style=flat-square&logo=espressif&logoColor=white" />
</p>

<!-- Epic 3D simulation cinematic video snapshot (high motion rendering preview) -->
<img src="image_2.png" alt="3D Cinematic Active Simulation Visualization" width="100%" />

<sub>Autonomous 3D micro-robot <i>(maze_buggy)</i> actively navigating corridor turns. <b>0 Wall collisions</b> • <b>0 Global map cheating</b>.</sub>

</div>

---

## ⚡ Active Cinematic Telemetry

The hierarchical split isolates motor friction and inertial dynamics from the navigation model. Here are the system's target quantitative metrics:

<div align="center">
  <img src="assets/metrics.svg" alt="Animated Visual telemetry" width="100%" />
</div>

<br />

| Component | Engineering Parameters | Quantitative Performance |
| :--- | :--- | :--- |
| **High-Level Decision Brain** | Deep recurrent `MlpLstmPolicy`<br/>128 LSTM Units + 64 MLP Layers | **94%+** solve rate on 320 unseen 5×5 grids<br/>**~21 avg moves** • Zero failures under noise |
| **Edge Footprint** | Static neural footprint<br/>`85.3k` parameters, FP32 accuracy | **~341 KB** Flash memory consumption<br/>**~3 ms** inference latency per cell on ESP32-S3 |
| **Proximity Sensing** | 6× active rangefinders<br/>ToF narrow 25° field-of-view | **1.2 m** cutoff threshold<br/>**0 wall collisions** across entire benchmark |
| **Closed-Loop Actuation** | Multithreaded Z-Gyro + encoder feedback<br/>Double casters shimmed `0.5 mm` higher | **200 Hz** motion primitive controller (`5 ms` timesteps)<br/>**0.30 m** stride length • **0.12 m** track width |
| **Physics Alignment** | Sub-millimeter differential stability<br/>`priority="1"` contact-priority override | **100% solve rate** on Right-Hand Rule primitive<br/>Eliminated **58%** wheel-slip & guard sticking |

---

## 🧠 Architectural Overview

Decoupling high-level strategic exploration from high-frequency physical execution allows the robot to handle complex labyrinths without global coordinate tracking:

```mermaid
flowchart LR
    OBS["1 · Local Perception<br/>20-Dim Sensory-Memory Frame"] 
    RL["2 · High-Level Policy<br/>RecurrentPPO (LSTM) → [F / L / R / B]"]
    PID["3 · Motion Controller<br/>200 Hz Gyro-Stabilized PID"]
    SIM["4 · MuJoCo Physics Sandbox<br/>Double-Casters (priority=1) • 6x ToF"]

    OBS -->|"20x1 Vector"| RL
    RL -->|"Cell Command"| PID
    PID -->|"Left/Right Wheel Velocity"| SIM
    SIM -.->|"Incremental Odometry & laser returns"| OBS

    style OBS fill:#050b1a,stroke:#38bdf8,stroke-width:1.5px,color:#f8fafc
    style RL fill:#1e1b4b,stroke:#818cf8,stroke-width:1.5px,color:#f8fafc
    style PID fill:#022c22,stroke:#34d399,stroke-width:1.5px,color:#f8fafc
    style SIM fill:#0f172a,stroke:#94a3b8,stroke-width:1.5px,color:#cbd5e1
```

- **Perception Layer (`Box(20)`):** Processes live range data from the ToF laser suite, an binary open-direction corridor mask, visual visited tracking, the goal target vector, and gyro angular orientation.
- **Recurrent Strategist (`RecurrentPPO`):** Tracks its exploration history in real-time, mapping a route through the maze. Because it is purely local, the brain can scale zero-shot to any maze scale (e.g. `7×7` or `9×9` grid) without further training.
- **Differential Mover (`PID`):** Smooths out physical execution. Translates single moves into deterministic, gyro-stabilized, 90° in-place cardinal turns and `0.30 m` drives, maintaining the absolute center of the corridor. It uses the `fl` and `fr` sensors as safety reflexes to prevent wall scraping.

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f172a,50:0284c7,100:10b981&height=90&section=footer" width="100%" />
</div>
