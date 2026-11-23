import mujoco
from sim.maze_gen import generate_maze


def build_maze_scene(
    buggy_xml_path: str = "assets/my_buggy.xml",
    rows: int = 5,
    cols: int = 5,
    cell_size: float = 0.30,
    seed: int | None = None,
):
    maze = generate_maze(rows=rows, cols=cols, cell_size=cell_size, seed=seed)

    scene = mujoco.MjSpec()
    scene.option.timestep = 0.005
    scene.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    scene.option.gravity = [0.0, 0.0, -9.81]

    # Lighting & floor
    scene.worldbody.add_light(pos=[cols * cell_size / 2, rows * cell_size / 2, 3.0], dir=[0, 0, -1])
    scene.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[10.0, 10.0, 0.1],
        friction=[1.0, 0.005, 0.001],
        rgba=[0.92, 0.92, 0.92, 1.0],
    )

    # Goal marker (contype=0, conaffinity=0 -> visual only, no collision)
    gx, gy = maze["goal_xy"]
    scene.worldbody.add_geom(
        name="goal_marker",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[gx, gy, 0.002],
        size=[cell_size * 0.28, 0.002, 0.0],
        contype=0,
        conaffinity=0,
        rgba=[0.1, 0.85, 0.3, 0.6],
    )

    # Maze walls
    for i, (cx, cy, hx, hy, hz) in enumerate(maze["walls"]):
        scene.worldbody.add_geom(
            name=f"wall_{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[cx, cy, hz],
            size=[hx, hy, hz],
            rgba=[0.22, 0.25, 0.32, 1.0],
        )

    # Attach buggy at start cell (z=0 because my_buggy.xml already has z=0.041)
    sx, sy = maze["start_xy"]
    buggy_spec = mujoco.MjSpec.from_file(buggy_xml_path)
    spawn_frame = scene.worldbody.add_frame(pos=[sx, sy, 0.0])
    spawn_frame.attach_body(buggy_spec.body("buggy"), "bot/", "")

    model = scene.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, maze