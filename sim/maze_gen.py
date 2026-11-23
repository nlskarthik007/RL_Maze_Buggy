import numpy as np



def generate_maze(
    rows: int = 5,
    cols: int = 5,
    cell_size: float = 0.30,
    wall_thickness: float = 0.02,
    wall_height: float = 0.12,
    seed: int | None = None,
):
    rng = np.random.default_rng(seed)

    h_walls = np.ones((rows + 1, cols), dtype=bool)
    v_walls = np.ones((rows, cols + 1), dtype=bool)
    visited = np.zeros((rows, cols), dtype=bool)

    stack = [(0, 0)]
    visited[0, 0] = True

    while stack:
        r, c = stack[-1]
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                neighbors.append((nr, nc, dr, dc))

        if not neighbors:
            stack.pop()
            continue

        nr, nc, dr, dc = neighbors[rng.integers(len(neighbors))]
        if dr == 1:
            h_walls[r + 1, c] = False
        elif dr == -1:
            h_walls[r, c] = False
        elif dc == 1:
            v_walls[r, c + 1] = False
        elif dc == -1:
            v_walls[r, c] = False

        visited[nr, nc] = True
        stack.append((nr, nc))

    walls = []
    hz = wall_height / 2.0
    half_t = wall_thickness / 2.0
    half_len = (cell_size + wall_thickness) / 2.0

    for r in range(rows + 1):
        for c in range(cols):
            if h_walls[r, c]:
                cx = (c + 0.5) * cell_size
                cy = r * cell_size
                walls.append((cx, cy, half_len, half_t, hz))

    for r in range(rows):
        for c in range(cols + 1):
            if v_walls[r, c]:
                cx = c * cell_size
                cy = (r + 0.5) * cell_size
                walls.append((cx, cy, half_t, half_len, hz))

    start_xy = (0.5 * cell_size, 0.5 * cell_size)
    goal_xy = ((cols - 0.5) * cell_size, (rows - 0.5) * cell_size)
    return {
        "walls": walls,
        "start_xy": start_xy,
        "goal_xy": goal_xy,
        "rows": rows,
        "cols": cols,
        "cell_size": cell_size,
    }