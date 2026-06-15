"""Grid rendering utilities for human-readable ASCII log output."""

from __future__ import annotations

import numpy as np

from src.utils.minigrid_maps import OBJECT_TO_STR, COLOR_TO_STR, STATE_TO_STR

DIRECTION_NAMES = ["right", "down", "left", "up"]
DIRECTION_ARROWS = {0: ">", 1: "v", 2: "<", 3: "^"}

AGENT_PARTIAL_OBS_ROW = 6
AGENT_PARTIAL_OBS_COL = 3


def _render_cell(
    object_type_index: int,
    color_index: int,
    state_index: int,
) -> str:
    """Render a single grid cell as a fixed-width 6-character string.

    Unseen cells show '  ??  ', empty/floor show '  .   '.
    Walls show '  ##  '.  Objects show 'color_abbrev:name' centered.
    """
    object_name = OBJECT_TO_STR.get(object_type_index, f"?{object_type_index}")
    color_name = COLOR_TO_STR.get(color_index, f"c{color_index}")
    color_abbrev = _color_abbreviation(color_name)

    if object_name == "unseen":
        return "  ??  "
    if object_name in ("empty", "floor"):
        return "  .   "
    if object_name == "wall":
        return "  ##  "
    if object_name == "agent":
        arrow = DIRECTION_ARROWS.get(state_index, "?")
        return f"  @{arrow}  "

    label = f"{color_abbrev}:{object_name}"
    return f"{label:^6s}"


def _color_abbreviation(color_name: str) -> str:
    abbreviations = {
        "red": "r",
        "green": "g",
        "blue": "b",
        "purple": "p",
        "yellow": "y",
        "grey": "x",
    }
    return abbreviations.get(color_name, "?")


def _render_grid_lines(
    image_grid: np.ndarray,
    agent_row: int | None,
    agent_col: int | None,
    agent_direction: int | None,
) -> list[str]:
    """Render the grid rows as fixed-width ASCII lines.

    If agent_row/agent_col are provided and the grid cell at that position
    is empty/floor, the agent marker (with direction arrow) is drawn there.
    """
    rows, cols, _ = image_grid.shape
    lines: list[str] = []

    column_header = "      " + "".join(f"{c:^6d}" for c in range(cols))
    lines.append(column_header)

    for row_index in range(rows):
        cells: list[str] = []
        for col_index in range(cols):
            is_agent_cell = (
                agent_row is not None
                and agent_col is not None
                and row_index == agent_row
                and col_index == agent_col
            )
            object_type_index = int(image_grid[row_index, col_index, 0])
            color_index = int(image_grid[row_index, col_index, 1])
            state_index = int(image_grid[row_index, col_index, 2])

            if is_agent_cell and object_type_index in (0, 1, 3):
                arrow = DIRECTION_ARROWS.get(agent_direction, "?")
                cells.append(f"  @{arrow}  ")
            else:
                cells.append(_render_cell(object_type_index, color_index, state_index))

        lines.append(f"  {row_index:2d}  " + "".join(cells))

    return lines


def _decode_visible_objects(image_grid: np.ndarray) -> list[str]:
    """Extract a list of human-readable object descriptions from the grid."""
    objects: list[str] = []
    rows, cols, _ = image_grid.shape
    for row_index in range(rows):
        for col_index in range(cols):
            object_type_index = int(image_grid[row_index, col_index, 0])
            if object_type_index <= 2:
                continue
            object_name = OBJECT_TO_STR.get(object_type_index, f"type_{object_type_index}")
            color_name = COLOR_TO_STR.get(
                int(image_grid[row_index, col_index, 1]),
                f"color_{image_grid[row_index, col_index, 1]}",
            )
            state_name = STATE_TO_STR.get(
                int(image_grid[row_index, col_index, 2]),
                f"state_{image_grid[row_index, col_index, 2]}",
            )
            objects.append(
                f"{color_name} {object_name} (state={state_name}) at ({row_index},{col_index})"
            )
    return objects


def render_partial_observation(
    image_grid: np.ndarray,
    direction: int,
    mission: str,
) -> str:
    """Render the agent's 7x7 partial observation as human-readable ASCII."""
    facing_name = (
        DIRECTION_NAMES[direction]
        if direction < len(DIRECTION_NAMES)
        else str(direction)
    )

    lines: list[str] = []
    lines.append(f"  Mission: {mission}")
    lines.append(f"  Facing:  {facing_name}  (forward = top of this view)")
    lines.append("")

    image_grid_display = image_grid.transpose(1, 0, 2)

    grid_lines = _render_grid_lines(
        image_grid_display,
        agent_row=AGENT_PARTIAL_OBS_ROW,
        agent_col=AGENT_PARTIAL_OBS_COL,
        agent_direction=3,
    )
    lines.extend(grid_lines)

    lines.append("")
    lines.append("  Visible objects:")
    visible_objects = _decode_visible_objects(image_grid_display)
    if visible_objects:
        for object_description in visible_objects:
            lines.append(f"    - {object_description}")
    else:
        lines.append("    (none)")

    return "\n".join(lines)


def render_full_grid(
    full_grid_encoded: np.ndarray,
    agent_position: tuple[int, int],
    agent_direction: int,
    mission: str,
) -> str:
    """Render the full environment grid as human-readable ASCII."""
    grid_width, grid_height, _ = full_grid_encoded.shape
    display_grid = full_grid_encoded.transpose(1, 0, 2)

    agent_col, agent_row = agent_position
    facing_name = (
        DIRECTION_NAMES[agent_direction]
        if agent_direction < len(DIRECTION_NAMES)
        else str(agent_direction)
    )

    lines: list[str] = []
    lines.append(f"  Mission: {mission}")
    lines.append(f"  Agent at: ({agent_col}, {agent_row})  Facing: {facing_name}")
    lines.append(f"  Grid size: {grid_width} x {grid_height}")
    lines.append("")

    grid_lines = _render_grid_lines(
        display_grid,
        agent_row=agent_row,
        agent_col=agent_col,
        agent_direction=agent_direction,
    )
    lines.extend(grid_lines)

    return "\n".join(lines)
