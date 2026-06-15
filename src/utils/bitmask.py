"""
Canonical bitmask stringifier for the 7x7 MiniGrid agent view.

The agent sits at local position (3, 6) in the 7x7 grid, facing "north"
(toward row 0). Coordinates are reported relative to the agent:
  - "forward" = decreasing j  (toward row 0)
  - "right"   = increasing i  (toward col 6)
"""

import numpy as np

from src.utils.minigrid_maps import OBJECT_TO_STR, COLOR_TO_STR


def stringify_bitmask(img: np.ndarray) -> str:
    """Turn a 7x7x3 observation image into a human-readable text description.

    Returns one line per visible object, e.g.
        "a red ball 2 steps forward and 1 step right"
    or "nothing of interest" when the view is empty.
    """
    desc: list[str] = []

    for i in range(7):
        for j in range(7):
            obj_idx = img[i, j, 0]
            if obj_idx <= 1:
                continue

            color = COLOR_TO_STR.get(img[i, j, 1], "none")
            obj = OBJECT_TO_STR.get(obj_idx, "unknown")

            dx = i - 3
            dy = 6 - j

            if dx == 0 and dy == 0:
                continue

            parts: list[str] = []
            if dy > 0:
                parts.append(f"{dy} step{'s' if dy > 1 else ''} forward")
            if dx != 0:
                side = "right" if dx > 0 else "left"
                parts.append(f"{abs(dx)} step{'s' if abs(dx) > 1 else ''} {side}")

            pos_str = " and ".join(parts) if parts else "at your location"
            desc.append(f"a {color} {obj} {pos_str}")

    return "\n".join(desc) if desc else "nothing of interest"
