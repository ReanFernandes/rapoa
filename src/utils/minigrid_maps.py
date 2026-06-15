"""
Canonical MiniGrid object, color, and state mappings.

These match the official MiniGrid/BabyAI encoding:
  image[:, :, 0] = object type
  image[:, :, 1] = color
  image[:, :, 2] = state (for doors)
"""

OBJECT_TO_STR = {
    0: "unseen",
    1: "empty",
    2: "wall",
    3: "floor",
    4: "door",
    5: "key",
    6: "ball",
    7: "box",
    8: "goal",
    9: "lava",
    10: "agent",
}

COLOR_TO_STR = {
    0: "red",
    1: "green",
    2: "blue",
    3: "purple",
    4: "yellow",
    5: "grey",
}

STATE_TO_STR = {
    0: "open",
    1: "closed",
    2: "locked",
}
