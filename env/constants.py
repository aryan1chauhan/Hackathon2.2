"""
Constants and enumerations for the Disaster Relief simulation environment.
"""

from enum import IntEnum

# Action primitives
ACTION_STAY = 0
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_PICKUP = 5
ACTION_DROPOFF = 6

ACTION_NAMES = {
    ACTION_STAY: "STAY",
    ACTION_UP: "UP",
    ACTION_DOWN: "DOWN",
    ACTION_LEFT: "LEFT",
    ACTION_RIGHT: "RIGHT",
    ACTION_PICKUP: "PICKUP",
    ACTION_DROPOFF: "DROPOFF",
}

# Movement directional offsets in (dx, dy)
# where x is horizontal column [0..width-1] and y is vertical row [0..height-1]
ACTION_DELTAS = {
    ACTION_STAY: (0, 0),
    ACTION_UP: (0, -1),
    ACTION_DOWN: (0, 1),
    ACTION_LEFT: (-1, 0),
    ACTION_RIGHT: (1, 0),
}


class CellType(IntEnum):
    EMPTY = 0
    OBSTACLE = 1
    SHELTER = 2
    DEPOT = 3


class Severity(IntEnum):
    STABLE = 1
    SERIOUS = 2
    CRITICAL = 3


SEVERITY_NAMES = {
    Severity.STABLE: "STABLE",
    Severity.SERIOUS: "SERIOUS",
    Severity.CRITICAL: "CRITICAL",
}

DEFAULT_MAX_WAITS = {
    Severity.STABLE: 60,
    Severity.SERIOUS: 40,
    Severity.CRITICAL: 25,
}

# Agent states
STATE_IDLE = "idle"
STATE_CARRYING = "carrying"
