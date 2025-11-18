# env/constants.py
from enum import Enum, auto
from typing import List

class Severity(Enum):
    STABLE = 1
    SERIOUS = 2
    CRITICAL = 3

class AgentState(Enum):
    IDLE = "idle"
    CARRYING_VICTIM = "carrying"

class CellType(Enum):
    EMPTY = 0
    OBSTACLE = 1
    SHELTER = 2
    DEPOT = 3

# Actions (0-6)
ACTION_STAY = 0
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_PICKUP = 5
ACTION_DROPOFF = 6

ACTION_NAMES: List[str] = [
    "STAY", "UP", "DOWN", "LEFT", "RIGHT", "PICKUP", "DROPOFF"
]

# Reasonable defaults (use values you had before if different)
DEFAULT_GRID_SIZE = 10
DEFAULT_NUM_AGENTS = 3
DEFAULT_NUM_SHELTERS = 2
DEFAULT_NUM_DEPOTS = 2
DEFAULT_MAX_VICTIMS = 50
DEFAULT_VICTIM_SPAWN_RATE = 0.05
DEFAULT_OBSTACLE_DENSITY = 0.08
DEFAULT_MAX_STEPS = 1000
