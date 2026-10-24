"""
Core Simulation Environment for Multi-Agent Disaster Relief Resource Allocation.
"""

import random
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from env.constants import (
    ACTION_STAY,
    ACTION_UP,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_PICKUP,
    ACTION_DROPOFF,
    ACTION_DELTAS,
    CellType,
    Severity,
    SEVERITY_NAMES,
    DEFAULT_MAX_WAITS,
    STATE_IDLE,
    STATE_CARRYING,
)
from env.spawn_policies import SpawnPolicy, DefaultSpawnPolicy


class Victim:
    """Represents an injured victim on the disaster grid."""

    def __init__(
        self,
        victim_id: int,
        pos: Tuple[int, int],
        severity: int = Severity.STABLE,
        max_wait: Optional[int] = None,
    ):
        self.id = victim_id
        self.pos = pos
        self.severity = int(severity)
        self.severity_name = SEVERITY_NAMES.get(self.severity, "UNKNOWN")
        self.wait_time = 0
        self.max_wait = max_wait if max_wait is not None else DEFAULT_MAX_WAITS.get(self.severity, 40)
        self.rescued = False
        self.died = False

    def urgency(self) -> float:
        """
        Calculates dynamic urgency score.
        Formula: severity * (1.0 + wait_time / max(1, max_wait - wait_time))
        Returns 0.0 if rescued or deceased.
        """
        if self.rescued or self.died:
            return 0.0
        denominator = max(1, self.max_wait - self.wait_time)
        return float(self.severity * (1.0 + (self.wait_time / denominator)))

    def deteriorate(self, carried: bool = False) -> bool:
        """
        Advances wait_time if not rescued and not carried in transit.
        Returns True if victim died at this exact timestep.
        """
        if self.rescued or self.died or carried:
            return False

        self.wait_time += 1
        if self.wait_time >= self.max_wait:
            self.died = True
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "pos": list(self.pos),
            "severity": self.severity,
            "severity_name": self.severity_name,
            "wait_time": self.wait_time,
            "max_wait": self.max_wait,
            "urgency": float(round(self.urgency(), 2)),
            "rescued": self.rescued,
            "died": self.died,
        }


class Agent:
    """Represents a rescue unit operating on the grid with multi-victim inventory."""

    def __init__(self, agent_id: int, pos: Tuple[int, int], max_capacity: int = 2):
        self.id = agent_id
        self.pos = pos
        self.max_capacity = max_capacity
        self.inventory: List[Victim] = []
        self.state = STATE_IDLE
        self.total_rescues = 0
        self.total_moves = 0
        self.assignment: Optional[str] = None

    @property
    def carrying_victim(self) -> Optional[Victim]:
        """Backward compatibility: returns first victim in inventory or None."""
        return self.inventory[0] if self.inventory else None

    @carrying_victim.setter
    def carrying_victim(self, val: Optional[Victim]) -> None:
        """Backward compatibility: clears or resets inventory with single victim."""
        if val is None:
            self.inventory = []
        else:
            self.inventory = [val]

    @property
    def carrying_victim_id(self) -> Optional[int]:
        """Dynamically returns the ID of the first carried victim or None."""
        return self.inventory[0].id if self.inventory else None

    @property
    def available(self) -> bool:
        """Returns True if agent is idle and not carrying any victim."""
        return len(self.inventory) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "pos": list(self.pos),
            "state": self.state,
            "carrying": self.carrying_victim_id,
            "inventory": [v.id for v in self.inventory],
            "max_capacity": self.max_capacity,
            "available": self.available,
            "total_rescues": self.total_rescues,
            "total_moves": self.total_moves,
            "assignment": self.assignment,
        }


class DisasterReliefEnv:
    """
    10x10 Discrete Grid Disaster Relief Multi-Agent Environment.
    """

    def __init__(
        self,
        grid_size: int = 10,
        num_agents: int = 3,
        num_shelters: int = 2,
        num_depots: int = 2,
        max_victims: int = 6,
        seed: Optional[int] = None,
        spawn_policy: Optional[SpawnPolicy] = None,
    ):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_shelters = num_shelters
        self.num_depots = num_depots
        self.max_victims = max_victims
        self.seed_val = seed
        self.spawn_policy = spawn_policy if spawn_policy is not None else DefaultSpawnPolicy()

        if self.seed_val is not None:
            random.seed(self.seed_val)
            np.random.seed(self.seed_val)

        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.agents: List[Agent] = []
        self.victims: List[Victim] = []
        self.shelters: List[Tuple[int, int]] = []
        self.depots: List[Tuple[int, int]] = []
        self.obstacles: set[Tuple[int, int]] = set()

        self.total_rescued = 0
        self.total_died = 0
        self.step_count = 0
        self.victim_counter = 0
        self.rescued_wait_times: List[int] = []

        self.reset(seed=self.seed_val)

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Resets the environment to initial state."""
        if seed is not None:
            self.seed_val = seed
            random.seed(seed)
            np.random.seed(seed)

        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.total_rescued = 0
        self.total_died = 0
        self.step_count = 0
        self.victim_counter = 0
        self.rescued_wait_times = []
        self.obstacles = set()
        self.shelters = []
        self.depots = []
        self.agents = []
        self.victims = []

        # 1. Place fixed or deterministic shelters and depots
        # Shelters placed symmetrically near top-left and bottom-right
        self.shelters = [(1, 1), (self.grid_size - 2, self.grid_size - 2)]
        for s in self.shelters:
            self.grid[s[1], s[0]] = CellType.SHELTER

        # Depots placed near staging points
        self.depots = [(0, 0), (self.grid_size - 1, 0)]
        for d in self.depots:
            self.grid[d[1], d[0]] = CellType.DEPOT

        # 2. Place static obstacle cells (rubble / damaged roads)
        # Avoid blocking depots and shelters
        candidate_obstacles = [
            (3, 1), (3, 2), (3, 3),
            (6, 6), (6, 7), (6, 8),
            (2, 5), (7, 4)
        ]
        for obs in candidate_obstacles:
            if 0 <= obs[0] < self.grid_size and 0 <= obs[1] < self.grid_size:
                if obs not in self.shelters and obs not in self.depots:
                    self.obstacles.add(obs)
                    self.grid[obs[1], obs[0]] = CellType.OBSTACLE

        # 3. Initialize agents at depots
        for i in range(self.num_agents):
            depot_pos = self.depots[i % len(self.depots)]
            agent = Agent(agent_id=i, pos=depot_pos)
            self.agents.append(agent)

        # 4. Spawn initial batch of victims
        initial_victim_count = min(3, self.max_victims)
        for _ in range(initial_victim_count):
            self._spawn_victim()

        return self.get_observation()

    def get_random_empty_cell(self) -> Optional[Tuple[int, int]]:
        """Finds an unoccupied, non-obstacle, non-shelter, non-depot grid cell."""
        occupied = set(self.obstacles)
        occupied.update(self.shelters)
        occupied.update(self.depots)
        occupied.update(v.pos for v in self.victims if not v.rescued and not v.died)

        candidates = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in occupied
        ]
        return random.choice(candidates) if candidates else None

    def _spawn_victim(
        self,
        pos: Optional[Tuple[int, int]] = None,
        severity: Optional[int] = None,
        max_wait: Optional[int] = None,
    ) -> Optional[Victim]:
        """Spawns a new victim on the grid."""
        if pos is None:
            pos = self.get_random_empty_cell()
            if pos is None:
                return None

        if severity is None:
            # Stochastic distribution: 40% Stable, 40% Serious, 20% Critical
            r = random.random()
            if r < 0.40:
                severity = Severity.STABLE
            elif r < 0.80:
                severity = Severity.SERIOUS
            else:
                severity = Severity.CRITICAL

        self.victim_counter += 1
        victim = Victim(
            victim_id=self.victim_counter,
            pos=pos,
            severity=severity,
            max_wait=max_wait,
        )
        self.victims.append(victim)
        return victim

    def step(self, actions: List[int]) -> Tuple[Dict[str, Any], List[float], bool, Dict[str, Any]]:
        """
        Advances the simulation by one timestep using 7 discrete actions.
        Returns: (observation, rewards, done, info)
        """
        self.step_count += 1
        new_rescues = 0
        new_deaths = 0
        rewards = [0.0] * self.num_agents

        hub_cells = set(self.shelters) | set(self.depots)
        committed_positions: set[Tuple[int, int]] = set()

        # Map actions to agents
        for i, agent in enumerate(self.agents):
            action = actions[i] if i < len(actions) else ACTION_STAY
            curr_pos = agent.pos

            # --- Physical Movement [1, 2, 3, 4] ---
            if action in ACTION_DELTAS and action != ACTION_STAY:
                dx, dy = ACTION_DELTAS[action]
                nx = curr_pos[0] + dx
                ny = curr_pos[1] + dy

                # Boundary check
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    # Obstacle collision check
                    if (nx, ny) not in self.obstacles:
                        target_pos = (nx, ny)
                        collision = False
                        if target_pos not in hub_cells:
                            # 1. Collision with lower-index agent's committed move
                            if target_pos in committed_positions:
                                collision = True
                            # 2. Collision with higher-index agent currently at target_pos
                            elif any(self.agents[j].pos == target_pos for j in range(i + 1, len(self.agents))):
                                collision = True

                        if not collision:
                            agent.pos = target_pos
                            agent.total_moves += 1
                            rewards[i] -= 0.05  # Slight energy/time movement penalty

                            # If carrying victim(s), update victim positions
                            carrying = getattr(agent, "inventory", [agent.carrying_victim] if agent.carrying_victim else [])
                            for v in carrying:
                                if v is not None:
                                    v.pos = agent.pos

            # --- Pickup Action [5] ---
            elif action == ACTION_PICKUP:
                if len(agent.inventory) < agent.max_capacity:
                    # Locate active victim at current agent cell
                    candidates = [
                        v for v in self.victims
                        if v.pos == agent.pos and not v.rescued and not v.died
                    ]
                    all_carried_vids = {v.id for a in self.agents for v in a.inventory}
                    candidates = [v for v in candidates if v.id not in all_carried_vids]
                    if candidates:
                        # Pick up the most critical victim if multiple exist
                        target_v = max(candidates, key=lambda v: v.urgency())
                        agent.inventory.append(target_v)
                        agent.state = STATE_CARRYING
                        rewards[i] += 2.0  # Reward successful pickup

            # --- Dropoff Action [6] ---
            elif action == ACTION_DROPOFF:
                if agent.inventory and agent.pos in self.shelters:
                    # Discharge all boarded victims at the shelter
                    for v in agent.inventory:
                        v.rescued = True
                        agent.total_rescues += 1
                        self.total_rescued += 1
                        new_rescues += 1
                        self.rescued_wait_times.append(v.wait_time)
                        rewards[i] += 10.0  # Significant delivery reward
                    agent.inventory = []
                    agent.state = STATE_IDLE

            # Track committed position for collision prevention
            if agent.pos not in hub_cells:
                committed_positions.add(agent.pos)

        # --- Victim Health Deterioration ---
        # Note: executed after actions so simultaneous pickup avoids decay
        carried_vids = {
            v.id for a in self.agents for v in a.inventory
        }

        for v in self.victims:
            is_carried = v.id in carried_vids
            died = v.deteriorate(carried=is_carried)
            if died:
                self.total_died += 1
                new_deaths += 1
                # Penalty across all agents for victim expiration
                for r_idx in range(len(rewards)):
                    rewards[r_idx] -= 5.0

        # --- Dynamic Victim Spawning ---
        if self.spawn_policy:
            new_spawns = self.spawn_policy.spawn(self)
            for spawn_pos in new_spawns:
                self._spawn_victim(pos=spawn_pos)

        # --- Terminal Done Condition ---
        active_victims = [v for v in self.victims if not v.rescued and not v.died]
        done = (len(active_victims) == 0 and len(self.victims) >= self.max_victims) or (self.step_count >= 300)

        info = {
            "step": self.step_count,
            "new_rescues": new_rescues,
            "new_deaths": new_deaths,
            "total_rescued": self.total_rescued,
            "total_died": self.total_died,
        }

        return self.get_observation(), rewards, done, info

    def get_observation(self) -> Dict[str, Any]:
        """Packages complete world state into structured JSON-compatible dictionary."""
        # Convert grid into string rows for lightweight serialization
        grid_serialized = [
            " ".join(str(int(self.grid[r, c])) for c in range(self.grid_size))
            for r in range(self.grid_size)
        ]

        active_victims = [
            v.to_dict() for v in self.victims if not v.rescued and not v.died
        ]

        shelters_data = [{"id": idx, "pos": list(s)} for idx, s in enumerate(self.shelters)]
        depots_data = [{"id": idx, "pos": list(d)} for idx, d in enumerate(self.depots)]
        agents_data = [a.to_dict() for a in self.agents]

        assignments = [
            {"agent_id": a.id, "task": a.assignment} for a in self.agents if a.assignment
        ]

        return {
            "grid_size": self.grid_size,
            "step": self.step_count,
            "grid": grid_serialized,
            "agents": agents_data,
            "victims": active_victims,
            "shelters": shelters_data,
            "depots": depots_data,
            "assignments": assignments,
        }
