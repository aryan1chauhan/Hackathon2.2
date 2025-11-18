import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from env.constants import (
    Severity, AgentState, CellType,
    ACTION_STAY, ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT,
    ACTION_PICKUP, ACTION_DROPOFF, ACTION_NAMES,
    DEFAULT_GRID_SIZE, DEFAULT_NUM_AGENTS, DEFAULT_NUM_SHELTERS,
    DEFAULT_NUM_DEPOTS, DEFAULT_MAX_VICTIMS, DEFAULT_VICTIM_SPAWN_RATE,
    DEFAULT_OBSTACLE_DENSITY, DEFAULT_MAX_STEPS
)
from env.spawn_policies import SpawnPolicy, RandomSpawnPolicy


@dataclass
class Victim:
    """Represents a victim needing rescue"""
    id: int
    pos: Tuple[int, int]
    severity: Severity
    spawn_time: int
    wait_time: int = 0
    max_wait: int = 100
    rescued: bool = False
    rescued_time: Optional[int] = None
    rescued_by: Optional[int] = None  # Agent ID
    died: bool = False

    def deteriorate(self) -> bool:
        """Victim condition worsens over time. Returns True if died."""
        if not self.rescued and not self.died:
            self.wait_time += 1
            if self.wait_time >= self.max_wait:
                self.died = True
                return True
        return False

    def urgency(self) -> float:
        """Calculate urgency (0-10 scale)"""
        severity_weight = self.severity.value
        time_factor = min(1.0, self.wait_time / self.max_wait)
        return severity_weight * 2 + time_factor * 4  # Max score ~10


@dataclass
class Agent:
    """Represents a rescue agent (ambulance/drone)"""
    id: int
    pos: Tuple[int, int]
    state: AgentState = AgentState.IDLE
    carrying_victim: Optional[Victim] = None
    target_pos: Optional[Tuple[int, int]] = None
    capacity: int = 1
    speed: int = 1
    total_rescues: int = 0
    total_moves: int = 0
    path_history: List[Tuple[int, int]] = field(default_factory=list)

    def __post_init__(self):
        self.path_history = [self.pos]

    def is_available(self) -> bool:
        """Check if agent can take new assignments"""
        return self.state == AgentState.IDLE and self.carrying_victim is None

    def distance_to(self, pos: Tuple[int, int]) -> int:
        """Manhattan distance to position"""
        return abs(self.pos[0] - pos[0]) + abs(self.pos[1] - pos[1])


class DisasterReliefEnv:
    """
    Gym-style disaster relief simulation environment.

    Observation: Complete state dict with grid, agents, victims, shelters
    Actions: List of action IDs (0-6) for each agent
    Rewards: Per-agent rewards based on rescue success and efficiency
    """

    def __init__(
        self,
        grid_size: int = DEFAULT_GRID_SIZE,
        num_agents: int = DEFAULT_NUM_AGENTS,
        num_shelters: int = DEFAULT_NUM_SHELTERS,
        num_depots: int = DEFAULT_NUM_DEPOTS,
        max_victims: int = DEFAULT_MAX_VICTIMS,
        spawn_policy: Optional[SpawnPolicy] = None,
        obstacle_density: float = DEFAULT_OBSTACLE_DENSITY,
        max_steps: int = DEFAULT_MAX_STEPS,
        seed: Optional[int] = None
    ):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_shelters = num_shelters
        self.num_depots = num_depots
        self.max_victims = max_victims
        self.obstacle_density = obstacle_density
        self.max_steps = max_steps

        # Spawn policy
        self.spawn_policy = spawn_policy or RandomSpawnPolicy(DEFAULT_VICTIM_SPAWN_RATE, seed)

        # Random seed
        self.seed = seed
        self._random = random.Random(seed)
        self._np_random = np.random.default_rng(seed)

        # World state (grid may be None until reset)
        self.grid: Optional[np.ndarray] = None
        self.agents: List[Agent] = []
        self.victims: List[Victim] = []
        self.shelters: List[Tuple[int, int]] = []
        self.depots: List[Tuple[int, int]] = []

        # Statistics
        self.current_step = 0
        self.total_spawned = 0
        self.total_rescued = 0
        self.total_died = 0
        self.victims_by_severity = {s: 0 for s in Severity}
        self.episode_rewards: List[float] = []
        # Metrics tracking
        self.rescued_wait_times: List[int] = []  # Wait times for rescued victims
        self.all_wait_times: List[int] = []  # Final wait times for all victims (rescued, died, or at end)
        self.agent_workloads: Dict[int, int] = {}  # Agent ID -> number of rescues

        # History for visualization
        self.state_history: List[Dict] = []

        # Initialize
        self.reset()

    def reset(self) -> Dict:
        """Reset environment to initial state"""
        self.current_step = 0
        self.total_spawned = 0
        self.total_rescued = 0
        self.total_died = 0
        self.victims_by_severity = {s: 0 for s in Severity}
        self.episode_rewards = []
        self.state_history = []
        # Reset metrics
        self.rescued_wait_times = []
        self.all_wait_times = []
        self.agent_workloads = {}

        # 1) Initialize the grid first (prevents None indexing)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # 2) Place obstacles on the now-initialized grid
        self._place_obstacles()

        # 3) Place shelters
        self.shelters = []
        for _ in range(self.num_shelters):
            pos = self._find_empty_position()
            self.shelters.append(pos)
            self.grid[pos[1], pos[0]] = CellType.SHELTER.value

        # 4) Place depots
        self.depots = []
        for _ in range(self.num_depots):
            pos = self._find_empty_position()
            self.depots.append(pos)
            self.grid[pos[1], pos[0]] = CellType.DEPOT.value

        # 5) Initialize agents at/near depots
        self.agents = []
        for i in range(self.num_agents):
            # If no depots exist (edge case), place agent randomly
            if len(self.depots) == 0:
                agent_pos = self._find_empty_position()
            else:
                depot_pos = self.depots[i % len(self.depots)]
                agent_pos = self._find_position_near(depot_pos, max_dist=1)
            agent = Agent(id=i, pos=agent_pos)
            self.agents.append(agent)

        # 6) Initialize victims (start with a few)
        self.victims = []
        for _ in range(min(3, self.max_victims)):
            self._spawn_victim()

        # Save initial state and return observation
        self._save_state()
        return self.get_observation()

    def step(self, actions: List[int]) -> Tuple[Dict, List[float], bool, Dict]:
        """
        Execute one timestep.

        Args:
            actions: List of action IDs (0-6), one per agent

        Returns:
            observation: Current state dict
            rewards: List of rewards per agent
            done: Episode termination flag
            info: Additional information dict
        """
        if len(actions) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} actions, got {len(actions)}")

        self.current_step += 1
        step_rewards = [0.0] * self.num_agents

        # Execute agent actions
        for i, (agent, action) in enumerate(zip(self.agents, actions)):
            reward = self._execute_action(agent, action)
            step_rewards[i] = reward

        # Update victims (deterioration)
        for victim in self.victims:
            if victim.deteriorate():
                self.total_died += 1
                # Track final wait time for died victims
                self.all_wait_times.append(victim.wait_time)

        # Spawn new victims if policy allows
        active_victims = len([v for v in self.victims if not v.rescued and not v.died])
        if self.spawn_policy.should_spawn(self.current_step, active_victims, self.max_victims):
            self._spawn_victim()

        # Check termination
        done = self.current_step >= self.max_steps

        # If episode ends, track final wait times for remaining victims
        if done:
            for victim in self.victims:
                if not victim.rescued and not victim.died:
                    self.all_wait_times.append(victim.wait_time)

        # Info dict
        info = {
            "step": self.current_step,
            "total_spawned": self.total_spawned,
            "total_rescued": self.total_rescued,
            "total_died": self.total_died,
            "active_victims": active_victims,
            "success_rate": self.total_rescued / max(1, self.total_rescued + self.total_died),
            "agents_idle": sum(1 for a in self.agents if a.is_available()),
            "agents_carrying": sum(1 for a in self.agents if a.carrying_victim is not None)
        }

        self.episode_rewards.append(sum(step_rewards))
        self._save_state()

        return self.get_observation(), step_rewards, done, info

    def _execute_action(self, agent: Agent, action: int) -> float:
        """Execute single agent action and return reward"""
        reward = -0.01  # Small step penalty

        # Movement actions
        if action == ACTION_UP:
            new_pos = (agent.pos[0], max(0, agent.pos[1] - 1))
            if self._is_valid_position(new_pos):
                agent.pos = new_pos
                agent.total_moves += 1
                agent.path_history.append(agent.pos)

        elif action == ACTION_DOWN:
            new_pos = (agent.pos[0], min(self.grid_size - 1, agent.pos[1] + 1))
            if self._is_valid_position(new_pos):
                agent.pos = new_pos
                agent.total_moves += 1
                agent.path_history.append(agent.pos)

        elif action == ACTION_LEFT:
            new_pos = (max(0, agent.pos[0] - 1), agent.pos[1])
            if self._is_valid_position(new_pos):
                agent.pos = new_pos
                agent.total_moves += 1
                agent.path_history.append(agent.pos)

        elif action == ACTION_RIGHT:
            new_pos = (min(self.grid_size - 1, agent.pos[0] + 1), agent.pos[1])
            if self._is_valid_position(new_pos):
                agent.pos = new_pos
                agent.total_moves += 1
                agent.path_history.append(agent.pos)

        elif action == ACTION_PICKUP:
            if agent.carrying_victim is None:
                victim = self._get_victim_at(agent.pos)
                if victim and not victim.rescued and not victim.died:
                    agent.carrying_victim = victim
                    agent.state = AgentState.CARRYING_VICTIM
                    reward += victim.urgency() * 2  # Reward for picking up

        elif action == ACTION_DROPOFF:
            if agent.carrying_victim is not None:
                if agent.pos in self.shelters:
                    victim = agent.carrying_victim
                    victim.rescued = True
                    victim.rescued_time = self.current_step
                    victim.rescued_by = agent.id
                    agent.carrying_victim = None
                    agent.state = AgentState.IDLE
                    agent.total_rescues += 1
                    self.total_rescued += 1

                    # Track metrics
                    self.rescued_wait_times.append(victim.wait_time)
                    self.all_wait_times.append(victim.wait_time)
                    if agent.id not in self.agent_workloads:
                        self.agent_workloads[agent.id] = 0
                    self.agent_workloads[agent.id] += 1

                    # Reward calculation
                    time_bonus = max(0, victim.max_wait - victim.wait_time) * 0.5
                    severity_bonus = victim.severity.value * 10
                    reward += 50 + severity_bonus + time_bonus

        return reward

    def _spawn_victim(self):
        """Spawn a new victim using spawn policy"""
        pos = self._find_empty_position()
        severity, max_wait = self.spawn_policy.get_spawn_params(self.current_step)

        victim = Victim(
            id=self.total_spawned,
            pos=pos,
            severity=severity,
            spawn_time=self.current_step,
            max_wait=max_wait
        )

        self.victims.append(victim)
        self.victims_by_severity[severity] += 1
        self.total_spawned += 1

    def _place_obstacles(self):
        """Place random obstacles on grid"""
        if self.grid is None:
            raise RuntimeError("_place_obstacles called before grid initialization")
        num_obstacles = int(self.grid_size * self.grid_size * self.obstacle_density)
        for _ in range(num_obstacles):
            x = self._random.randint(0, self.grid_size - 1)
            y = self._random.randint(0, self.grid_size - 1)
            # avoid overwriting shelters/depots if already set
            if self.grid[y, x] == 0:
                self.grid[y, x] = CellType.OBSTACLE.value

    def _find_empty_position(self) -> Tuple[int, int]:
        """Find random empty position"""
        max_attempts = 100
        for _ in range(max_attempts):
            x = self._random.randint(0, self.grid_size - 1)
            y = self._random.randint(0, self.grid_size - 1)
            if self._is_valid_position((x, y)) and not self._position_occupied((x, y)):
                return (x, y)
        # Fallback: return any valid position
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self._is_valid_position((x, y)) and not self._position_occupied((x, y)):
                    return (x, y)
        return (0, 0)  # Last resort

    def _find_position_near(self, pos: Tuple[int, int], max_dist: int = 2) -> Tuple[int, int]:
        """Find empty position near given position"""
        for _ in range(20):
            dx = self._random.randint(-max_dist, max_dist)
            dy = self._random.randint(-max_dist, max_dist)
            new_pos = (
                max(0, min(self.grid_size - 1, pos[0] + dx)),
                max(0, min(self.grid_size - 1, pos[1] + dy))
            )
            if self._is_valid_position(new_pos) and not self._position_occupied(new_pos):
                return new_pos
        return pos

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is valid (not obstacle, in bounds)"""
        x, y = pos
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return False
        if self.grid is None:
            return False
        # treat non-zero cells as occupied by obstacles/shelters/depots
        return self.grid[y, x] != CellType.OBSTACLE.value

    def _position_occupied(self, pos: Tuple[int, int]) -> bool:
        """Check if position has agent or victim"""
        for agent in self.agents:
            if agent.pos == pos:
                return True
        for victim in self.victims:
            if victim.pos == pos and not victim.rescued and not victim.died:
                return True
        # consider obstacles/shelters/depots as occupied for spawn purposes
        if self.grid is not None:
            x, y = pos
            if self.grid[y, x] != 0 and self.grid[y, x] not in (CellType.SHELTER.value, CellType.DEPOT.value):
                return True
        return False

    def _get_victim_at(self, pos: Tuple[int, int]) -> Optional[Victim]:
        """Get victim at position"""
        for victim in self.victims:
            if victim.pos == pos and not victim.rescued and not victim.died:
                return victim
        return None

    def get_observation(self) -> Dict:
        """Get complete state observation"""
        return {
            "grid": self.grid.copy() if self.grid is not None else None,
            "grid_size": self.grid_size,
            "step": self.current_step,
            "agents": [
                {
                    "id": a.id,
                    "pos": a.pos,
                    "state": a.state.value,
                    "carrying": a.carrying_victim.id if a.carrying_victim else None,
                    "available": a.is_available(),
                    "total_rescues": a.total_rescues,
                    "total_moves": a.total_moves
                }
                for a in self.agents
            ],
            "victims": [
                {
                    "id": v.id,
                    "pos": v.pos,
                    "severity": v.severity.value,
                    "severity_name": v.severity.name,
                    "wait_time": v.wait_time,
                    "max_wait": v.max_wait,
                    "urgency": v.urgency(),
                    "rescued": v.rescued,
                    "died": v.died
                }
                for v in self.victims if not v.rescued and not v.died
            ],
            "shelters": self.shelters,
            "depots": self.depots
        }

    def get_state_summary(self) -> str:
        """
        Get human-readable state summary for LLM.
        This will be used by the coordinator.
        """
        obs = self.get_observation()

        summary = f"=== Disaster Relief Situation (Step {self.current_step}) ===\n\n"

        # Agents status
        summary += "AGENTS:\n"
        for agent_info in obs['agents']:
            status = "IDLE" if agent_info['available'] else agent_info['state'].upper()
            carrying = f" (carrying V{agent_info['carrying']})" if agent_info['carrying'] is not None else ""
            summary += f"  Agent {agent_info['id']}: pos={agent_info['pos']}, {status}{carrying}\n"

        summary += "\n"

        # Victims status
        summary += "VICTIMS:\n"
        if not obs['victims']:
            summary += "  No active victims\n"
        else:
            # Sort by urgency
            victims_sorted = sorted(obs['victims'], key=lambda v: -v['urgency'])
            for victim in victims_sorted:
                summary += (f"  Victim {victim['id']}: pos={victim['pos']}, "
                            f"severity={victim['severity_name']}, "
                            f"urgency={victim['urgency']:.1f}, "
                            f"wait={victim['wait_time']}/{victim['max_wait']}\n")

        summary += "\n"

        # Shelters and depots
        summary += f"SHELTERS: {obs['shelters']}\n"
        summary += f"DEPOTS: {obs['depots']}\n"

        # Statistics
        summary += f"\nSTATS: Rescued={self.total_rescued}, Died={self.total_died}, Active={len(obs['victims'])}\n"

        return summary

    def _save_state(self):
        """Save state for history/replay"""
        self.state_history.append({
            "step": self.current_step,
            "observation": self.get_observation(),
            "stats": {
                "rescued": self.total_rescued,
                "died": self.total_died,
                "active": len([v for v in self.victims if not v.rescued and not v.died])
            }
        })

    def render(self, save_path: Optional[str] = None, show_paths: bool = False):
        """Visualize current state"""
        if self.grid is None:
            raise RuntimeError("render called before grid initialization")

        fig, ax = plt.subplots(figsize=(10, 10))

        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(self.grid_size))
        ax.set_yticks(range(self.grid_size))

        # Draw obstacles
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.grid[y, x] == CellType.OBSTACLE.value:
                    rect = patches.Rectangle((x - 0.4, y - 0.4), 0.8, 0.8,
                                             linewidth=1, edgecolor='black',
                                             facecolor='gray')
                    ax.add_patch(rect)

        # Draw shelters
        for sx, sy in self.shelters:
            rect = patches.Rectangle((sx - 0.4, sy - 0.4), 0.8, 0.8,
                                     linewidth=2, edgecolor='green',
                                     facecolor='lightgreen', alpha=0.7)
            ax.add_patch(rect)
            ax.text(sx, sy, 'S', ha='center', va='center',
                    fontsize=14, fontweight='bold')

        # Draw depots
        for dx, dy in self.depots:
            rect = patches.Rectangle((dx - 0.4, dy - 0.4), 0.8, 0.8,
                                     linewidth=2, edgecolor='blue',
                                     facecolor='lightblue', alpha=0.7)
            ax.add_patch(rect)
            ax.text(dx, dy, 'D', ha='center', va='center',
                    fontsize=14, fontweight='bold')

        # Draw victims
        severity_colors = {
            Severity.CRITICAL: 'red',
            Severity.SERIOUS: 'orange',
            Severity.STABLE: 'yellow'
        }
        for victim in self.victims:
            if not victim.rescued and not victim.died:
                vx, vy = victim.pos
                color = severity_colors[victim.severity]
                circle = patches.Circle((vx, vy), 0.3, linewidth=1,
                                        edgecolor='black', facecolor=color, alpha=0.8)
                ax.add_patch(circle)
                ax.text(vx, vy - 0.5, f'{victim.wait_time}', ha='center',
                        va='top', fontsize=8)

        # Draw agents and paths (single correct block)
        agent_colors = ['blue', 'purple', 'brown', 'pink', 'cyan']
        for agent in self.agents:
            color = agent_colors[agent.id % len(agent_colors)]

            # Draw path
            if show_paths and len(agent.path_history) > 1:
                path_x = [p[0] for p in agent.path_history[-20:]]
                path_y = [p[1] for p in agent.path_history[-20:]]
                ax.plot(path_x, path_y, '-', alpha=0.3, linewidth=1)

            # Draw agent (unpack position correctly)
            ax_pos, ay_pos = agent.pos
            agent_circle = patches.Circle((ax_pos, ay_pos), 0.35,
                                          linewidth=1, edgecolor='black',
                                          facecolor=color, zorder=5)
            ax.add_patch(agent_circle)
            ax.text(ax_pos, ay_pos - 0.45, f'A{agent.id}', ha='center', va='top',
                    fontsize=9, fontweight='bold', zorder=6)

            # If agent is carrying a victim, mark it
            if agent.carrying_victim is not None:
                ax.text(ax_pos, ay_pos + 0.45, 'V', ha='center', va='bottom',
                        fontsize=9, fontweight='bold', color='black', zorder=6)

        # Optional: save or show
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
