"""
Declarative Task Executor and State-Machine Controller for Coordinated Agents.
"""

from typing import List, Dict, Any, Optional, Tuple
from agents.greedy_agent import manhattan_distance
from agents.pathfinding import a_star_search
from env.constants import (
    ACTION_STAY,
    ACTION_UP,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_PICKUP,
    ACTION_DROPOFF,
    ACTION_DELTAS,
    STATE_CARRYING,
    CellType,
)


class TaskExecutor:
    """
    Executes declarative coordination assignments (e.g. 'rescue V2' or 'wait').
    Manages low-level waypoint navigation, pickup, and delivery to shelters.
    """

    def __init__(self, num_agents: int = 3):
        self.num_agents = num_agents
        self._targets: Dict[int, Optional[int]] = {i: None for i in range(num_agents)}
        self._assignments: Dict[int, Optional[str]] = {i: None for i in range(num_agents)}

    def set_assignment(self, agent_id: int, assignment: Optional[str]) -> None:
        """Parses declarative directive into concrete victim target ID."""
        if agent_id < 0 or agent_id >= self.num_agents:
            return

        self._assignments[agent_id] = assignment

        if not assignment or not isinstance(assignment, str):
            self._targets[agent_id] = None
            return

        clean = assignment.strip().lower()
        if clean.startswith("rescue v"):
            try:
                vid_str = clean.replace("rescue v", "").strip()
                self._targets[agent_id] = int(vid_str)
            except ValueError:
                self._targets[agent_id] = None
        else:
            self._targets[agent_id] = None

    def get_assignment(self, agent_id: int) -> Optional[str]:
        return self._assignments.get(agent_id)

    def _get_obstacles(self, obs: Dict[str, Any]) -> set[Tuple[int, int]]:
        """Extracts obstacle coordinates from observation grid."""
        obstacles = set()
        grid_rows = obs.get("grid", [])
        for r_idx, row in enumerate(grid_rows):
            if isinstance(row, str):
                cells = row.split()
                for c_idx, val in enumerate(cells):
                    if val == str(int(CellType.OBSTACLE)):
                        obstacles.add((c_idx, r_idx))
        return obstacles

    def _move_towards(
        self,
        src: Tuple[int, int],
        dst: Tuple[int, int],
        obstacles: set[Tuple[int, int]],
        grid_size: int = 10,
        dynamic_blocked_next: Optional[set[Tuple[int, int]]] = None,
    ) -> int:
        """
        Calculates cardinal step towards dst using A* pathfinding with obstacle avoidance.
        """
        if src == dst:
            return ACTION_STAY

        path = a_star_search(
            src, dst, obstacles, grid_size, dynamic_blocked_next=dynamic_blocked_next
        )
        if not path:
            return ACTION_STAY

        next_pos = path[0]
        dx = next_pos[0] - src[0]
        dy = next_pos[1] - src[1]

        if dx == 1:
            return ACTION_RIGHT
        elif dx == -1:
            return ACTION_LEFT
        elif dy == 1:
            return ACTION_DOWN
        elif dy == -1:
            return ACTION_UP

        return ACTION_STAY

    def act(self, obs: Dict[str, Any]) -> List[int]:
        """Calculates joint actions for all managed agents with inter-agent collision avoidance."""
        actions = []
        agent_infos = obs.get("agents", [])
        obstacles = self._get_obstacles(obs)
        grid_size = obs.get("grid_size", 10)
        reserved_next_positions: set[Tuple[int, int]] = set()

        shelters_data = obs.get("shelters", [])
        shelters_set = {tuple(s["pos"] if isinstance(s, dict) else s) for s in shelters_data}
        depots_data = obs.get("depots", [])
        depots_set = {tuple(d["pos"] if isinstance(d, dict) else d) for d in depots_data}
        hub_cells = shelters_set | depots_set

        # Build active victim lookup map: {id: victim_dict}
        active_victims = {
            v["id"]: v
            for v in obs.get("victims", [])
            if not v.get("rescued", False) and not v.get("died", False)
        }

        for aid in range(self.num_agents):
            agent_info = agent_infos[aid] if aid < len(agent_infos) else {"pos": (0, 0), "carrying": None}
            my_pos = tuple(agent_info["pos"])

            # Check inventory capacity
            inv_ids = agent_info.get("inventory")
            if inv_ids is None:
                inv_ids = [agent_info.get("carrying")] if agent_info.get("carrying") is not None else []
            curr_capacity = len(inv_ids)
            max_cap = agent_info.get("max_capacity", 2)
            is_carrying = curr_capacity > 0 or agent_info.get("state") == STATE_CARRYING

            shelters = obs.get("shelters", [])
            shelter_pos = None
            if shelters:
                nearest_shelter = min(
                    shelters,
                    key=lambda s: manhattan_distance(my_pos, tuple(s["pos"] if isinstance(s, dict) else s)),
                )
                shelter_pos = tuple(nearest_shelter["pos"] if isinstance(nearest_shelter, dict) else nearest_shelter)

            # Check opportunistic adjacent/co-located victim when partially loaded
            opportunistic_target = None
            if 0 < curr_capacity < max_cap and shelter_pos:
                candidates_opp = [
                    v for v in active_victims.values()
                    if v["id"] not in inv_ids and manhattan_distance(my_pos, tuple(v["pos"])) <= 1
                ]
                for cand in candidates_opp:
                    cand_pos = tuple(cand["pos"])
                    detour_steps = manhattan_distance(my_pos, cand_pos) + 1 + manhattan_distance(cand_pos, shelter_pos)
                    safe = True
                    for b_id in inv_ids:
                        b_victim = next((v for v in obs.get("victims", []) if v["id"] == b_id), None)
                        if b_victim:
                            remaining = b_victim.get("max_wait", 40) - b_victim.get("wait_time", 0)
                            if remaining <= detour_steps:
                                safe = False
                                break
                    if safe:
                        opportunistic_target = cand
                        break

            # 1a. Carrying mode with a safe opportunistic pickup
            if is_carrying and opportunistic_target is not None:
                target_pos = tuple(opportunistic_target["pos"])
                if my_pos == target_pos:
                    actions.append(ACTION_PICKUP)
                    act = ACTION_PICKUP
                else:
                    act = self._move_towards(
                        my_pos,
                        target_pos,
                        obstacles,
                        grid_size,
                        dynamic_blocked_next=reserved_next_positions,
                    )
                    actions.append(act)

            # 1b. Carrying mode at max capacity or without safe opportunistic pickup -> Deliver to shelter
            elif is_carrying:
                if not shelters:
                    actions.append(ACTION_STAY)
                    act = ACTION_STAY
                else:
                    if my_pos == shelter_pos:
                        actions.append(ACTION_DROPOFF)
                        self._targets[aid] = None
                        self._assignments[aid] = None
                        act = ACTION_DROPOFF
                    else:
                        act = self._move_towards(
                            my_pos,
                            shelter_pos,
                            obstacles,
                            grid_size,
                            dynamic_blocked_next=reserved_next_positions,
                        )
                        actions.append(act)

            # 2. Idle mode -> Navigate to assigned target victim
            else:
                target_vid = self._targets.get(aid)

                # Edge Case EC-2.2: Target has died, was rescued, or was not set
                if target_vid is None or target_vid not in active_victims:
                    self._targets[aid] = None
                    self._assignments[aid] = None
                    actions.append(ACTION_STAY)
                    act = ACTION_STAY
                else:
                    target_v = active_victims[target_vid]
                    target_pos = tuple(target_v["pos"])

                    if my_pos == target_pos:
                        actions.append(ACTION_PICKUP)
                        act = ACTION_PICKUP
                    else:
                        act = self._move_towards(
                            my_pos,
                            target_pos,
                            obstacles,
                            grid_size,
                            dynamic_blocked_next=reserved_next_positions,
                        )
                        actions.append(act)

            # Track reserved next position for lower-priority agents
            if act in ACTION_DELTAS and act != ACTION_STAY:
                dx, dy = ACTION_DELTAS[act]
                next_cell = (my_pos[0] + dx, my_pos[1] + dy)
            else:
                next_cell = my_pos

            if next_cell not in hub_cells:
                reserved_next_positions.add(next_cell)

        return actions
