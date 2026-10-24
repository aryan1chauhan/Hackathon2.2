"""
A* Pathfinding implementation with dynamic obstacle support for disaster relief agents.
"""

import heapq
from typing import Tuple, List, Set, Optional

Coord = Tuple[int, int]


def manhattan(a: Coord, b: Coord) -> int:
    """Calculates Manhattan distance between two coordinates."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star_search(
    start: Coord,
    goal: Coord,
    obstacles: Set[Coord],
    grid_size: int = 10,
    dynamic_blocked_next: Optional[Set[Coord]] = None,
) -> List[Coord]:
    """
    Returns the shortest path as a list of (x, y) coordinates from start to goal
    (excluding start, including goal). Returns [] if unreachable or start == goal.
    """
    if start == goal:
        return []

    dynamic_blocked = dynamic_blocked_next or set()
    open_heap: List[Tuple[int, int, Coord]] = []
    heapq.heappush(open_heap, (manhattan(start, goal), 0, start))

    came_from: dict[Coord, Coord] = {}
    g_score: dict[Coord, int] = {start: 0}
    deltas = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT

    while open_heap:
        _, current_g, current = heapq.heappop(open_heap)

        if current == goal:
            path: List[Coord] = []
            curr = goal
            while curr in came_from:
                path.append(curr)
                curr = came_from[curr]
            path.reverse()
            return path

        if current_g > g_score.get(current, float("inf")):
            continue

        for dx, dy in deltas:
            nx, ny = current[0] + dx, current[1] + dy
            neighbor = (nx, ny)

            if not (0 <= nx < grid_size and 0 <= ny < grid_size):
                continue
            if neighbor in obstacles and neighbor != goal:
                continue
            # Avoid colliding with another agent on the immediate next step
            if current == start and neighbor in dynamic_blocked:
                continue

            tentative_g = current_g + 1
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + manhattan(neighbor, goal)
                heapq.heappush(open_heap, (f_score, tentative_g, neighbor))

    return []


if __name__ == "__main__":
    start_pos: Coord = (0, 0)
    goal_pos: Coord = (4, 4)
    wall: Set[Coord] = {(1, 0), (1, 1), (1, 2), (2, 2), (3, 2)}

    print(f"Running A* Search: start={start_pos}, goal={goal_pos}")
    print(f"Obstacles: {sorted(list(wall))}")

    result_path = a_star_search(start_pos, goal_pos, obstacles=wall, grid_size=10)
    print(f"Path found ({len(result_path)} steps): {result_path}")

