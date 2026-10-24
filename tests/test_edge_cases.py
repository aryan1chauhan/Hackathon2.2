"""
Phase 6: Edge Case Hardening & Regression Suite.
Exhaustively tests all 8 critical edge cases and failure modes defined in docs/EDGE_CASES.md.
"""

import os
import sys
import pytest
import numpy as np

# Ensure local module priority
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import DisasterReliefEnv, CellType, Severity
from env.constants import (
    ACTION_STAY,
    ACTION_UP,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_PICKUP,
    ACTION_DROPOFF,
    STATE_IDLE,
    STATE_CARRYING,
)
from agents import GreedyAgent, TaskExecutor, MultiAgentController
from backend.coordinator.simple_coordinator import SimpleCoordinator
from coordinator.llm_coordinator import LLMCoordinator, validate_llm_output
from backend.api import SimulationManager


def test_ec01_zero_victims_remaining():
    """
    EC-1: Zero victims remaining on grid.
    Both GreedyAgent and TaskExecutor must issue safe ACTION_STAY without unhandled exceptions.
    """
    env = DisasterReliefEnv(grid_size=6, num_agents=2, max_victims=0, seed=42)
    obs = env.reset()
    env.victims = []  # Explicitly clear any victims

    # Test GreedyAgent
    greedy_agent = GreedyAgent(agent_id=0)
    action = greedy_agent.act(obs["agents"][0], obs)
    assert action == ACTION_STAY, f"Expected ACTION_STAY on zero victims, got {action}"

    # Test TaskExecutor
    executor = TaskExecutor(num_agents=2)
    actions = executor.act(obs)
    assert actions == [ACTION_STAY, ACTION_STAY], f"Expected all ACTION_STAY, got {actions}"

    # Test SimpleCoordinator
    coord = SimpleCoordinator()
    assignments = coord.assign(obs)
    assert assignments[0] is None and assignments[1] is None


def test_ec02_victim_expires_en_route():
    """
    EC-2: Victim dies while an agent is en route.
    TaskExecutor must abort target, clear target state, and issue ACTION_STAY without crash.
    """
    env = DisasterReliefEnv(grid_size=6, num_agents=1, max_victims=1, seed=42)
    obs = env.reset()
    v = env.victims[0]
    vid = v.id

    executor = TaskExecutor(num_agents=1)
    executor.set_assignment(0, f"rescue V{vid}")
    assert executor._targets[0] == vid

    # Simulate victim dying before pickup
    v.died = True
    obs = env.get_observation()

    # Next action must abort obsolete target
    action = executor.act(obs)[0]
    assert action == ACTION_STAY
    assert executor._targets[0] is None, "Target was not cleared upon victim expiration"


def test_ec03_simultaneous_pickup_and_expiration():
    """
    EC-3: Simultaneous pickup on the exact step of maximum wait time.
    Agent action is processed before deterioration; victim is safely rescued rather than expired.
    """
    env = DisasterReliefEnv(grid_size=6, num_agents=1, max_victims=1, seed=42)
    env.reset()
    agent = env.agents[0]
    v = env.victims[0]

    # Co-locate agent with victim
    agent.pos = v.pos
    # Set wait_time to exactly max_wait - 1 (next step will hit max_wait)
    v.wait_time = v.max_wait - 1

    # Execute ACTION_PICKUP
    obs, rewards, done, info = env.step([ACTION_PICKUP])

    assert agent.carrying_victim is not None
    assert agent.carrying_victim.id == v.id
    assert v.died is False, "Victim died despite simultaneous pickup"
    assert env.total_died == 0, "total_died incremented incorrectly on simultaneous pickup"


def test_ec04_llm_duplicate_assignments():
    """
    EC-4: LLM assigns multiple agents to the same victim.
    Deduplicator grants assignment to first agent and falls back conflicting agents to 'wait'.
    """
    raw_json = '{"0": "rescue V2", "1": "rescue V2", "2": "rescue V2"}'
    valid_ids = {1, 2, 3}

    assignments = validate_llm_output(raw_json, num_agents=3, valid_victim_ids=valid_ids)
    assert assignments["0"] == "rescue V2", "Agent 0 should retain target"
    assert assignments["1"] == "wait", "Agent 1 should be deduplicated to wait"
    assert assignments["2"] == "wait", "Agent 2 should be deduplicated to wait"


def test_ec05_fairness_zero_workload():
    """
    EC-5: Zero rescues across all agents.
    Jain's Fairness Index must evaluate to exactly 1.0 (perfect equality convention).
    """
    fairness_empty = SimulationManager._compute_jains_fairness([])
    assert fairness_empty == 1.0

    fairness_zeros = SimulationManager._compute_jains_fairness([0.0, 0.0, 0.0])
    assert fairness_zeros == 1.0


def test_ec06_fairness_single_agent_skew():
    """
    EC-6: Maximum workload skew (one agent performs all rescues).
    Jain's Fairness Index must evaluate to 1 / n (0.3333 for n = 3 agents).
    """
    n = 3
    fairness_skew = SimulationManager._compute_jains_fairness([6.0, 0.0, 0.0])
    expected = 1.0 / n
    assert abs(fairness_skew - expected) < 1e-4, f"Expected {expected}, got {fairness_skew}"


def test_ec07_obstacle_and_grid_boundary_collision():
    """
    EC-7: Movement into obstacles and out-of-bounds boundaries.
    Agent position must remain unchanged, preventing boundary violations or terrain clipping.
    """
    env = DisasterReliefEnv(grid_size=6, num_agents=1, seed=42)
    env.reset()
    agent = env.agents[0]

    # Test top boundary
    agent.pos = (0, 0)
    env.step([ACTION_UP])
    assert agent.pos == (0, 0), "Agent breached upper boundary"

    # Test left boundary
    env.step([ACTION_LEFT])
    assert agent.pos == (0, 0), "Agent breached left boundary"

    # Test right boundary
    agent.pos = (5, 5)
    env.step([ACTION_RIGHT])
    assert agent.pos == (5, 5), "Agent breached right boundary"

    # Test bottom boundary
    env.step([ACTION_DOWN])
    assert agent.pos == (5, 5), "Agent breached bottom boundary"

    # Test obstacle collision
    agent.pos = (2, 2)
    env.obstacles.add((2, 1))
    env.grid[1, 2] = CellType.OBSTACLE
    env.step([ACTION_UP])
    assert agent.pos == (2, 2), "Agent walked into obstacle cell"


def test_ec08_llm_hallucinated_ids():
    """
    EC-8: LLM hallucinates non-existent victim IDs.
    Validator must reject non-existent IDs and assign 'wait' without crashing.
    """
    raw_response = '```json\n{"0": "rescue V999", "1": "rescue V888", "2": "wait"}\n```'
    active_ids = {1, 2, 3}

    assignments = validate_llm_output(raw_response, num_agents=3, valid_victim_ids=active_ids)
    assert assignments["0"] == "wait", f"Hallucinated V999 was not rejected: {assignments['0']}"
    assert assignments["1"] == "wait", f"Hallucinated V888 was not rejected: {assignments['1']}"
    assert assignments["2"] == "wait"
