"""
Unified test runner and verification suite for Phase 1 (env/), Phase 2 (agents/), and Phase 3 (coordinator/).
"""

import os
import sys
import time

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
from coordinator import SimpleCoordinator, LLMCoordinator, validate_llm_output, make_smart_llm_client


def run_phase_1_suite():
    print("\n" + "=" * 60)
    print("RUNNING PHASE 1 VERIFICATION SUITE: Simulation Core & Environment Engine")
    print("=" * 60)

    # Test 1.1: Environment Initialization
    env = DisasterReliefEnv(grid_size=10, num_agents=3, num_shelters=2, num_depots=2, seed=42)
    obs = env.reset()
    assert obs["grid_size"] == 10, "Failed grid_size check"
    assert len(obs["agents"]) == 3, "Failed agent count check"
    assert len(obs["shelters"]) == 2, "Failed shelter count check"
    assert len(obs["depots"]) == 2, "Failed depot count check"
    assert len(obs["victims"]) >= 1, "Failed initial victim population check"
    print("  [PASS] 1.1 Environment initialization & structured observation packaging")

    # Test 1.2: Boundary & Obstacle Physics
    env = DisasterReliefEnv(grid_size=6, num_agents=1, seed=42)
    env.reset()
    env.agents[0].pos = (0, 0)
    env.step([ACTION_UP])
    assert env.agents[0].pos == (0, 0), "Agent breached upper boundary"
    env.step([ACTION_LEFT])
    assert env.agents[0].pos == (0, 0), "Agent breached left boundary"

    env.agents[0].pos = (2, 2)
    env.obstacles.add((2, 1))
    env.grid[1, 2] = CellType.OBSTACLE
    env.step([ACTION_UP])
    assert env.agents[0].pos == (2, 2), "Agent moved into obstacle cell"
    print("  [PASS] 1.2 Boundary enforcement & obstacle collision avoidance")

    # Test 1.3: Victim Deterioration & Expiration Lifecycle
    env = DisasterReliefEnv(grid_size=6, num_agents=1, max_victims=1, seed=42)
    env.reset()
    v = env.victims[0]
    initial_urgency = v.urgency()
    initial_wait = v.wait_time
    for _ in range(5):
        env.step([ACTION_STAY])
    assert v.wait_time == initial_wait + 5, "Victim wait_time did not increment properly"
    assert v.urgency() > initial_urgency, "Victim urgency did not increase with wait_time"

    max_w = v.max_wait
    for _ in range(max_w):
        env.step([ACTION_STAY])
    assert v.died is True, "Victim did not transition to died status"
    assert env.total_died >= 1, "Environment total_died counter failed to increment"
    print("  [PASS] 1.3 Dynamic victim deterioration, urgency escalation, & mortality")

    # Test 1.4: In-Transit Expiration Pause (EC-1.3)
    env = DisasterReliefEnv(grid_size=6, num_agents=1, max_victims=1, seed=42)
    env.reset()
    env.agents[0].pos = (1, 1)
    env.victims = []
    short_v = env._spawn_victim(pos=(1, 1), max_wait=3)
    env.step([ACTION_PICKUP])
    assert env.agents[0].carrying_victim_id == short_v.id, "Pickup failed"

    for _ in range(10):
        env.step([ACTION_STAY])
    assert short_v.died is False, "Victim expired while in transit"
    assert env.total_died == 0, "Total died counter changed unexpectedly"
    print("  [PASS] 1.4 In-transit deterioration pause (EC-1.3 verified)")

    # Test 1.5: Division-by-Zero Safeguard
    short_v.wait_time = short_v.max_wait + 20
    urg = short_v.urgency()
    assert isinstance(urg, float) and urg > 0, "Urgency crashed on wait_time >= max_wait"
    print("  [PASS] 1.5 Division-by-zero protection in urgency calculation")


def run_phase_2_suite():
    print("\n" + "=" * 60)
    print("RUNNING PHASE 2 VERIFICATION SUITE: Tactical Agent Layer & Command Interpreter")
    print("=" * 60)

    # Test 2.1: GreedyAgent Movement & Pickup
    env = DisasterReliefEnv(grid_size=6, num_agents=1, seed=42)
    env.reset()
    env.agents[0].pos = (1, 1)
    env.victims = []
    v = env._spawn_victim(pos=(3, 1))

    ga = GreedyAgent(agent_id=0)
    obs = env.get_observation()
    action = ga.act(obs["agents"][0], obs)
    assert action == ACTION_RIGHT, f"GreedyAgent expected ACTION_RIGHT(4), got {action}"
    print("  [PASS] 2.1 GreedyAgent nearest-victim directional heuristics")

    env.agents[0].pos = (3, 1)
    obs = env.get_observation()
    action = ga.act(obs["agents"][0], obs)
    assert action == ACTION_PICKUP, f"GreedyAgent expected ACTION_PICKUP(5), got {action}"
    print("  [PASS] 2.2 GreedyAgent co-located pickup trigger")

    # Test 2.2: GreedyAgent Shelter Delivery
    env.shelters = [(3, 3)]
    env.grid[3, 3] = CellType.SHELTER
    env.agents[0].carrying_victim = v
    env.agents[0].state = STATE_CARRYING

    obs = env.get_observation()
    action = ga.act(obs["agents"][0], obs)
    assert action == ACTION_DOWN, f"GreedyAgent expected ACTION_DOWN(2) to shelter, got {action}"

    env.agents[0].pos = (3, 3)
    obs = env.get_observation()
    action = ga.act(obs["agents"][0], obs)
    assert action == ACTION_DROPOFF, f"GreedyAgent expected ACTION_DROPOFF(6) at shelter, got {action}"
    print("  [PASS] 2.3 GreedyAgent transit navigation & shelter delivery")

    # Test 2.3: TaskExecutor Declarative Assignment Lifecycle
    env = DisasterReliefEnv(grid_size=6, num_agents=1, seed=10)
    env.reset()
    env.agents[0].pos = (1, 1)
    env.victims = []
    v = env._spawn_victim(pos=(1, 1))
    target_id = v.id
    env.shelters = [(1, 3)]
    env.grid[3, 1] = CellType.SHELTER

    executor = TaskExecutor(num_agents=1)
    executor.set_assignment(0, f"rescue V{target_id}")

    act_pickup = executor.act(env.get_observation())[0]
    assert act_pickup == ACTION_PICKUP, "TaskExecutor failed to issue pickup"
    env.step([act_pickup])

    for _ in range(10):
        actions = executor.act(env.get_observation())
        env.step(actions)
        if env.total_rescued >= 1:
            break

    assert env.total_rescued == 1, "TaskExecutor failed to complete delivery"
    print("  [PASS] 2.4 TaskExecutor declarative command execution ('rescue V{id}')")

    # Test 2.4: Target Expiration En-Route Abort (EC-2.2)
    env = DisasterReliefEnv(grid_size=6, num_agents=1, seed=42)
    env.reset()
    env.agents[0].pos = (0, 0)
    env.victims = []
    v2 = env._spawn_victim(pos=(5, 5))
    executor.set_assignment(0, f"rescue V{v2.id}")

    v2.died = True

    action_aborted = executor.act(env.get_observation())[0]
    assert action_aborted == ACTION_STAY, "TaskExecutor did not abort expired target"
    assert executor.get_assignment(0) is None, "TaskExecutor did not clear dead target"
    print("  [PASS] 2.5 TaskExecutor en-route target mortality abort (EC-2.2 verified)")


def run_phase_3_suite():
    print("\n" + "=" * 60)
    print("RUNNING PHASE 3 VERIFICATION SUITE: High-Level Coordination Suite")
    print("=" * 60)

    # Test 3.1: Hungarian Assignment
    env = DisasterReliefEnv(grid_size=6, num_agents=1, seed=42)
    env.reset()
    env.agents[0].pos = (0, 0)
    env.victims = []
    v1 = env._spawn_victim(pos=(3, 0), severity=Severity.STABLE, max_wait=60)
    v2 = env._spawn_victim(pos=(0, 3), severity=Severity.CRITICAL, max_wait=10)
    v2.wait_time = 8  # Urgency ~ 15.0

    coord = SimpleCoordinator(urgency_weight=4.0)
    assignments = coord.assign(env.get_observation())
    assert assignments[0] == f"rescue V{v2.id}", f"Expected V{v2.id}, got {assignments[0]}"
    print("  [PASS] 3.1 Hungarian bipartite optimal matching (Distance vs Urgency weighting)")

    # Test 3.2: Zero-Victims Handled
    env.victims = []
    coord_empty = SimpleCoordinator()
    assignments_empty = coord_empty.assign(env.get_observation())
    assert assignments_empty[0] is None
    print("  [PASS] 3.2 Coordinator zero-victim robustness (EC-3.4)")

    # Test 3.3: LLM Output Validation & Deduplication
    raw_fence = "Plan:\n```json\n{\"0\": \"rescue V1\", \"1\": \"rescue V1\", \"2\": \"rescue V999\"}\n```"
    res = validate_llm_output(raw_fence, num_agents=3, valid_victim_ids={1, 2})
    assert res["0"] == "rescue V1", "Agent 0 failed valid assignment"
    assert res["1"] == "wait", "Duplicate assignment to V1 was not deduplicated (EC-4.2)"
    assert res["2"] == "wait", "Hallucinated V999 was not rejected (EC-4.1)"
    print("  [PASS] 3.3 LLM Output JSON validation, fence stripping, & deduplication")

    # Test 3.4: Circuit Breaker on LLM Failure
    def failing_api(prompt):
        raise TimeoutError("LLM Gateway Timeout")

    llm_coord = LLMCoordinator(llm_client=failing_api)
    fallback_res = llm_coord.assign(env.get_observation())
    assert len(fallback_res) == 1
    assert "0" in fallback_res
    print("  [PASS] 3.4 Heuristic circuit breaker on LLM exception (EC-4.3)")


def run_phase_4_suite():
    print("\n" + "=" * 60)
    print("RUNNING PHASE 4 VERIFICATION SUITE: Backend Gateway & Serialization")
    print("=" * 60)
    from fastapi.testclient import TestClient
    from backend.api import app, SimulationManager
    from backend.runner import run_episode

    client = TestClient(app)

    # 4.1 Reset
    res = client.post("/reset", json={"mode": "coordinated", "seed": 42})
    assert res.status_code == 200, "POST /reset failed"
    data = res.json()
    assert data["mode"] == "coordinated"
    print("  [PASS] 4.1 FastAPI /reset endpoint initialization")

    # 4.2 Step
    step_res = client.post("/step")
    assert step_res.status_code == 200, "POST /step failed"
    step_data = step_res.json()
    assert "state" in step_data and "events" in step_data and "metrics" in step_data
    print("  [PASS] 4.2 FastAPI /step state transition and delta event streaming")

    # 4.3 Jain's Fairness Boundary
    assert SimulationManager._compute_jains_fairness([0.0, 0.0, 0.0]) == 1.0
    skew = SimulationManager._compute_jains_fairness([3.0, 0.0, 0.0])
    assert abs(skew - (1.0 / 3.0)) < 1e-4
    print("  [PASS] 4.3 Jain's Fairness Index mathematical boundary correctness")

    # 4.4 Headless runner
    ep_res = run_episode(mode="coordinated", steps=10, seed=42, verbose=False)
    assert ep_res["summary"]["steps"] == 10
    print("  [PASS] 4.4 Headless episode runner trace export")


def run_phase_6_suite():
    print("\n" + "=" * 60)
    print("RUNNING PHASE 6 VERIFICATION SUITE: Edge Case Hardening & Regression")
    print("=" * 60)
    from tests.test_edge_cases import (
        test_ec01_zero_victims_remaining,
        test_ec02_victim_expires_en_route,
        test_ec03_simultaneous_pickup_and_expiration,
        test_ec04_llm_duplicate_assignments,
        test_ec05_fairness_zero_workload,
        test_ec06_fairness_single_agent_skew,
        test_ec07_obstacle_and_grid_boundary_collision,
        test_ec08_llm_hallucinated_ids,
    )

    test_ec01_zero_victims_remaining()
    print("  [PASS] EC-1 Zero victims remaining on grid handled safely")

    test_ec02_victim_expires_en_route()
    print("  [PASS] EC-2 En-route victim expiration handled with abort")

    test_ec03_simultaneous_pickup_and_expiration()
    print("  [PASS] EC-3 Simultaneous pickup and deterioration precedence")

    test_ec04_llm_duplicate_assignments()
    print("  [PASS] EC-4 Duplicate assignment deduplication")

    test_ec05_fairness_zero_workload()
    print("  [PASS] EC-5 Fairness index zero workload convention (J=1.0)")

    test_ec06_fairness_single_agent_skew()
    print("  [PASS] EC-6 Fairness index maximum workload skew (J=1/n)")

    test_ec07_obstacle_and_grid_boundary_collision()
    print("  [PASS] EC-7 Obstacle and grid boundary collision avoidance")

    test_ec08_llm_hallucinated_ids()
    print("  [PASS] EC-8 Hallucinated LLM victim IDs discarded")


def run_phase_7_suite():
    print("\n" + "=" * 60)
    print("RUNNING PHASE 7 VERIFICATION SUITE: RL Baseline & Gymnasium Environment")
    print("=" * 60)
    from tests.test_ppo import (
        test_gym_wrapper_shapes_and_reset,
        test_gym_wrapper_step_transition,
        test_ppo_fleet_policy_fallback_and_prediction,
    )

    test_gym_wrapper_shapes_and_reset()
    print("  [PASS] 7.1 Gymnasium environment wrapper shapes and reset packaging")

    test_gym_wrapper_step_transition()
    print("  [PASS] 7.2 Multi-agent step transition, rewards, and truncation limits")

    test_ppo_fleet_policy_fallback_and_prediction()
    print("  [PASS] 7.3 PPO fleet policy loading and deterministic action prediction")


def run_coordinated_multi_agent_episode():
    print("\n" + "=" * 60)
    print("RUNNING END-TO-END COORDINATED SIMULATION EPISODE (Hungarian + TaskExecutor)")
    print("=" * 60)

    env = DisasterReliefEnv(grid_size=10, num_agents=3, num_shelters=2, num_depots=2, seed=42)
    coordinator = SimpleCoordinator(assign_interval=1, urgency_weight=4.0)
    executor = TaskExecutor(num_agents=3)

    start_time = time.time()
    steps = 40
    for s in range(steps):
        obs = env.get_observation()
        assignments = coordinator.assign(obs, step=s)
        for aid, task in assignments.items():
            if task is not None and executor._targets[aid] is None:
                executor.set_assignment(aid, task)
        actions = executor.act(obs)
        obs, rewards, done, info = env.step(actions)

    elapsed = time.time() - start_time
    print(f"  [COORDINATED EPISODE COMPLETED] {steps} steps in {elapsed*1000:.2f}ms")
    print(f"  - Total Rescued: {env.total_rescued}")
    print(f"  - Total Casualties: {env.total_died}")
    print(f"  - Rescues per Agent: {[a.total_rescues for a in env.agents]}")
    print(f"  - Moves per Agent: {[a.total_moves for a in env.agents]}")
    assert env.total_rescued > 0, "Coordinated fleet failed to achieve any rescues"
    print("  [PASS] Coordinated multi-agent episode completed with zero errors")


if __name__ == "__main__":
    print("=" * 60)
    print("STARTING COMPLETE H-MAS PLATFORM VERIFICATION (PHASES 1 - 7)")
    print("=" * 60)
    run_phase_1_suite()
    run_phase_2_suite()
    run_phase_3_suite()
    run_phase_4_suite()
    run_phase_6_suite()
    run_phase_7_suite()
    run_coordinated_multi_agent_episode()
    print("\n" + "=" * 60)
    print("ALL VERIFICATION GATES PASSED (100% SUCCESS)")
    print("=" * 60)

