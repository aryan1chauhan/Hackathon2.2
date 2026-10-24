import os
import sys

# Ensure local module priority
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import DisasterReliefEnv, Severity
from coordinator import SimpleCoordinator, LLMCoordinator, validate_llm_output, make_smart_llm_client


def test_simple_coordinator_assigns_highest_urgency():
    """Verify Hungarian coordinator prioritizes victims with higher urgency."""
    env = DisasterReliefEnv(grid_size=6, num_agents=1, num_shelters=1, num_depots=1, seed=42)
    env.reset()

    # Agent at (0, 0)
    env.agents[0].pos = (0, 0)
    env.victims = []

    # Spawn V1 (stable, low urgency) and V2 (critical, high urgency) at same distance
    v1 = env._spawn_victim(pos=(3, 0), severity=Severity.STABLE, max_wait=60)
    v2 = env._spawn_victim(pos=(0, 3), severity=Severity.CRITICAL, max_wait=10)
    v2.wait_time = 8  # Push urgency very high: 3 * (1 + 8/2) = 15.0

    coord = SimpleCoordinator(assign_interval=1, urgency_weight=4.0)
    assignments = coord.assign(env.get_observation(), step=0)

    # Should assign V2 because urgency weight outweighs equal distance
    assert assignments[0] == f"rescue V{v2.id}"


def test_simple_coordinator_zero_active_victims():
    """Verify coordinator cleanly handles zero active victims on grid (EC-3.4)."""
    env = DisasterReliefEnv(grid_size=6, num_agents=2, seed=42)
    env.reset()
    env.victims = []  # No victims

    coord = SimpleCoordinator()
    assignments = coord.assign(env.get_observation())

    assert len(assignments) == 2
    assert assignments[0] is None
    assert assignments[1] is None


def test_simple_coordinator_carrying_agent_not_reassigned():
    """Verify an agent already carrying a victim is not assigned another pickup."""
    env = DisasterReliefEnv(grid_size=6, num_agents=2, seed=42)
    env.reset()

    # Agent 0 is carrying victim
    v_carried = env.victims[0]
    env.agents[0].carrying_victim = v_carried
    env.agents[0].state = "carrying"

    coord = SimpleCoordinator()
    assignments = coord.assign(env.get_observation())

    # Agent 0 should NOT be assigned a new victim
    assert assignments[0] is None
    # Agent 1 (idle) can still receive an assignment if victims exist
    if len(env.victims) > 1:
        assert assignments[1] is not None


def test_llm_validator_clean_json():
    """Verify validate_llm_output parses standard JSON assignment schema."""
    raw = '{"0": "rescue V2", "1": "wait", "2": "rescue V1"}'
    res = validate_llm_output(raw, num_agents=3)
    assert res["0"] == "rescue V2"
    assert res["1"] == "wait"
    assert res["2"] == "rescue V1"


def test_llm_validator_extracts_from_markdown_fences():
    """Verify validator extracts JSON surrounded by conversational text and fences (EC-3.5)."""
    raw = (
        "Here is the strategic plan for this step:\n"
        "```json\n"
        '{\n  "0": "rescue V3",\n  "1": "wait",\n  "2": "rescue V2"\n}\n'
        "```\n"
        "Please execute with priority."
    )
    res = validate_llm_output(raw, num_agents=3)
    assert res["0"] == "rescue V3"
    assert res["1"] == "wait"
    assert res["2"] == "rescue V2"


def test_llm_validator_deduplicates_target_assignments():
    """Verify duplicate assignments to same victim are deduplicated (EC-4.2)."""
    raw = '{"0": "rescue V1", "1": "rescue V1", "2": "rescue V2"}'
    res = validate_llm_output(raw, num_agents=3)

    # First agent keeps V1; second agent is converted to 'wait'
    assert res["0"] == "rescue V1"
    assert res["1"] == "wait"
    assert res["2"] == "rescue V2"


def test_llm_validator_rejects_hallucinated_ids():
    """Verify validator discards hallucinated victim IDs not in active set (EC-4.1)."""
    raw = '{"0": "rescue V999", "1": "rescue V1"}'
    valid_ids = {1, 2, 3}  # V999 does not exist
    res = validate_llm_output(raw, num_agents=2, valid_victim_ids=valid_ids)

    assert res["0"] == "wait"  # Hallucinated ID rejected
    assert res["1"] == "rescue V1"  # Valid ID accepted


def test_llm_coordinator_circuit_breaker_on_exception():
    """Verify LLMCoordinator triggers circuit breaker on API/network failure (EC-4.3)."""
    def failing_client(prompt: str) -> str:
        raise ConnectionError("LLM API Quota Exceeded / Rate Limit")

    coord = LLMCoordinator(llm_client=failing_client)

    env = DisasterReliefEnv(grid_size=6, num_agents=2, seed=42)
    obs = env.reset()

    # Should not throw ConnectionError; circuit breaker handles gracefully
    assignments = coord.assign(obs)
    assert len(assignments) == 2
    assert "0" in assignments
    assert "1" in assignments
