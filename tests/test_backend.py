import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure local module priority
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.api import app, SimulationManager
from backend.runner import run_episode


client = TestClient(app)


def test_api_reset_endpoint():
    """Verify /reset endpoint initializes simulation in coordinated, greedy, and llm modes."""
    # Coordinated mode
    res = client.post("/reset", json={"mode": "coordinated", "seed": 42})
    assert res.status_code == 200
    data = res.json()
    assert data["mode"] == "coordinated"
    assert data["grid_size"] == 10
    assert len(data["agents"]) == 3
    assert len(data["victims"]) >= 1
    assert data["step"] == 0

    # Greedy mode
    res_greedy = client.post("/reset", json={"mode": "greedy", "seed": 99})
    assert res_greedy.status_code == 200
    assert res_greedy.json()["mode"] == "greedy"


def test_api_step_endpoint():
    """Verify /step endpoint advances state, generates delta events and metrics."""
    client.post("/reset", json={"mode": "coordinated", "seed": 42})
    step_res = client.post("/step")
    assert step_res.status_code == 200
    data = step_res.json()

    assert "state" in data
    assert "events" in data
    assert "metrics" in data
    assert data["state"]["step"] == 1
    assert isinstance(data["events"]["newly_rescued"], list)
    assert isinstance(data["events"]["newly_died"], list)
    assert "assignments" in data["events"]


def test_api_state_and_metrics_endpoints():
    """Verify /state and /metrics return coherent analytical representations."""
    client.post("/reset", json={"mode": "coordinated", "seed": 42})
    
    # State endpoint
    state_res = client.get("/state")
    assert state_res.status_code == 200
    s_data = state_res.json()
    assert len(s_data["grid"]) == 10
    assert isinstance(s_data["grid"][0], str)
    assert len(s_data["grid_matrix"]) == 10
    assert len(s_data["agents"]) == 3

    # Metrics endpoint
    m_res = client.get("/metrics")
    assert m_res.status_code == 200
    m_data = m_res.json()
    assert "total_rescued" in m_data
    assert "total_died" in m_data
    assert "fairness_index" in m_data
    # With 0 rescues, fairness index convention must be 1.0 (EC-4.2 / EC-05)
    assert m_data["fairness_index"] == 1.0


def test_jains_fairness_calculation():
    """Verify mathematical correctness and boundary conditions of Jain's fairness index."""
    # Zero workload convention -> 1.0
    assert SimulationManager._compute_jains_fairness([]) == 1.0
    assert SimulationManager._compute_jains_fairness([0.0, 0.0, 0.0]) == 1.0

    # Perfect balance [2, 2, 2] -> 1.0
    assert SimulationManager._compute_jains_fairness([2.0, 2.0, 2.0]) == 1.0

    # Single agent skew [3, 0, 0] -> 1/3 ~ 0.3333
    skew = SimulationManager._compute_jains_fairness([3.0, 0.0, 0.0])
    assert abs(skew - (1.0 / 3.0)) < 1e-4


def test_headless_runner_episode():
    """Verify backend headless episode runner executes and produces summary trace."""
    res = run_episode(mode="coordinated", steps=15, seed=42, verbose=False)
    summary = res["summary"]
    trace = res["trace"]

    assert summary["steps"] == 15
    assert len(trace) == 15
    assert "fairness_index" in summary
    assert "total_rescued" in summary
    assert "agent_rescues" in summary


def test_api_llm_mode_non_blocking_interim_bridge():
    """Verify LLM mode initializes background coordination and executes step with interim bridge."""
    res_reset = client.post("/reset", json={"mode": "llm", "seed": 42})
    assert res_reset.status_code == 200
    assert res_reset.json()["mode"] == "llm"

    # Step in LLM mode must execute promptly without blocking
    step_res = client.post("/step")
    assert step_res.status_code == 200
    data = step_res.json()

    assert data["state"]["step"] == 1
    assert "assignments" in data["events"]
    # Check that agents received valid targets (either from cached LLM or interim bridge)
    assignments = data["events"]["assignments"]
    assert len(assignments) == 3
