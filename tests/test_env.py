import os
import sys

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


def test_reset_observation():
    """Verify clean initialization of observation dictionary and entities."""
    env = DisasterReliefEnv(grid_size=10, num_agents=3, num_shelters=2, num_depots=2, seed=42)
    obs = env.reset()

    assert obs["grid_size"] == 10
    assert obs["step"] == 0
    assert len(obs["agents"]) == 3
    assert len(obs["shelters"]) == 2
    assert len(obs["depots"]) == 2
    assert len(obs["victims"]) >= 1

    # Verify grid structure
    assert len(obs["grid"]) == 10
    for row in obs["grid"]:
        assert len(row.split()) == 10


def test_pickup_and_dropoff():
    """Verify complete lifecycle: pickup at victim cell and delivery at shelter."""
    env = DisasterReliefEnv(grid_size=6, num_agents=1, num_shelters=1, num_depots=1, seed=10)
    env.reset()

    # Place agent and victim at (2, 2)
    env.agents[0].pos = (2, 2)
    env.victims = []
    v = env._spawn_victim(pos=(2, 2), severity=Severity.SERIOUS, max_wait=50)
    victim_id = v.id

    # Configure single shelter at (2, 3)
    env.shelters = [(2, 3)]
    env.grid[3, 2] = CellType.SHELTER

    # 1. Pickup
    obs, rewards, done, info = env.step([ACTION_PICKUP])
    assert env.agents[0].state == STATE_CARRYING
    assert env.agents[0].carrying_victim_id == victim_id
    assert rewards[0] > 0

    # 2. Move to shelter (from (2,2) down to (2,3))
    obs, rewards, done, info = env.step([ACTION_DOWN])
    assert env.agents[0].pos == (2, 3)
    assert v.pos == (2, 3)

    # 3. Dropoff at shelter
    obs, rewards, done, info = env.step([ACTION_DROPOFF])
    assert env.agents[0].state == STATE_IDLE
    assert env.agents[0].carrying_victim_id is None
    assert v.rescued is True
    assert env.total_rescued == 1
    assert info["new_rescues"] == 1


def test_victim_deterioration_and_death():
    """Verify victim health deterioration and transition to died status."""
    env = DisasterReliefEnv(grid_size=6, num_agents=1, max_victims=1, seed=42)
    env.reset()

    v = env.victims[0]
    initial_urgency = v.urgency()
    max_wait = v.max_wait

    # Step partially and check urgency increases
    for _ in range(5):
        env.step([ACTION_STAY])

    assert v.wait_time == 5
    assert v.urgency() > initial_urgency

    # Step until victim expires
    for _ in range(max_wait):
        env.step([ACTION_STAY])

    assert v.died is True
    assert env.total_died >= 1


def test_carrying_victim_does_not_die_in_transit():
    """Verify victim carried by agent pauses deterioration and avoids in-transit death."""
    env = DisasterReliefEnv(grid_size=6, num_agents=1, max_victims=1, seed=42)
    env.reset()

    # Spawn victim with short max_wait of 5
    env.agents[0].pos = (1, 1)
    env.victims = []
    v = env._spawn_victim(pos=(1, 1), max_wait=5)
    
    # Pick up victim
    env.step([ACTION_PICKUP])
    assert env.agents[0].carrying_victim_id == v.id

    # Step 10 times while in transit
    for _ in range(10):
        env.step([ACTION_STAY])

    # Should NOT have died because deterioration is paused while carried
    assert v.died is False
    assert env.total_died == 0


def test_obstacle_and_boundary_collision():
    """Verify agent cannot cross grid boundaries or walk into obstacle cells."""
    env = DisasterReliefEnv(grid_size=6, num_agents=1, seed=42)
    env.reset()

    # Test top-left boundary (0, 0)
    env.agents[0].pos = (0, 0)
    env.step([ACTION_UP])
    assert env.agents[0].pos == (0, 0)

    env.step([ACTION_LEFT])
    assert env.agents[0].pos == (0, 0)

    # Test obstacle collision
    env.agents[0].pos = (2, 2)
    env.obstacles.add((2, 1)) # Obstacle directly UP
    env.grid[1, 2] = CellType.OBSTACLE

    env.step([ACTION_UP])
    # Position should remain unchanged
    assert env.agents[0].pos == (2, 2)


def test_urgency_division_by_zero_safe():
    """Verify urgency calculation is safe against division by zero."""
    env = DisasterReliefEnv(grid_size=6, num_agents=1, seed=42)
    env.reset()
    v = env.victims[0]
    
    # Force wait_time to equal max_wait
    v.wait_time = v.max_wait
    urgency_val = v.urgency()
    assert isinstance(urgency_val, float)
    assert urgency_val > 0.0

    # Force wait_time to exceed max_wait
    v.wait_time = v.max_wait + 10
    urgency_val_exceeded = v.urgency()
    assert isinstance(urgency_val_exceeded, float)


def test_inter_agent_collision_prevention():
    """Verify two agents moving into the same corridor cell do not overlap or swap."""
    env = DisasterReliefEnv(grid_size=8, num_agents=2, seed=42)
    env.reset()

    # Clear obstacles for predictable corridor geometry
    env.obstacles = set()

    # Place agents on either side of corridor cell (3, 2)
    env.agents[0].pos = (2, 2)
    env.agents[1].pos = (4, 2)

    # Both agents attempt to step into (3, 2) simultaneously
    env.step([ACTION_RIGHT, ACTION_LEFT])

    # Agent 0 commits (3, 2); Agent 1 is blocked and stays at (4, 2)
    assert env.agents[0].pos == (3, 2)
    assert env.agents[1].pos == (4, 2)
    assert env.agents[0].pos != env.agents[1].pos, "Agents overlapped on same cell"

    # Test swap collision: Agent 0 at (3, 2) tries RIGHT, Agent 1 at (4, 2) tries LEFT
    env.step([ACTION_RIGHT, ACTION_LEFT])
    # Swap is prevented; neither agent can move into the other's occupied cell
    assert env.agents[0].pos == (3, 2)
    assert env.agents[1].pos == (4, 2)
