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
)
from agents import GreedyAgent, TaskExecutor, MultiAgentController


def test_greedy_agent_moves_to_nearest_victim():
    """Verify GreedyAgent chooses action towards closest active victim."""
    env = DisasterReliefEnv(grid_size=6, num_agents=1, num_shelters=1, num_depots=1, seed=10)
    env.reset()

    # Place agent at (0, 0) and victim at (2, 0)
    env.agents[0].pos = (0, 0)
    env.victims = []
    v = env._spawn_victim(pos=(2, 0))

    obs = env.get_observation()
    agent_info = obs["agents"][0]
    ga = GreedyAgent(agent_id=0)

    action = ga.act(agent_info, obs)
    assert action == ACTION_RIGHT


def test_greedy_agent_picks_up_when_co_located():
    """Verify GreedyAgent issues ACTION_PICKUP when occupying same cell as victim."""
    env = DisasterReliefEnv(grid_size=6, num_agents=1, seed=10)
    env.reset()

    env.agents[0].pos = (2, 2)
    env.victims = []
    env._spawn_victim(pos=(2, 2))

    obs = env.get_observation()
    agent_info = obs["agents"][0]
    ga = GreedyAgent(agent_id=0)

    action = ga.act(agent_info, obs)
    assert action == ACTION_PICKUP


def test_greedy_agent_delivers_to_shelter_when_carrying():
    """Verify GreedyAgent navigates to shelter and drops off when carrying."""
    env = DisasterReliefEnv(grid_size=6, num_agents=1, num_shelters=1, seed=10)
    env.reset()

    # Shelter at (2, 2)
    env.shelters = [(2, 2)]
    env.grid[2, 2] = CellType.SHELTER

    # Agent carrying victim at (2, 0)
    env.agents[0].pos = (2, 0)
    env.victims = []
    v = env._spawn_victim(pos=(2, 0))
    env.agents[0].carrying_victim = v
    env.agents[0].state = "carrying"

    obs = env.get_observation()
    agent_info = obs["agents"][0]
    ga = GreedyAgent(agent_id=0)

    # From (2, 0) to (2, 2) should move DOWN
    action = ga.act(agent_info, obs)
    assert action == ACTION_DOWN

    # If co-located at shelter (2, 2)
    env.agents[0].pos = (2, 2)
    obs = env.get_observation()
    action_at_shelter = ga.act(obs["agents"][0], obs)
    assert action_at_shelter == ACTION_DROPOFF


def test_task_executor_lifecycle():
    """Verify TaskExecutor parses assignment string, navigates, and drops off."""
    env = DisasterReliefEnv(grid_size=6, num_agents=1, num_shelters=1, seed=5)
    env.reset()

    env.agents[0].pos = (1, 1)
    env.victims = []
    v = env._spawn_victim(pos=(1, 1))
    victim_id = v.id
    env.shelters = [(1, 2)]
    env.grid[2, 1] = CellType.SHELTER

    executor = TaskExecutor(num_agents=1)
    executor.set_assignment(0, f"rescue V{victim_id}")

    obs = env.get_observation()
    actions = executor.act(obs)
    assert actions[0] == ACTION_PICKUP

    # Apply pickup
    env.step([actions[0]])

    # Next step moves towards shelter
    obs = env.get_observation()
    actions = executor.act(obs)
    assert actions[0] != ACTION_STAY

    # Step repeatedly until delivered
    for _ in range(10):
        obs, rewards, done, info = env.step(executor.act(env.get_observation()))
        if env.total_rescued > 0:
            break

    assert env.total_rescued >= 1


def test_task_executor_aborts_on_victim_death():
    """Verify TaskExecutor safely aborts target assignment if victim dies en route (EC-2.2)."""
    env = DisasterReliefEnv(grid_size=6, num_agents=1, seed=42)
    env.reset()

    env.agents[0].pos = (0, 0)
    env.victims = []
    v = env._spawn_victim(pos=(5, 5))
    vid = v.id

    executor = TaskExecutor(num_agents=1)
    executor.set_assignment(0, f"rescue V{vid}")

    # Mark victim as died mid-transit
    v.died = True

    obs = env.get_observation()
    actions = executor.act(obs)
    # Since victim died, target is invalidated and agent stays
    assert actions[0] == ACTION_STAY
    assert executor.get_assignment(0) is None


def test_multi_agent_controller():
    """Verify MultiAgentController coordinates multiple agents."""
    env = DisasterReliefEnv(grid_size=6, num_agents=3, seed=42)
    env.reset()

    agents = [GreedyAgent(agent_id=i) for i in range(3)]
    controller = MultiAgentController(agents)

    obs = env.get_observation()
    actions = controller.act(obs)

    assert len(actions) == 3
    for act in actions:
        assert act in [ACTION_STAY, ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_PICKUP, ACTION_DROPOFF]


def test_task_executor_navigates_concave_obstacle_without_oscillating():
    """Verify TaskExecutor with A* routes out of a concave U-shaped barrier without oscillating."""
    env = DisasterReliefEnv(grid_size=8, num_agents=1, seed=42)
    env.reset()

    # Agent inside U-shaped trap at (2, 2)
    env.agents[0].pos = (2, 2)
    env.victims = []
    # Victim outside U-shaped trap at (2, 5)
    v = env._spawn_victim(pos=(2, 5))

    # Construct U-shaped obstacle surrounding (2, 2) on left, bottom, right
    u_obstacles = [
        (1, 1), (1, 2), (1, 3),  # Left wall
        (2, 3),                  # Bottom wall
        (3, 1), (3, 2), (3, 3),  # Right wall
    ]
    env.obstacles = set(u_obstacles)
    for obs_pos in u_obstacles:
        env.grid[obs_pos[1], obs_pos[0]] = CellType.OBSTACLE

    executor = TaskExecutor(num_agents=1)
    executor.set_assignment(0, f"rescue V{v.id}")

    # Step repeatedly and verify the agent successfully reaches the victim and picks it up
    reached = False
    for _ in range(25):
        obs = env.get_observation()
        actions = executor.act(obs)
        if actions[0] == ACTION_PICKUP:
            reached = True
            break
        env.step(actions)

    assert reached, "Agent failed to escape U-shaped obstacle pocket to reach victim"


def test_agent_multi_victim_capacity():
    """Verify agent can board multiple victims up to max_capacity and discharge both at shelter."""
    env = DisasterReliefEnv(grid_size=6, num_agents=1, num_shelters=1, seed=42)
    env.reset()

    agent = env.agents[0]
    agent.pos = (2, 2)
    agent.max_capacity = 2
    env.victims = []

    v1 = env._spawn_victim(pos=(2, 2), severity=Severity.STABLE)
    v2 = env._spawn_victim(pos=(2, 2), severity=Severity.CRITICAL)

    env.shelters = [(2, 3)]
    env.grid[3, 2] = CellType.SHELTER

    # 1. First pickup boards highest urgency (v2)
    env.step([ACTION_PICKUP])
    assert len(agent.inventory) == 1
    assert agent.carrying_victim.id == v2.id

    # 2. Second pickup boards remaining victim (v1)
    env.step([ACTION_PICKUP])
    assert len(agent.inventory) == 2
    assert {v.id for v in agent.inventory} == {v1.id, v2.id}

    # 3. Third pickup ignored since agent at max_capacity
    v3 = env._spawn_victim(pos=(2, 2))
    env.step([ACTION_PICKUP])
    assert len(agent.inventory) == 2

    # 4. Move to shelter and drop off both victims
    env.step([ACTION_DOWN])
    assert agent.pos == (2, 3)
    env.step([ACTION_DROPOFF])

    assert len(agent.inventory) == 0
    assert agent.carrying_victim is None
    assert agent.state == "idle"
    assert v1.rescued is True and v2.rescued is True
    assert env.total_rescued == 2
