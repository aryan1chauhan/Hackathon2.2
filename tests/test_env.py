import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from env import DisasterReliefEnv, Victim
from env.constants import ACTION_PICKUP, ACTION_DROPOFF, ACTION_STAY, Severity


def test_reset_observation():
    env = DisasterReliefEnv(seed=123)
    obs = env.reset()
    assert obs['grid_size'] == env.grid_size
    assert len(obs['agents']) == env.num_agents
    # grid should be a numpy array
    assert obs['grid'] is not None


def test_pickup_and_dropoff():
    # small env with one agent to simplify
    env = DisasterReliefEnv(grid_size=6, num_agents=1, num_shelters=1, num_depots=1, seed=1)
    env.reset()

    # create a victim next to the agent
    agent = env.agents[0]
    agent.pos = (1, 1)
    # place a shelter at (2,2)
    env.shelters = [(2, 2)]

    # spawn a victim at agent position with known severity
    env.victims = []
    v = Victim(id=999, pos=agent.pos, severity=Severity.STABLE, spawn_time=env.current_step, max_wait=10)

    env.victims.append(v)

    # Step 1: pickup
    obs, rewards, done, info = env.step([ACTION_PICKUP])
    assert agent.carrying_victim_id == v.id

    # Step 2-3: move towards shelter (may need a couple moves)
    # compute moves until at shelter
    while agent.pos != env.shelters[0]:
        # simple movement: move right if x <, move down if y <
        ax, ay = agent.pos
        sx, sy = env.shelters[0]
        if ax < sx:
            action = 4  # ACTION_RIGHT
        elif ax > sx:
            action = 3  # ACTION_LEFT
        elif ay < sy:
            action = 2  # ACTION_DOWN
        elif ay > sy:
            action = 1  # ACTION_UP
        else:
            action = ACTION_STAY
        obs, rewards, done, info = env.step([action])

    # Now at shelter, perform dropoff
    obs, rewards, done, info = env.step([ACTION_DROPOFF])
    assert v.rescued is True or agent.carrying_victim is None


def test_victim_deterioration_and_death():
    env = DisasterReliefEnv(grid_size=5, num_agents=0, seed=2)
    env.reset()

    # Create a victim with tiny max_wait so it dies quickly
    v = Victim(id=1000, pos=(0, 0), severity=Severity.STABLE, spawn_time=env.current_step, max_wait=1)
    env.victims.append(v)

    # run a single step (no agents) and victim should deteriorate and die
    obs, rewards, done, info = env.step([] if env.num_agents == 0 else [ACTION_STAY]*env.num_agents)
    dead = any(vv.died for vv in env.victims if vv.id == 1000)
    assert dead is True
