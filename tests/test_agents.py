import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from env import DisasterReliefEnv
from agents import GreedyAgent, TaskExecutor
from env.constants import ACTION_PICKUP, ACTION_DROPOFF, ACTION_STAY, CellType


def test_greedy_agent_moves_to_nearest_victim():
    env = DisasterReliefEnv(grid_size=6, num_agents=1, num_shelters=1, num_depots=1, seed=10)
    env.reset()

    # place agent at (0,0) and victim at (2,0)
    env.agents[0].pos = (0, 0)
    # ensure a single victim exists
    env.victims = []
    env._spawn_victim()
    # override victim pos to a known location
    env.victims[0].pos = (2, 0)

    obs = env.get_observation()
    agent_info = obs['agents'][0]
    ga = GreedyAgent(agent_id=0)

    action = ga.act(agent_info, obs)
    # from (0,0) to (2,0) should be ACTION_RIGHT
    assert action == 4


def test_task_executor_rescue_flow():
    env = DisasterReliefEnv(grid_size=6, num_agents=1, num_shelters=1, num_depots=1, seed=5)
    env.reset()
    


    # Put agent and victim in known positions
    env.agents[0].pos = (1, 1)
    env.victims = []
    env._spawn_victim()
    env.victims[0].pos = (1, 1)
    victim_id = env.victims[0].id

    # ensure a shelter exists at a reachable position
    env.shelters = [(1, 2)]
    # clear obstacles to ensure path is clear

    executor = TaskExecutor(num_agents=1)
    executor.set_assignment(0, f'rescue V{victim_id}')

    obs = env.get_observation()
    actions = executor.act(obs)
    # since agent is on victim and not carrying, expect PICKUP
    assert actions[0] == ACTION_PICKUP

    # apply pickup
    obs, rewards, done, info = env.step([actions[0]])

    # Now carrying, next action should move towards shelter or DROPOFF if at shelter
    obs = env.get_observation()
    actions = executor.act(obs)
    # agent carrying; if not at shelter, action != STAY
    assert actions[0] != ACTION_STAY

    # Move repeatedly until dropoff
    for _ in range(10):
        obs, rewards, done, info = env.step(executor.act(env.get_observation()))
        if env.total_rescued > 0:
            break

    assert env.total_rescued >= 1
