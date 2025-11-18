import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from env import DisasterReliefEnv
from coordinator import SimpleCoordinator
from env.constants import Severity


from unittest.mock import MagicMock

def test_simple_coordinator_assigns_highest_urgency():
    env = DisasterReliefEnv(grid_size=6, num_agents=2, num_shelters=1, num_depots=1, seed=42)
    env.reset()

    # clear existing victims and create two victims with different urgencies
    env.victims = []

    v_a = MagicMock()
    v_a.id = 1
    v_a.urgency.return_value = 1
    v_a.pos = (0, 0)
    v_a.rescued = False
    v_a.died = False

    v_b = MagicMock()
    v_b.id = 2
    v_b.urgency.return_value = 10
    v_b.pos = (1, 1)
    v_b.rescued = False
    v_b.died = False

    env.victims = [v_a, v_b]

    # ensure both agents are idle
    for a in env.agents:
        a.carrying_victim = None

    coord = SimpleCoordinator(assign_interval=1)
    assigns = coord.assign(env, env.current_step)

    # expect at least one agent to be assigned to victim 2 (highest urgency)
    assert assigns[0] == 'rescue V2' or assigns[1] == 'rescue V2'
