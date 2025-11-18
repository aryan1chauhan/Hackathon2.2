import sys
import os
import time
import json
import random
from datetime import datetime

# ensure package path
sys.path.append(os.path.dirname(__file__))


from env import DisasterReliefEnv
from agents import GreedyAgent, MultiAgentController, TaskExecutor
from coordinator.llm_coordinator import LLMCoordinator, validate_llm_output
import argparse
from coordinator import SimpleCoordinator


def build_controller(env, mode='greedy', llm_client=None):
    if mode == 'greedy':
        agents = [GreedyAgent(agent_id=i) for i in range(env.num_agents)]
        return MultiAgentController(agents)
    elif mode == 'assigned':
        te = TaskExecutor(num_agents=env.num_agents)
        # default assignment: first agent rescues victim 0 (if exists)
        if env.victims:
            te.set_assignment(0, f'rescue V{env.victims[0].id}')
        return te
    elif mode == 'coordinated':
        te = TaskExecutor(num_agents=env.num_agents)
        # coordinator will be used in run loop to set assignments
        return te
    elif mode == 'llm':
        # LLMCoordinator will be used in run loop
        te = TaskExecutor(num_agents=env.num_agents)
        return te
    else:
        raise ValueError('unknown mode')


def run_episode(save_dir="examples/saved_runs", max_steps=200, seed=None, mode='greedy', llm_client=None):
    os.makedirs(save_dir, exist_ok=True)
    env = DisasterReliefEnv(seed=seed)
    obs = env.reset()

    trace = {
        'started_at': datetime.utcnow().isoformat() + 'Z',
        'grid_size': env.grid_size,
        'num_agents': env.num_agents,
        'steps': []
    }

    controller = build_controller(env, mode=mode, llm_client=llm_client)
    coordinator = None
    llm_coordinator = None
    if mode == 'coordinated':
        coordinator = SimpleCoordinator(assign_interval=1, urgency_weight=1.0)
    elif mode == 'llm':
        # Dummy LLM client for now (replace with real call)
        def dummy_llm_client(prompt):
            # Always assign 'wait' for all agents
            return json.dumps({str(i): 'wait' for i in range(env.num_agents)})
        llm_coordinator = LLMCoordinator(
            llm_client=llm_client or dummy_llm_client,
            validator=validate_llm_output
        )

    for step in range(max_steps):
        obs = env.get_observation()
        # LLM mode: get assignments from LLMCoordinator
        if llm_coordinator is not None:
            try:
                assignments = llm_coordinator.assign(env, obs)
                # Only TaskExecutor supports set_assignment
                from agents.simple_executor import TaskExecutor
                if isinstance(controller, TaskExecutor):
                    for aid, a in assignments.items():
                        if a is not None:
                            controller.set_assignment(int(aid), a)
            except Exception as e:
                # fallback: do nothing, controller will use previous assignments
                pass
        elif coordinator is not None:
            assigns = coordinator.assign(env, env.current_step)
            from agents.simple_executor import TaskExecutor
            if isinstance(controller, TaskExecutor):
                for aid, a in assigns.items():
                    if a is not None:
                        controller.set_assignment(aid, a)

        actions = controller.act(obs)
        obs, rewards, done, info = env.step(actions)

        trace['steps'].append({
            'step': info['step'],
            'actions': actions,
            'rewards': rewards,
            'info': info,
            'agents': obs['agents'],
            'victims': obs['victims']
        })

        if done:
            break

    trace['ended_at'] = datetime.utcnow().isoformat() + 'Z'
    trace['summary'] = {
        'total_spawned': env.total_spawned,
        'total_rescued': env.total_rescued,
        'total_died': env.total_died,
        'steps': env.current_step
    }

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = os.path.join(save_dir, f'run_{ts}.json')
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(trace, f, indent=2)

    return fname, trace


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['greedy', 'assigned', 'coordinated', 'llm'], default='greedy')
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    fname, trace = run_episode(max_steps=args.steps, seed=args.seed, mode=args.mode)
    print('Run saved to', fname)
    print('Summary:', trace['summary'])
