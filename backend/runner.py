import os
import sys
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from env import DisasterReliefEnv
from agents import GreedyAgent, MultiAgentController, TaskExecutor, PPOAgent
from coordinator import SimpleCoordinator
from coordinator.llm_coordinator import LLMCoordinator


def run_experiment(mode: str = 'greedy', max_steps: int = 200, save_dir: str = 'examples/saved_runs', seed: int | None = None, urgency_weight: float = 4.0, llm_client=None, validator=None, model_dir: str = 'models'):
    os.makedirs(save_dir, exist_ok=True)
    env = DisasterReliefEnv(seed=seed)
    env.reset()

    coordinator = None
    if mode == 'greedy':
        agents = [GreedyAgent(agent_id=i) for i in range(env.num_agents)]
        controller = MultiAgentController(agents)
    elif mode == 'ppo':
        agents = [PPOAgent(agent_id=i, model_path=os.path.join(model_dir, f'ppo_agent_{i}.zip')) for i in range(env.num_agents)]
        controller = MultiAgentController(agents)
    elif mode == 'assigned':
        controller = TaskExecutor(num_agents=env.num_agents)
        if env.victims:
            controller.set_assignment(0, f'rescue V{env.victims[0].id}')
    elif mode == 'coordinated':
        if llm_client is None or validator is None:
            raise ValueError("LLM client and validator must be provided for 'coordinated' mode")
        controller = TaskExecutor(num_agents=env.num_agents)
        coordinator = LLMCoordinator(llm_client=llm_client, validator=validator)
    elif mode == 'ppo_coordinated':
        if llm_client is None or validator is None:
            raise ValueError("LLM client and validator must be provided for 'ppo_coordinated' mode")
        # Use PPO agents but with LLM coordinator for assignments
        agents = [PPOAgent(agent_id=i, model_path=os.path.join(model_dir, f'ppo_agent_{i}.zip')) for i in range(env.num_agents)]
        controller = MultiAgentController(agents)
        coordinator = LLMCoordinator(llm_client=llm_client, validator=validator)
        # Note: For ppo_coordinated, the coordinator assigns tasks, but PPO agents decide actions autonomously
        # This might need adjustment in LLMCoordinator to work with PPO agents
    else:
        raise ValueError('unknown mode')

    trace = {
        'started_at': datetime.utcnow().isoformat() + 'Z',
        'mode': mode,
        'steps': []
    }

    for step_idx in range(max_steps):
        obs = env.get_observation()

        # If coordinated, ask coordinator for assignments and apply to TaskExecutor
        if coordinator is not None and mode in ['coordinated', 'ppo_coordinated']:
            assigns = coordinator.assign(env, obs)
            if mode == 'coordinated':
                for aid, a in assigns.items():
                    if a is not None:
                        controller.set_assignment(int(aid), a)
            # For ppo_coordinated, assignments are just for logging, PPO acts autonomously

        actions = controller.act(obs)
        obs, rewards, done, info = env.step(actions)

        trace['steps'].append({
            'step': info['step'],
            'actions': actions,
            'info': info
        })

        if done:
            break

    trace['ended_at'] = datetime.utcnow().isoformat() + 'Z'
    trace['summary'] = {
        'total_spawned': env.total_spawned,
        'total_rescued': env.total_rescued,
        'total_died': env.total_died,
        'steps': env.current_step,
        'avg_wait_time': sum(env.rescued_wait_times) / len(env.rescued_wait_times) if env.rescued_wait_times else 0,
        'agent_workloads': env.agent_workloads,
        'episode_rewards': env.episode_rewards
    }

    # Add coordinator metrics if available
    if coordinator is not None and hasattr(coordinator, 'successful_decisions'):
        trace['summary']['llm_success_rate'] = coordinator.successful_decisions / coordinator.total_decisions if coordinator.total_decisions > 0 else 0
        trace['summary']['fallback_count'] = coordinator.fallback_count

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = os.path.join(save_dir, f'backend_run_{mode}_{ts}.json')
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(trace, f, indent=2)

    return fname, trace

def dummy_llm_client(prompt):
    # This is a placeholder for an actual LLM client
    # In a real scenario, this would interact with an LLM API
    print(f"LLM Prompt: {prompt[:100]}...")
    # Return a dummy assignment for now
    return '{"0": "wait", "1": "wait"}'

def dummy_validator(llm_response, num_agents):
    # This is a placeholder for a real validator
    # It should parse the LLM response and ensure it's valid
    try:
        assignments = json.loads(llm_response)
        if not isinstance(assignments, dict):
            raise ValueError('Assignments must be a dict')
        for i in range(num_agents):
            if str(i) not in assignments:
                assignments[str(i)] = 'wait'
        return assignments
    except Exception as e:
        raise ValueError(f'Invalid LLM output: {e}')

if __name__ == '__main__':
    # Example of running greedy
    fname_greedy, trace_greedy = run_experiment('greedy', max_steps=200, seed=0)
    print('Saved greedy run to', fname_greedy)

    # Example of running PPO (assuming models are trained)
    try:
        fname_ppo, trace_ppo = run_experiment('ppo', max_steps=200, seed=0)
        print('Saved PPO run to', fname_ppo)
    except FileNotFoundError as e:
        print(f"PPO models not found: {e}. Train models first using scripts/train_ppo.py")

    # Example of running coordinated with LLM
    # For a real LLM, you would pass an actual client and validator
    fname_llm, trace_llm = run_experiment('coordinated', max_steps=200, seed=0, llm_client=dummy_llm_client, validator=dummy_validator)
    print('Saved LLM coordinated run to', fname_llm)
