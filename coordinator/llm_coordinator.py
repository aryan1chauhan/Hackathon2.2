"""
LLM Coordinator for disaster relief simulation.

This module provides coordination logic using Large Language Models (LLMs)
to assign tasks to agents in a multi-agent disaster relief scenario.
"""

import json
from agents import GreedyAgent


class LLMCoordinator:
    def __init__(self, llm_client, validator, fallback=None):
        self.llm_client = llm_client
        self.validator = validator
        self.fallback = fallback or GreedyAgent
        self.last_valid_assignments = None
        # Track agent workload for fairness
        self.agent_workload = None
        # Metrics tracking
        self.successful_decisions = 0
        self.total_decisions = 0
        self.fallback_count = 0

    def summarize_state(self, env, obs):
        # Track agent workload (rescues performed)
        if self.agent_workload is None:
            self.agent_workload = {a['id']: 0 for a in obs['agents']}
        # Add health, distance, and workload for fairness/efficiency
        agents_summary = []
        for a in obs['agents']:
            agent_id = a['id']
            pos = a['pos']
            has_victim = a.get('has_victim', a.get('carrying', None))
            workload = self.agent_workload.get(agent_id, 0)
            agents_summary.append({'id': agent_id, 'pos': pos, 'has_victim': has_victim, 'workload': workload})
        victims_summary = []
        for v in obs['victims']:
            v_id = v['id']
            vpos = v['pos']
            urgency = v['urgency']
            # Compute min distance to any agent
            min_dist = min(abs(vpos[0]-a['pos'][0]) + abs(vpos[1]-a['pos'][1]) for a in obs['agents']) if obs['agents'] else 0
            victims_summary.append({'id': v_id, 'pos': vpos, 'urgency': urgency, 'min_agent_dist': min_dist})
        return {
            'step': env.current_step,
            'agents': agents_summary,
            'victims': victims_summary,
            'depots': env.depots,
            'shelters': env.shelters,
            'grid_size': env.grid_size
        }

    def build_prompt(self, state_summary):
        # Build a prompt for the LLM with efficiency and fairness objectives
        prompt = (
            "Disaster scenario:\n" +
            json.dumps(state_summary, indent=2) +
            "\n\nAssign each agent a task in JSON. Example: {\"0\": \"rescue V1\", \"1\": \"wait\"}\n"
            "Valid tasks: 'rescue Vx', 'wait', 'return', etc.\n"
            "Instructions: Prioritize rescuing victims with highest urgency and closest to agents. Distribute tasks so all agents participate equally over time (fairness). Avoid assigning multiple agents to the same victim unless necessary."
        )
        return prompt

    def postprocess_assignments(self, assignments, obs):
        # Ensure unique assignments (no duplicate victim assignments unless needed)
        assigned_victims = set()
        victim_ids = {str(v['id']) for v in obs.get('victims', [])}
        agent_ids = [str(a['id']) for a in obs.get('agents', [])]
        new_assignments = {}
        for aid in agent_ids:
            a_task = assignments.get(aid, 'wait')
            if a_task.startswith('rescue V'):
                vid = a_task.split('V')[-1]
                if vid in assigned_victims:
                    new_assignments[aid] = 'wait'
                elif vid not in victim_ids:
                    new_assignments[aid] = 'wait'
                else:
                    new_assignments[aid] = a_task
                    assigned_victims.add(vid)
            else:
                new_assignments[aid] = a_task
        # Fairness: if some agents are idle and others are overloaded, reassign idle agents to unassigned victims
        unassigned_victims = victim_ids - assigned_victims
        idle_agents = [aid for aid in agent_ids if new_assignments[aid] == 'wait']
        for aid, vid in zip(idle_agents, unassigned_victims):
            new_assignments[aid] = f'rescue V{vid}'
        return new_assignments

    def update_workload(self, obs):
        # Update agent workload (rescues performed)
        if self.agent_workload is None:
            self.agent_workload = {a['id']: 0 for a in obs['agents']}
        for a in obs['agents']:
            agent_id = a['id']
            # If agent just dropped off a victim, increment workload
            # (Assume info about last action is not available, so this is a placeholder)
            # In a real system, track from env info['events'] or similar
            pass

    def assign(self, env, obs):
        self.update_workload(obs)
        state_summary = self.summarize_state(env, obs)
        prompt = self.build_prompt(state_summary)
        self.total_decisions += 1
        try:
            llm_response = self.llm_client(prompt)
            assignments = self.validator(llm_response, env.num_agents)
            # Post-process for efficiency/fairness
            assignments = self.postprocess_assignments(assignments, obs)
            self.last_valid_assignments = assignments
            self.successful_decisions += 1
            return assignments
        except Exception:
            # Fallback: use GreedyAgent for this step
            self.fallback_count += 1
            agents = [self.fallback(agent_id=i) for i in range(env.num_agents)]
            actions = {}
            agents_info = {a['id']: a for a in obs['agents']}
            for i, agent in enumerate(agents):
                agent_info = agents_info.get(i, {})
                actions[i] = agent.act(agent_info, obs)
            # Convert actions to assignment strings (e.g., 'rescue Vx' or 'wait')
            return {i: 'wait' for i in range(env.num_agents)}

# Example validator

def validate_llm_output(llm_response, num_agents):
    try:
        assignments = json.loads(llm_response)
        if not isinstance(assignments, dict):
            raise ValueError('Assignments must be a dict')
        # Ensure all agent ids are present
        for i in range(num_agents):
            if str(i) not in assignments:
                assignments[str(i)] = 'wait'
        return assignments
    except Exception as e:
        raise ValueError(f'Invalid LLM output: {e}')
