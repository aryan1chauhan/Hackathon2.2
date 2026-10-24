"""
Cognitive LLM Coordinator with Schema Validation, Deduplication, and Heuristic Circuit Breaker.
"""

import json
import re
from typing import Dict, Any, Optional, Callable, Set
from agents.greedy_agent import manhattan_distance
from env.constants import STATE_CARRYING


def validate_llm_output(
    output_str: str,
    num_agents: int = 3,
    valid_victim_ids: Optional[Set[int]] = None,
) -> Dict[str, Optional[str]]:
    """
    Validates, parses, deduplicates, and sanitizes LLM task assignments.
    
    Edge Cases Handled:
    - Code fences and conversational preamble (EC-3.5)
    - Hallucinated victim IDs (EC-4.1)
    - Multiple agents assigned to the same victim (EC-4.2)
    - Missing agent keys or malformed structures
    """
    default_assignments: Dict[str, Optional[str]] = {str(i): "wait" for i in range(num_agents)}

    if not output_str or not isinstance(output_str, str):
        return default_assignments

    # 1. Extract JSON substring if surrounded by fences or markdown commentary
    text = output_str.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # Search for first '{' to last '}'
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = text[start : end + 1]
        else:
            json_str = text

    # 2. Parse JSON object
    try:
        data = json.loads(json_str)
        if not isinstance(data, dict):
            return default_assignments
    except Exception:
        return default_assignments

    # 3. Sanitize assignments and enforce schema
    sanitized: Dict[str, Optional[str]] = {}
    claimed_victims: Set[int] = set()

    for i in range(num_agents):
        # Look for key as string or integer
        raw_val = data.get(str(i), data.get(i, "wait"))
        if raw_val is None:
            sanitized[str(i)] = "wait"
            continue

        action_str = str(raw_val).strip()
        lower_action = action_str.lower()

        if lower_action.startswith("rescue v"):
            try:
                vid_part = lower_action.replace("rescue v", "").strip()
                vid = int(vid_part)

                # EC-4.1: Reject hallucinated victim IDs if active set is provided
                if valid_victim_ids is not None and vid not in valid_victim_ids:
                    sanitized[str(i)] = "wait"
                    continue

                # EC-4.2: Deduplicate target victim assignments
                if vid in claimed_victims:
                    # Target already claimed by a higher-priority agent; prevent race condition
                    sanitized[str(i)] = "wait"
                else:
                    claimed_victims.add(vid)
                    sanitized[str(i)] = f"rescue V{vid}"
            except ValueError:
                sanitized[str(i)] = "wait"
        elif lower_action == "wait" or lower_action == "none":
            sanitized[str(i)] = "wait"
        else:
            sanitized[str(i)] = "wait"

    return sanitized


def make_smart_llm_client() -> Callable[[str], str]:
    """
    Heuristic circuit breaker client.
    When external LLM APIs are unreachable, times out, or lack API keys (EC-4.3),
    generates optimal priority assignments in valid JSON format.
    """
    def fallback_client(prompt: str) -> str:
        # Extract victim IDs and urgencies from prompt summary if present
        victim_matches = re.findall(r"Victim (\d+).*?urgency=(\d+\.?\d*)", prompt)
        victim_urgencies = []
        for vid, urg in victim_matches:
            try:
                victim_urgencies.append((int(vid), float(urg)))
            except ValueError:
                pass

        # Sort victims by highest urgency
        victim_urgencies.sort(key=lambda x: -x[1])

        # Assign up to 3 agents
        assignments = {}
        for aid in range(3):
            if aid < len(victim_urgencies):
                assignments[str(aid)] = f"rescue V{victim_urgencies[aid][0]}"
            else:
                assignments[str(aid)] = "wait"

        return json.dumps(assignments)

    return fallback_client


def make_gemini_client(api_key: Optional[str] = None) -> Callable[[str], str]:
    """
    Constructs a live Gemini API client using google-generativeai.
    Falls back cleanly to circuit breaker if request fails.
    """
    import os
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    key = api_key or os.getenv("GEMINI_API_KEY")

    def gemini_caller(prompt: str) -> str:
        if not key:
            raise ValueError("GEMINI_API_KEY not configured")

        import google.generativeai as genai
        genai.configure(api_key=key)

        candidate_models = ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-1.5-flash", "gemini-flash-latest"]
        last_err = None

        for model_name in candidate_models:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    prompt,
                    generation_config={"temperature": 0.1, "max_output_tokens": 150},
                    request_options={"timeout": 5.0},
                )
                if response and hasattr(response, "text") and response.text:
                    return response.text
            except Exception as e:
                last_err = e
                continue

        if last_err is not None:
            raise last_err
        raise RuntimeError("No candidate Gemini model succeeded")

    return gemini_caller


class LLMCoordinator:
    """
    Cognitive multi-agent coordinator that summarizes disaster world state,
    queries an LLM, sanitizes outputs with deduplication, and falls back to
    a circuit breaker heuristic upon API unavailability.
    """

    def __init__(
        self,
        llm_client: Optional[Callable[[str], str]] = None,
        validator: Optional[Callable] = None,
        api_key: Optional[str] = None,
        assign_interval: int = 3,
    ):
        import os
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        resolved_key = api_key or os.getenv("GEMINI_API_KEY")

        if llm_client is not None:
            self.llm_client = llm_client
        elif resolved_key:
            self.llm_client = make_gemini_client(api_key=resolved_key)
        else:
            self.llm_client = make_smart_llm_client()

        self.validator = validator if validator is not None else validate_llm_output
        self.circuit_breaker = make_smart_llm_client()
        self.assign_interval = assign_interval
        self.last_assigned_step = -1
        self.current_assignments: Dict[str, Optional[str]] = {}

    def summarize_state(self, obs: Dict[str, Any]) -> str:
        """Serializes current disaster observation into concise markdown state summary."""
        agents = obs.get("agents", [])
        victims = [
            v for v in obs.get("victims", [])
            if not v.get("rescued", False) and not v.get("died", False)
        ]

        lines = [
            f"Disaster Simulation State (Grid {obs.get('grid_size', 10)}x{obs.get('grid_size', 10)}, Step {obs.get('step', 0)}):",
            "Agents:"
        ]
        for a in agents:
            carrying = a.get("carrying")
            status = f"carrying V{carrying}" if carrying is not None else "idle"
            lines.append(f"- Agent {a['id']} at {a['pos']}: {status}, total_rescues={a.get('total_rescues', 0)}")

        lines.append("Active Victims:")
        if not victims:
            lines.append("- None (All victims either rescued or deceased)")
        else:
            for v in victims:
                lines.append(
                    f"- Victim {v['id']} at {v['pos']}: severity={v.get('severity_name', 'UNKNOWN')}, "
                    f"urgency={v.get('urgency', 0.0)}, wait={v.get('wait_time', 0)}/{v.get('max_wait', 40)}"
                )

        return "\n".join(lines)

    def build_prompt(self, summary: str, num_agents: int) -> str:
        """Constructs prompt enforcing strict JSON output and coordination principles."""
        return (
            "You are the Strategic Coordinator for a multi-agent disaster relief operation.\n"
            "Given the current simulation state, assign each available agent a unique target victim to rescue, "
            "or 'wait' if carrying a victim or no victims are left.\n\n"
            "Rules:\n"
            "1. Prioritize victims with highest urgency and closest distance.\n"
            "2. Never assign multiple agents to the same victim.\n"
            "3. If an agent is already carrying a victim, assign 'wait'.\n"
            "4. Output MUST be ONLY a single JSON object mapping agent IDs as strings to tasks.\n"
            f'Format example: {{"0": "rescue V1", "1": "wait", "2": "rescue V3"}}\n\n'
            f"{summary}\n\n"
            "JSON Response:"
        )

    def assign(self, env_or_obs: Any, step: int = 0) -> Dict[str, Optional[str]]:
        """
        Executes end-to-end cognitive coordination loop with circuit breaker fallback.
        """
        # Normalize observation
        if isinstance(env_or_obs, dict):
            obs = env_or_obs
        elif hasattr(env_or_obs, "get_observation"):
            obs = env_or_obs.get_observation()
        else:
            raise ValueError("env_or_obs must be DisasterReliefEnv or observation dict")

        num_agents = len(obs.get("agents", []))
        active_victims = [
            v for v in obs.get("victims", [])
            if not v.get("rescued", False) and not v.get("died", False)
        ]
        valid_vids: Set[int] = {v["id"] for v in active_victims}

        # Check cached macro-plan if within assign_interval and targets remain valid
        if (
            self.last_assigned_step >= 0
            and step - self.last_assigned_step < self.assign_interval
            and self.current_assignments
        ):
            all_valid = True
            for aid_str, task_str in self.current_assignments.items():
                if task_str and task_str.startswith("rescue V"):
                    try:
                        vid = int(task_str.replace("rescue V", ""))
                        if vid not in valid_vids:
                            all_valid = False
                            break
                    except ValueError:
                        pass
            if all_valid:
                return self.current_assignments

        summary = self.summarize_state(obs)
        prompt = self.build_prompt(summary, num_agents)

        raw_response = ""
        try:
            raw_response = self.llm_client(prompt)
        except Exception:
            # EC-4.3: LLM API Failure -> trigger circuit breaker fallback
            raw_response = self.circuit_breaker(prompt)

        # Validate, deduplicate, and eliminate hallucinated IDs
        assignments = self.validator(
            raw_response,
            num_agents=num_agents,
            valid_victim_ids=valid_vids,
        )

        self.current_assignments = assignments
        self.last_assigned_step = step
        return assignments
