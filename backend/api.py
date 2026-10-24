"""
FastAPI Backend Gateway for Disaster LLM Coordinator Platform.
Provides REST endpoints and WebSocket channels for simulation lifecycle management,
robust JSON serialization of NumPy arrays, delta event tracking, and metrics calculation.
"""

import os
import sys
import asyncio
from typing import Dict, Any, List, Optional
import numpy as np

# Ensure local module priority
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel

from env import DisasterReliefEnv, CellType, Severity
from agents import GreedyAgent, TaskExecutor, MultiAgentController
from backend.coordinator.simple_coordinator import SimpleCoordinator
from coordinator.llm_coordinator import LLMCoordinator


app = FastAPI(
    title="Disaster LLM Coordinator Platform Gateway",
    description="High-performance asynchronous simulation gateway and multi-agent coordination broker.",
    version="2.0.0",
)

# Enable CORS for local web visualizer and headless test scripts
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResetRequest(BaseModel):
    mode: str = "coordinated"  # "greedy", "coordinated", or "llm"
    seed: int = 42
    grid_size: int = 10
    num_agents: int = 3
    num_victims: int = 8
    num_obstacles: int = 12


class SimulationManager:
    """Manages active simulation instance and agent coordination state machine."""

    def __init__(self):
        self.env: Optional[DisasterReliefEnv] = None
        self.mode: str = "coordinated"
        self.seed: int = 42
        self.controller: Optional[Any] = None
        self.executor: Optional[TaskExecutor] = None
        self.coordinator: Optional[Any] = None
        self._interim_coordinator = SimpleCoordinator(assign_interval=1, urgency_weight=4.0)
        self._llm_task: Optional[asyncio.Task] = None
        self._cached_assignments: Dict[str, Optional[str]] = {}
        self.last_assignments: Dict[int, Optional[str]] = {}
        self.previous_rescued_ids: set = set()
        self.previous_died_ids: set = set()
        # Initialize default simulation on startup (EC-4.3)
        self.reset(ResetRequest())

    def reset(self, req: ResetRequest) -> Dict[str, Any]:
        self.mode = req.mode.lower()
        self.seed = req.seed

        if self._llm_task is not None and not self._llm_task.done():
            self._llm_task.cancel()
        self._llm_task = None
        self._cached_assignments = {}

        self.env = DisasterReliefEnv(
            grid_size=req.grid_size,
            num_agents=req.num_agents,
            max_victims=req.num_victims,
            num_shelters=2,
            num_depots=2,
            seed=self.seed,
        )
        obs = self.env.reset()
        self.previous_rescued_ids = set()
        self.previous_died_ids = set()
        self.last_assignments = {i: None for i in range(req.num_agents)}

        if self.mode == "greedy":
            self.controller = MultiAgentController([GreedyAgent(i) for i in range(req.num_agents)])
            self.executor = None
            self.coordinator = None
        elif self.mode == "coordinated":
            self.controller = None
            self.executor = TaskExecutor(num_agents=req.num_agents)
            self.coordinator = SimpleCoordinator(assign_interval=1, urgency_weight=4.0)
        elif self.mode == "llm":
            self.controller = None
            self.executor = TaskExecutor(num_agents=req.num_agents)
            self.coordinator = LLMCoordinator()
        else:
            # Default fallback to coordinated
            self.mode = "coordinated"
            self.controller = None
            self.executor = TaskExecutor(num_agents=req.num_agents)
            self.coordinator = SimpleCoordinator(assign_interval=1, urgency_weight=4.0)

        return self.get_serialized_state()

    def step(self) -> Dict[str, Any]:
        if self.env is None:
            self.reset(ResetRequest())

        obs = self.env.get_observation()
        current_step = self.env.step_count

        # 1. Determine agent actions based on active coordination architecture
        if self.mode == "greedy":
            actions = self.controller.act(obs)
            assignments = {i: "greedy_nearest" for i in range(len(self.env.agents))}
        elif self.mode == "coordinated":
            raw_assignments = self.coordinator.assign(obs, step=current_step)
            assignments = {}
            for aid_key, task in raw_assignments.items():
                aid = int(aid_key)
                assignments[aid] = task
                if task is not None and self.executor._targets[aid] is None:
                    self.executor.set_assignment(aid, task)
            actions = self.executor.act(obs)
        elif self.mode == "llm":
            # 1. Harvest finished background LLM task if ready
            if self._llm_task is not None and self._llm_task.done():
                try:
                    res = self._llm_task.result()
                    if isinstance(res, dict):
                        for k, v in res.items():
                            self._cached_assignments[str(k)] = v
                except Exception:
                    pass
                self._llm_task = None

            # 2. Check if a new async LLM task should be dispatched in background
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if self._llm_task is None:
                if loop is not None:
                    self._llm_task = loop.create_task(
                        asyncio.to_thread(self.coordinator.assign, obs, current_step)
                    )
                else:
                    # Synchronous fallback when running outside active asyncio event loop
                    res = self.coordinator.assign(obs, step=current_step)
                    if isinstance(res, dict):
                        for k, v in res.items():
                            self._cached_assignments[str(k)] = v

            # 3. Apply cached assignments or bridge with interim coordinator
            assignments = {}
            interim_needed = False
            active_vids = {v.id for v in self.env.victims if not v.rescued and not v.died}

            for aid in range(len(self.env.agents)):
                task = self._cached_assignments.get(str(aid))
                if self.executor._targets[aid] is None:
                    valid_task = False
                    if task and task.startswith("rescue V"):
                        try:
                            vid = int(task.replace("rescue V", "").strip())
                            if vid in active_vids:
                                self.executor.set_assignment(aid, task)
                                assignments[aid] = task
                                valid_task = True
                        except ValueError:
                            pass
                    if not valid_task:
                        interim_needed = True
                else:
                    assignments[aid] = self.executor.get_assignment(aid)

            # 4. Instantaneous zero-latency interim bridge for idle agents
            if interim_needed:
                interim = self._interim_coordinator.assign(obs, step=current_step)
                for aid_key, interim_task in interim.items():
                    aid = int(aid_key)
                    if self.executor._targets[aid] is None and interim_task is not None:
                        self.executor.set_assignment(aid, interim_task)
                        assignments[aid] = interim_task
                        self._cached_assignments[str(aid)] = interim_task

            actions = self.executor.act(obs)

        self.last_assignments = assignments

        # 2. Advance environment physics
        next_obs, rewards, done, info = self.env.step(actions)

        # 3. Compute delta events (EC-4.1)
        active_and_historical_victims = self.env.victims
        current_rescued_ids = {v.id for v in active_and_historical_victims if v.rescued}
        current_died_ids = {v.id for v in active_and_historical_victims if v.died}

        newly_rescued = list(current_rescued_ids - self.previous_rescued_ids)
        newly_died = list(current_died_ids - self.previous_died_ids)

        self.previous_rescued_ids = current_rescued_ids
        self.previous_died_ids = current_died_ids

        events = {
            "newly_rescued": newly_rescued,
            "newly_died": newly_died,
            "assignments": {str(k): v for k, v in self.last_assignments.items()},
            "done": bool(done),
        }

        return {
            "state": self.get_serialized_state(),
            "events": events,
            "metrics": self.compute_metrics(),
        }

    def compute_metrics(self) -> Dict[str, Any]:
        if self.env is None:
            return {}

        total_rescued = int(self.env.total_rescued)
        total_died = int(self.env.total_died)

        # Rescued victims wait time
        rescued_victims = [v for v in self.env.victims if v.rescued]
        if rescued_victims:
            avg_wait = float(np.mean([v.wait_time for v in rescued_victims]))
        else:
            avg_wait = 0.0

        # Jain's Fairness Index across agent rescues
        agent_rescues = [float(a.total_rescues) for a in self.env.agents]
        fairness = self._compute_jains_fairness(agent_rescues)

        agent_stats = [
            {
                "id": int(a.id),
                "pos": [int(a.pos[0]), int(a.pos[1])],
                "state": str(a.state),
                "carrying": int(a.carrying_victim.id) if a.carrying_victim else None,
                "total_rescues": int(a.total_rescues),
                "total_moves": int(a.total_moves),
                "current_task": self.last_assignments.get(a.id),
            }
            for a in self.env.agents
        ]

        active_victims_count = len(
            [v for v in self.env.victims if not v.rescued and not v.died]
        )

        return {
            "step": int(self.env.step_count),
            "mode": self.mode,
            "seed": int(self.seed),
            "total_rescued": total_rescued,
            "total_died": total_died,
            "active_victims": active_victims_count,
            "avg_wait_time": round(avg_wait, 2),
            "fairness_index": round(fairness, 4),
            "agent_stats": agent_stats,
        }

    @staticmethod
    def _compute_jains_fairness(workloads: List[float]) -> float:
        """
        Computes Jain's Fairness Index:
        J(x) = (sum(x_i))^2 / (n * sum(x_i^2))
        Convention: When sum(x_i) == 0, returns 1.0 (EC-4.2 / EC-05).
        """
        if not workloads:
            return 1.0
        total = sum(workloads)
        if total == 0:
            return 1.0
        n = len(workloads)
        sum_sq = sum(w * w for w in workloads)
        if sum_sq == 0:
            return 1.0
        return float((total * total) / (n * sum_sq))

    def get_serialized_state(self) -> Dict[str, Any]:
        """Serializes environment observation, converting all NumPy types to JSON primitives."""
        if self.env is None:
            return {}

        obs = self.env.get_observation()
        # Grid as formatted strings list (spec 1.4.2)
        grid_lines = list(obs["grid"])
        # Raw 2D matrix for direct coordinate lookups
        grid_matrix = [
            [int(self.env.grid[r, c]) for c in range(self.env.grid_size)]
            for r in range(self.env.grid_size)
        ]

        agents_serialized = [
            {
                "id": int(a["id"]),
                "pos": [int(a["pos"][0]), int(a["pos"][1])],
                "state": str(a["state"]),
                "carrying": int(a["carrying"]) if a["carrying"] is not None else None,
                "total_rescues": int(a.get("total_rescues", 0)),
                "total_moves": int(a.get("total_moves", 0)),
                "current_task": self.last_assignments.get(a["id"]),
            }
            for a in obs.get("agents", [])
        ]

        victims_serialized = [
            {
                "id": int(v["id"]),
                "pos": [int(v["pos"][0]), int(v["pos"][1])],
                "severity": int(v["severity"]),
                "severity_name": str(v["severity_name"]),
                "urgency": round(float(v["urgency"]), 2),
                "wait_time": int(v["wait_time"]),
                "max_wait": int(v["max_wait"]),
                "rescued": bool(v.get("rescued", False)),
                "died": bool(v.get("died", False)),
            }
            for v in obs.get("victims", [])
        ]

        return {
            "grid_size": int(obs["grid_size"]),
            "step": int(obs["step"]),
            "mode": self.mode,
            "grid": grid_lines,
            "grid_matrix": grid_matrix,
            "agents": agents_serialized,
            "victims": victims_serialized,
            "shelters": [
                {"id": int(s["id"]), "pos": [int(s["pos"][0]), int(s["pos"][1])]}
                for s in obs.get("shelters", [])
            ],
            "depots": [
                {"id": int(d["id"]), "pos": [int(d["pos"][0]), int(d["pos"][1])]}
                for d in obs.get("depots", [])
            ],
            "total_rescued": int(self.env.total_rescued),
            "total_died": int(self.env.total_died),
            "metrics": self.compute_metrics(),
        }


# Singleton manager
sim_manager = SimulationManager()


@app.post("/reset")
async def reset_simulation(req: Optional[ResetRequest] = None):
    """Resets environment and coordinator state machine."""
    if req is None:
        req = ResetRequest()
    state = sim_manager.reset(req)
    return state


@app.post("/step")
async def step_simulation():
    """Advances simulation by 1 step and returns state, events, and metrics without blocking on LLM."""
    result = sim_manager.step()
    return result


@app.get("/state")
def get_simulation_state():
    """Returns current serialized environment state."""
    return sim_manager.get_serialized_state()


@app.get("/metrics")
def get_simulation_metrics():
    """Returns current analytical metrics and agent workloads."""
    return sim_manager.compute_metrics()


# Mount frontend directory if present
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/frontend", StaticFiles(directory=frontend_dir, html=True), name="frontend")


@app.get("/")
def root_redirect():
    """Redirects root path to interactive dashboard if available, or returns API metadata."""
    if os.path.isdir(frontend_dir):
        return RedirectResponse(url="/frontend/index.html")
    return {
        "status": "online",
        "service": "Disaster LLM Coordinator Platform Gateway",
        "version": "2.0.0",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=False)
