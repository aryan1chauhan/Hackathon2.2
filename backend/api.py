
import asyncio
import json
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import threading
import time
from env import DisasterReliefEnv
from agents import GreedyAgent, MultiAgentController, TaskExecutor
from coordinator.llm_coordinator import LLMCoordinator, validate_llm_output

app = FastAPI()

# Mount static files for the frontend
app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")

# Allow CORS for all origins (for frontend dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global simulation state
SIM = {
    'env': None,
    'controller': None,
    'last_obs': None,
    'last_info': None,
    'last_trace': [],
    'mode': 'greedy',
    'llm_client': None
}

# --- Simulation helpers ---
def get_env():
    return SIM['env']

def get_controller():
    return SIM['controller']

def build_env(seed=None):
    env = DisasterReliefEnv(seed=seed)
    SIM['env'] = env
    return env

def build_controller(env, mode='greedy', llm_client=None):
    if mode == 'greedy':
        agents = [GreedyAgent(agent_id=i) for i in range(env.num_agents)]
        ctrl = MultiAgentController(agents)
    elif mode == 'assigned':
        ctrl = TaskExecutor(num_agents=env.num_agents)
    elif mode == 'llm':
        ctrl = TaskExecutor(num_agents=env.num_agents)
        SIM['llm_coordinator'] = LLMCoordinator(
            llm_client=llm_client or (lambda prompt: json.dumps({str(i): 'wait' for i in range(env.num_agents)})),
            validator=validate_llm_output
        )
    else:
        raise ValueError('Unknown mode')
    SIM['controller'] = ctrl
    return ctrl

# --- API Endpoints ---
@app.post("/reset")
def api_reset(payload: dict = {}):
    """
    Reset the simulation environment and controller.

    Args:
        payload (dict): Optional parameters including 'mode' (str) and 'seed' (int).

    Returns:
        dict: Response with 'ok' status and initial observation.
    """
    mode = payload.get('mode', 'greedy')
    seed = payload.get('seed', None)
    env = build_env(seed=seed)
    ctrl = build_controller(env, mode=mode)
    obs = env.reset()
    SIM['last_obs'] = obs
    SIM['last_info'] = {}
    SIM['last_trace'] = []
    SIM['mode'] = mode
    return {"ok": True, "obs": obs}

@app.post("/step")
def api_step(payload: dict):
    """
    Advance the simulation by one step.

    Args:
        payload (dict): Optional 'actions' list to override controller actions.

    Returns:
        dict: Updated observation, rewards, done flag, and info.
    """
    actions = payload.get('actions', None)
    env = get_env()
    ctrl = get_controller()
    obs = env.get_observation()
    # LLM mode: get assignments from coordinator
    if SIM['mode'] == 'llm' and 'llm_coordinator' in SIM:
        assignments = SIM['llm_coordinator'].assign(env, obs)
        if hasattr(ctrl, 'set_assignment'):
            for aid, a in assignments.items():
                ctrl.set_assignment(int(aid), a)
    if actions is None:
        actions = ctrl.act(obs)
    obs, rewards, done, info = env.step(actions)
    SIM['last_obs'] = obs
    SIM['last_info'] = info
    SIM['last_trace'].append({
        'actions': actions,
        'rewards': rewards,
        'info': info,
        'obs': obs
    })
    return {"obs": obs, "rewards": rewards, "done": done, "info": info}

@app.get("/state")
def api_state():
    """
    Get the current simulation state.

    Returns:
        dict: Current observation data.
    """
    env = get_env()
    obs = env.get_observation() if env else None
    return {"obs": obs}

@app.get("/metrics")
def api_metrics():
    """
    Get performance metrics for the current simulation.

    Returns:
        dict: Metrics including total spawned, rescued, died, and current step.
    """
    env = get_env()
    if not env:
        return {"error": "No environment"}
    return {
        "total_spawned": env.total_spawned,
        "total_rescued": env.total_rescued,
        "total_died": env.total_died,
        "current_step": env.current_step
    }

# --- WebSocket for live updates ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            env = get_env()
            obs = env.get_observation() if env else None
            await websocket.send_json({"obs": obs})
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass

# --- Run server ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
