# Disaster LLM Coordinator

A multi-agent simulation framework for disaster relief coordination, resource allocation, and LLM-assisted multi-agent operations.

## Features

- **Multi-Agent Simulation**: Autonomous search-and-rescue agents navigate a dynamic 2D grid to locate victims, provide immediate triage, and transport casualties to safety depots.
- **Multiple Coordination Modes**:
  - **Greedy**: Fully decentralized heuristics where each agent targets the nearest victim.
  - **Assigned**: Static partition/assignment for agents.
  - **Coordinated**: Centralized global coordination utilizing Hungarian minimum-cost bipartite matching with distance and urgency balancing.
  - **LLM**: Gemini LLM-driven strategic coordination with JSON validation, fallback circuit-breaking, and prompt-engineered dynamic task allocation.
  - **PPO Reinforcement Learning**: Trained fleet policies using Stable-Baselines3 Gymnasium environment wrapper.
- **Tactical Web Dashboard**: Real-time interactive browser interface with live grid visualizer, mode switching, telemetry, and metrics charts.
- **Fairness & Throughput Analytics**: Live tracking of victim survival rate, average wait time, and Jain's Fairness Index across agent workloads.
- **Resilient Architecture**: Collision-free pathfinding, dynamic obstacle avoidance, and edge case recovery.

## Prerequisites & Setup

### Requirements
- Python 3.10+ (tested on Python 3.12/3.13)
- pip

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/aryan1chauhan/Hackathon2.2.git
   cd Hackathon2.2
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Set up Gemini API Key for LLM mode:
   Create a `.env` file in the root directory (or export to your environment):
   ```bash
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

## Quickstart & Usage

### 1. Interactive Tactical Web Visualizer
Start the FastAPI server:
```bash
python backend/api.py
```
Open **`http://localhost:8000/frontend/index.html`** in your browser to run live interactive simulations.

### 2. Headless Simulation CLI
Run a command-line simulation episode:
```bash
python run_demo.py --mode coordinated --steps 40 --seed 42
```
Available modes: `greedy`, `assigned`, `coordinated`, `llm`

Launcher scripts are also available:
- Windows: `run_demo.bat --mode coordinated --steps 40`
- Linux/macOS: `./run_demo.sh --mode coordinated --steps 40`

### 3. Multi-Episode Comparative Benchmark
Benchmark and compare all coordination modes across multiple seeds:
```bash
python scripts/compare_modes.py
```

### 4. Reinforcement Learning Training (PPO)
Train the PPO multi-agent policy:
```bash
python scripts/train_ppo.py --timesteps 10000
```
Trained model weights are saved to `models/ppo_agent.zip`.

### 5. Running the Test Suite
Run the comprehensive verification suite:
```bash
python tests/run_tests.py
```
Or with pytest:
```bash
pytest tests/ -v
```

## Project Structure

```
Hackathon2.2/
├── agents/                 # Autonomous agent implementations (Greedy, PPO, Executor)
│   ├── agent_interface.py  # Base agent abstract interface
│   ├── greedy_agent.py     # Local nearest-neighbor heuristic agent
│   ├── ppo_agent.py        # Stable-Baselines3 PPO RL agent
│   ├── pathfinding.py      # Collision-free grid navigation
│   └── simple_executor.py  # Discrete task executor for coordinator commands
├── backend/                # FastAPI backend & runner
│   ├── api.py              # REST API endpoints & WebSocket streamer
│   ├── runner.py           # Simulation execution runner
│   └── coordinator/        # Coordinator integration service
├── coordinator/            # Central multi-agent coordinators
│   └── llm_coordinator.py  # Hungarian matching & Gemini LLM coordinator
├── env/                    # Disaster simulation environment
│   ├── disaster_env.py     # Grid world dynamics, triage, & physics
│   ├── constants.py        # Environment parameters & configurations
│   └── spawn_policies.py   # Victim & obstacle spawning policies
├── frontend/               # Interactive tactical web interface
│   ├── index.html          # Dashboard UI structure
│   ├── style.css           # Styling & tactical grid visualization
│   └── script.js           # API polling, controls, & Canvas rendering
├── models/                 # Pretrained RL models (ppo_agent.zip)
├── scripts/                # Benchmark & training utilities
├── tests/                  # Full verification & edge-case test suite
└── requirements.txt        # Project dependencies
```

## API Endpoints

- `POST /reset`: Initialize simulation with selected mode, grid size, and seed.
- `POST /step`: Advance simulation by one tick and return updated state.
- `GET /state`: Fetch current agent positions, victims, and environment grid.
- `GET /metrics`: Fetch performance metrics (rescues, casualties, fairness index).
- `WebSocket /ws`: Real-time streaming simulation channel.

## License

MIT License
