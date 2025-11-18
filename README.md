# Disaster LLM Coordinator

A multi-agent simulation for disaster relief coordination using Large Language Models (LLMs) to optimize agent assignments and improve response efficiency.

## Features

- **Multi-Agent Simulation**: Agents navigate a grid-based environment to rescue victims and deliver supplies.
- **Multiple Coordination Modes**:
  - **Greedy**: Agents act independently based on local heuristics.
  - **Assigned**: Pre-assigned tasks for agents.
  - **Coordinated**: Simple rule-based coordinator assigns tasks.
  - **LLM**: Uses LLM for intelligent task assignment and coordination.
- **Real-time Visualization**: Web-based frontend for monitoring simulation progress.
- **Performance Metrics**: Tracks throughput, fairness, and wait times.
- **Extensible Architecture**: Modular design for adding new agents, coordinators, and environments.

## Setup

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd disaster_llm_coordinator
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Set up LLM client for LLM mode:
   - Configure your LLM API key in the coordinator configuration.

## Running the Simulation

### Command Line Demo

Run a single episode from the command line:

```bash
python run_demo.py --mode greedy --steps 200 --seed 42
```

Available modes: `greedy`, `assigned`, `coordinated`, `llm`

### Web Interface

Start the backend server:

```bash
python backend/api.py
```

Open your browser to `http://localhost:8000/frontend/index.html` to view the real-time simulation.

### Training PPO Agent

Train a PPO agent for improved performance:

```bash
python scripts/train_ppo.py
```

### Comparing Modes

Compare performance across different coordination modes:

```bash
python scripts/compare_modes.py
```

## API Endpoints

- `POST /reset`: Reset the simulation with optional mode and seed.
- `POST /step`: Advance the simulation by one step.
- `GET /state`: Get current simulation state.
- `GET /metrics`: Get performance metrics.
- `WebSocket /ws`: Real-time updates.

## Project Structure

```
disaster_llm_coordinator/
├── env/                    # Simulation environment
├── agents/                 # Agent implementations
├── coordinator/            # Coordination logic
├── backend/                # FastAPI backend
├── frontend/               # Web interface
├── scripts/                # Training and comparison scripts
├── tests/                  # Unit tests
├── examples/               # Saved runs and analysis
└── README.md
```

## Screenshots

*(Screenshots will be added after capturing from the running frontend)*

- Simulation Grid: Real-time visualization of agents (blue circles), victims (red circles), and depots (blue squares) on the grid.
- Metrics Dashboard: Performance metrics including throughput (total rescued), average wait time, fairness (Jain Index), and current step.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License
