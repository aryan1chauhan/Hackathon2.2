const simulationGrid = document.getElementById('simulationGrid');
const startButton = document.getElementById('startButton');
const pauseButton = document.getElementById('pauseButton');
const stepButton = document.getElementById('stepButton');
const resetButton = document.getElementById('resetButton');
const modeSelect = document.getElementById('modeSelect');
const throughputSpan = document.getElementById('throughput');
const avgWaitTimeSpan = document.getElementById('avgWaitTime');
const fairnessSpan = document.getElementById('fairness');
const currentStepSpan = document.getElementById('currentStep');
const logOutput = document.getElementById('logOutput');

let GRID_SIZE = 10; // Default, will be updated dynamically
let simulationInterval = null;
let isPaused = false;

// Function to initialize the grid
function initializeGrid() {
    simulationGrid.style.gridTemplateColumns = `repeat(${GRID_SIZE}, 1fr)`;
    simulationGrid.innerHTML = ''; // Clear existing grid
    for (let i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
        const cell = document.createElement('div');
        cell.classList.add('grid-cell');
        cell.dataset.index = i;
        simulationGrid.appendChild(cell);
    }
}

// Function to render the simulation state
function renderSimulation(state) {
    // Clear previous entities and assignments
    document.querySelectorAll('.grid-cell').forEach(cell => {
        cell.innerHTML = '';
        cell.classList.remove('agent', 'victim', 'depot');
    });

    // Render agents, victims, depots
    state.grid.forEach((row, y) => {
        row.forEach((cellContent, x) => {
            const cell = simulationGrid.children[y * GRID_SIZE + x];
            if (cell) {
                cell.innerHTML = ''; // Clear previous content
                cell.classList.remove('agent', 'victim', 'depot'); // Clear previous types

                if (cellContent) {
                    const entityType = cellContent.type;
                    const entityId = cellContent.id;

                    cell.classList.add(entityType);
                    cell.textContent = entityId;
                }
            }
        });
    });

    // Assignments not available in current backend, skip for now

            // Update metrics
    throughputSpan.textContent = state.metrics.throughput;
    avgWaitTimeSpan.textContent = state.metrics.avgWaitTime;
    fairnessSpan.textContent = state.metrics.fairness;
    currentStepSpan.textContent = state.metrics.currentStep || 0;

    // Update logs
    state.logs.forEach(log => {
        const p = document.createElement('p');
        p.textContent = log;
        logOutput.appendChild(p);
    });
    logOutput.scrollTop = logOutput.scrollHeight; // Scroll to bottom
}

// Base URL for the backend API
const API_BASE_URL = ''; // Use relative path since frontend is served by backend

// Function to fetch and render the simulation state from the backend
async function fetchSimulationState() {
    try {
        const stateResponse = await fetch(`${API_BASE_URL}/state`);
        const stateData = await stateResponse.json();

        const metricsResponse = await fetch(`${API_BASE_URL}/metrics`);
        const metricsData = await metricsResponse.json();

        // Combine data into a format renderSimulation expects
        const simulationData = {
            grid: [], // This will be populated from stateData.obs.grid
            entities: [], // This will be populated from stateData.obs.agents, victims, depots
            assignments: [], // This will be populated from stateData.obs.assignments
            metrics: {
                throughput: metricsData.total_rescued, // Assuming total_rescued is throughput
                avgWaitTime: 'N/A', // Backend doesn't provide this directly yet
                fairness: 'N/A', // Backend doesn't provide this directly yet
                currentStep: metricsData.current_step
            },
            logs: [] // Logs will be fetched separately or pushed via WebSocket
        };

        if (stateData.obs) {
            const obs = stateData.obs;
            GRID_SIZE = obs.grid_size || 10; // Update GRID_SIZE dynamically
            initializeGrid(); // Reinitialize grid if size changed
            const grid = Array(GRID_SIZE).fill(null).map(() => Array(GRID_SIZE).fill(null));

            // Populate grid with agents, victims, depots
            obs.agents.forEach(agent => {
                grid[agent.pos[0]][agent.pos[1]] = { type: 'agent', id: `A${agent.id}` };
                simulationData.entities.push({ id: `A${agent.id}`, type: 'agent', x: agent.pos[1], y: agent.pos[0] });
            });
            obs.victims.forEach(victim => {
                grid[victim.pos[0]][victim.pos[1]] = { type: 'victim', id: `V${victim.id}` };
                simulationData.entities.push({ id: `V${victim.id}`, type: 'victim', x: victim.pos[1], y: victim.pos[0] });
            });
            obs.depots.forEach(depot => {
                grid[depot.pos[0]][depot.pos[1]] = { type: 'depot', id: `D${depot.id}` };
                simulationData.entities.push({ id: `D${depot.id}`, type: 'depot', x: depot.pos[1], y: depot.pos[0] });
            });

            simulationData.grid = grid;

            // Populate assignments (this structure might need adjustment based on actual backend output)
            // Assuming assignments are part of the observation or a separate endpoint
            // For now, let's mock some assignments if not available
            if (obs.assignments) {
                obs.assignments.forEach(assignment => {
                    // Assuming assignment format is { agent_id: ..., target_id: ..., agent_pos: [y,x], target_pos: [y,x] }
                    const agent = simulationData.entities.find(e => e.id === `A${assignment.agent_id}`);
                    const target = simulationData.entities.find(e => e.id === `V${assignment.target_id}` || e.id === `D${assignment.target_id}`);
                    if (agent && target) {
                        simulationData.assignments.push({ agent: agent, target: target });
                    }
                });
            }
        }

        renderSimulation(simulationData);
    } catch (error) {
        console.error('Error fetching simulation state:', error);
        logOutput.innerHTML += `<p style="color: red;">Error: ${error.message}</p>`;
    }
}

// Control buttons functionality
startButton.addEventListener('click', async () => {
    if (!simulationInterval) {
        // Reset the simulation on start
        try {
            const mode = modeSelect.value;
            await fetch(`${API_BASE_URL}/reset`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ mode: mode }) });
            logOutput.innerHTML += `<p>Simulation reset and started in ${mode} mode.</p>`;
            fetchSimulationState(); // Fetch initial state after reset
            isPaused = false;
            simulationInterval = setInterval(fetchSimulationState, 1000); // Fetch every second
        } catch (error) {
            console.error('Error resetting simulation:', error);
            logOutput.innerHTML += `<p style="color: red;">Error resetting simulation: ${error.message}</p>`;
        }
    } else if (isPaused) {
        isPaused = false;
        simulationInterval = setInterval(fetchSimulationState, 1000); // Resume fetching
        logOutput.innerHTML += '<p>Simulation resumed.</p>';
    }
});

resetButton.addEventListener('click', async () => {
    // Stop any running simulation
    if (simulationInterval) {
        clearInterval(simulationInterval);
        simulationInterval = null;
        isPaused = false;
    }
    try {
        const mode = modeSelect.value;
        await fetch(`${API_BASE_URL}/reset`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ mode: mode }) });
        logOutput.innerHTML += `<p>Simulation reset in ${mode} mode.</p>`;
        fetchSimulationState(); // Fetch initial state after reset
    } catch (error) {
        console.error('Error resetting simulation:', error);
        logOutput.innerHTML += `<p style="color: red;">Error resetting simulation: ${error.message}</p>`;
    }
});

pauseButton.addEventListener('click', () => {
    if (simulationInterval && !isPaused) {
        clearInterval(simulationInterval);
        simulationInterval = null;
        isPaused = true;
        logOutput.innerHTML += '<p>Simulation paused.</p>';
    }
});

stepButton.addEventListener('click', async () => {
    // Only step if paused or not started
    if (isPaused || !simulationInterval) {
        try {
            await fetch(`${API_BASE_URL}/step`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) });
            logOutput.innerHTML += '<p>Simulation stepped forward.</p>';
            fetchSimulationState(); // Fetch state after step
        } catch (error) {
            console.error('Error stepping simulation:', error);
            logOutput.innerHTML += `<p style="color: red;">Error stepping simulation: ${error.message}</p>`;
        }
    }
});

// Initial setup
initializeGrid();
fetchSimulationState(); // Fetch initial state on load
