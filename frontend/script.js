/**
 * AegisCoord Tactical Operations Center - Interactive Frontend Controller
 * Connects to the FastAPI backend gateway, renders dynamic 10x10 grid canvas/DOM,
 * handles user controls, updates live KPI counters, and streams incident logs.
 */

const API_BASE = window.location.origin;

// Application State
const state = {
  mode: "coordinated",
  isRunning: false,
  stepDelay: 350,
  seed: 42,
  timerId: null,
  isProcessing: false,
  previousRescued: 0,
  previousCasualties: 0,
};

// DOM Elements
const elements = {
  gridContainer: document.getElementById("grid-container"),
  statusBadge: document.getElementById("system-status-badge"),
  statusDot: document.getElementById("system-status-dot"),
  statusText: document.getElementById("system-status-text"),
  activeModeTag: document.getElementById("active-mode-tag"),
  kpiStep: document.getElementById("kpi-step"),
  kpiRescued: document.getElementById("kpi-rescued"),
  kpiTrendRescued: document.getElementById("kpi-trend-rescued"),
  kpiCasualties: document.getElementById("kpi-casualties"),
  kpiTrendCasualties: document.getElementById("kpi-trend-casualties"),
  kpiActive: document.getElementById("kpi-active"),
  kpiWait: document.getElementById("kpi-wait"),
  kpiFairness: document.getElementById("kpi-fairness"),
  fairnessBar: document.getElementById("fairness-bar"),
  agentRoster: document.getElementById("agent-roster"),
  consoleStream: document.getElementById("console-stream"),
  btnPlayPause: document.getElementById("btn-play-pause"),
  btnPlayIcon: document.getElementById("btn-play-icon"),
  btnPlayLabel: document.getElementById("btn-play-label"),
  btnStep: document.getElementById("btn-step"),
  btnReset: document.getElementById("btn-reset"),
  speedSlider: document.getElementById("speed-slider"),
  speedLabel: document.getElementById("speed-label"),
  seedInput: document.getElementById("seed-input"),
  btnSeedRandom: document.getElementById("btn-seed-random"),
  btnClearLog: document.getElementById("btn-clear-log"),
  modeButtons: {
    coordinated: document.getElementById("btn-mode-coordinated"),
    greedy: document.getElementById("btn-mode-greedy"),
    llm: document.getElementById("btn-mode-llm"),
  },
};

// 1. Grid Construction: Build 100 cell DOM nodes
function initializeGrid(gridSize = 10) {
  elements.gridContainer.innerHTML = "";
  for (let y = 0; y < gridSize; y++) {
    for (let x = 0; x < gridSize; x++) {
      const cell = document.createElement("div");
      cell.className = "cell";
      cell.id = `cell-${x}-${y}`;
      cell.dataset.x = x;
      cell.dataset.y = y;

      const coord = document.createElement("span");
      coord.className = "cell-coord";
      coord.textContent = `${x},${y}`;
      cell.appendChild(coord);

      elements.gridContainer.appendChild(cell);
    }
  }
}

// 2. Logging Stream
function appendLog(tag, message, category = "system") {
  const entry = document.createElement("div");
  entry.className = "log-entry";

  const now = new Date();
  const timeStr = now.toTimeString().split(" ")[0];

  entry.innerHTML = `
    <span class="log-time">[${timeStr}]</span>
    <span class="log-tag tag-${category}">${tag}</span>
    <span class="log-msg">${message}</span>
  `;

  elements.consoleStream.appendChild(entry);
  elements.consoleStream.scrollTop = elements.consoleStream.scrollHeight;
}

// 3. API Communication
async function apiReset(mode = state.mode, seed = state.seed) {
  try {
    const res = await fetch(`${API_BASE}/reset`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode, seed }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    state.previousRescued = 0;
    state.previousCasualties = 0;
    renderFullState(data);
    appendLog(
      "RESET",
      `Simulation initialized in ${mode.toUpperCase()} mode (Seed: ${seed})`,
      "system"
    );
  } catch (err) {
    appendLog("ERROR", `Failed to reset simulation: ${err.message}`, "casualty");
  }
}

async function apiStep() {
  if (state.isProcessing) return;
  state.isProcessing = true;

  try {
    const res = await fetch(`${API_BASE}/step`, { method: "POST" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    renderFullState(data.state);
    processDeltaEvents(data.events);

    if (data.events && data.events.done) {
      pauseSimulation();
      appendLog("TERMINAL", "Episode completed: all victims handled or max steps reached.", "system");
      elements.statusText.textContent = "EPISODE FINISHED";
    }
  } catch (err) {
    appendLog("ERROR", `Step failure: ${err.message}`, "casualty");
    pauseSimulation();
  } finally {
    state.isProcessing = false;
  }
}

// 4. State Rendering
function renderFullState(simState) {
  if (!simState) return;

  const { grid_matrix, agents, victims, shelters, depots, step, metrics } = simState;

  // Render KPIs
  elements.kpiStep.textContent = step;
  if (metrics) {
    elements.kpiRescued.textContent = metrics.total_rescued;
    elements.kpiCasualties.textContent = metrics.total_died;
    elements.kpiActive.textContent = metrics.active_victims;
    elements.kpiWait.innerHTML = `${metrics.avg_wait_time}<span class="kpi-unit">s</span>`;
    elements.kpiFairness.textContent = metrics.fairness_index.toFixed(4);

    const fairnessPct = Math.min(100, Math.max(0, metrics.fairness_index * 100));
    elements.fairnessBar.style.width = `${fairnessPct}%`;

    // Render Telemetry
    renderAgentRoster(metrics.agent_stats || []);
  }

  // Clear cells (preserve coordinate text)
  const allCells = elements.gridContainer.querySelectorAll(".cell");
  allCells.forEach((c) => {
    c.className = "cell";
    const existingTokens = c.querySelectorAll(".victim-token, .agent-token");
    existingTokens.forEach((t) => t.remove());
  });

  // Render static terrain (shelters, depots, obstacles)
  if (grid_matrix) {
    for (let y = 0; y < grid_matrix.length; y++) {
      for (let x = 0; x < grid_matrix[y].length; x++) {
        const cell = document.getElementById(`cell-${x}-${y}`);
        if (!cell) continue;
        const cellType = grid_matrix[y][x];
        if (cellType === 1) cell.classList.add("obstacle");
        else if (cellType === 2) cell.classList.add("shelter");
        else if (cellType === 3) cell.classList.add("depot");
      }
    }
  }

  // Render active victims
  if (victims && Array.isArray(victims)) {
    victims.forEach((v) => {
      if (v.rescued || v.died) return;
      const [vx, vy] = v.pos;
      const cell = document.getElementById(`cell-${vx}-${vy}`);
      if (!cell) return;

      const vToken = document.createElement("div");
      const sevClass = v.severity === 3 ? "critical" : v.severity === 2 ? "serious" : "stable";
      vToken.className = `victim-token ${sevClass}`;
      vToken.id = `victim-token-${v.id}`;
      vToken.innerHTML = `
        <span>V${v.id}</span>
        <span class="victim-urgency-pill">${v.urgency.toFixed(1)}</span>
      `;
      vToken.title = `Victim V${v.id}: ${v.severity_name} (Urgency: ${v.urgency.toFixed(1)}, Wait: ${v.wait_time}/${v.max_wait})`;
      cell.appendChild(vToken);
    });
  }

  // Render Agents
  if (agents && Array.isArray(agents)) {
    agents.forEach((a) => {
      const [ax, ay] = a.pos;
      const cell = document.getElementById(`cell-${ax}-${ay}`);
      if (!cell) return;

      const aToken = document.createElement("div");
      aToken.className = `agent-token agent-${a.id}`;
      aToken.id = `agent-token-${a.id}`;
      aToken.textContent = `A${a.id}`;

      if (a.carrying !== null && a.carrying !== undefined) {
        const carryBadge = document.createElement("span");
        carryBadge.className = "agent-carrying-badge";
        carryBadge.textContent = `+V${a.carrying}`;
        aToken.appendChild(carryBadge);
      }

      aToken.title = `Agent ${a.id}: ${a.state.toUpperCase()} | Rescues: ${a.total_rescues} | Moves: ${a.total_moves}`;
      cell.appendChild(aToken);
    });
  }
}

// 5. Delta Event Handling & Incident Reporting
function processDeltaEvents(events) {
  if (!events) return;

  if (events.newly_rescued && events.newly_rescued.length > 0) {
    events.newly_rescued.forEach((vid) => {
      appendLog("RESCUE", `Victim V${vid} successfully delivered to shelter!`, "rescue");
    });
    elements.kpiTrendRescued.textContent = `+${events.newly_rescued.length} new`;
  }

  if (events.newly_died && events.newly_died.length > 0) {
    events.newly_died.forEach((vid) => {
      appendLog("CASUALTY", `Victim V${vid} expired before rescue extraction.`, "casualty");
    });
    elements.kpiTrendCasualties.textContent = `+${events.newly_died.length} expired`;
  }

  if (events.assignments) {
    const summaryList = Object.entries(events.assignments)
      .map(([aid, task]) => `A${aid}→${task || "wait"}`)
      .join(", ");
    if (summaryList) {
      const tag = state.mode === "llm" ? "LLM" : "COORD";
      const cat = state.mode === "llm" ? "llm" : "coordinator";
      appendLog(tag, `Fleet assignments: ${summaryList}`, cat);
    }
  }
}

// 6. Fleet Telemetry Panel
function renderAgentRoster(agentStats) {
  elements.agentRoster.innerHTML = "";
  agentStats.forEach((a) => {
    const card = document.createElement("div");
    card.className = "agent-card";

    const isCarrying = a.carrying !== null && a.carrying !== undefined;
    const statusText = isCarrying ? `Carrying V${a.carrying}` : a.state;
    const statusClass = isCarrying ? "carrying" : "idle";

    card.innerHTML = `
      <div class="agent-card-header">
        <span class="agent-id-badge agent-${a.id}">● Agent ${a.id}</span>
        <span class="agent-status-pill ${statusClass}">${statusText}</span>
      </div>
      <div class="agent-card-stats">
        <span>Position: (${a.pos[0]}, ${a.pos[1]})</span>
        <span>Rescued: <strong>${a.total_rescues}</strong></span>
        <span>Moves: <strong>${a.total_moves}</strong></span>
      </div>
      <div>
        <span class="agent-task-text">${a.current_task ? `Goal: ${a.current_task}` : "Goal: Idle / Seeking"}</span>
      </div>
    `;

    elements.agentRoster.appendChild(card);
  });
}

// 7. Simulation Playback Controls
function startSimulation() {
  if (state.isRunning) return;
  state.isRunning = true;
  elements.btnPlayIcon.textContent = "⏸";
  elements.btnPlayLabel.textContent = "Pause Run";
  elements.btnPlayPause.classList.add("paused");
  elements.statusDot.className = "status-dot green";
  elements.statusText.textContent = "SIMULATION RUNNING";

  state.timerId = setInterval(() => {
    apiStep();
  }, state.stepDelay);
}

function pauseSimulation() {
  if (!state.isRunning) return;
  state.isRunning = false;
  if (state.timerId) {
    clearInterval(state.timerId);
    state.timerId = null;
  }
  elements.btnPlayIcon.textContent = "▶";
  elements.btnPlayLabel.textContent = "Resume Run";
  elements.btnPlayPause.classList.remove("paused");
  elements.statusDot.className = "status-dot";
  elements.statusText.textContent = "SIMULATION PAUSED";
}

function togglePlayPause() {
  if (state.isRunning) {
    pauseSimulation();
  } else {
    startSimulation();
  }
}

// 8. Event Listeners Setup
function setupEventListeners() {
  // Mode selection
  Object.keys(elements.modeButtons).forEach((modeKey) => {
    const btn = elements.modeButtons[modeKey];
    btn.addEventListener("click", () => {
      if (state.mode === modeKey) return;
      state.mode = modeKey;

      Object.values(elements.modeButtons).forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      elements.activeModeTag.textContent = modeKey.toUpperCase();
      pauseSimulation();
      apiReset(state.mode, state.seed);
    });
  });

  // Action buttons
  elements.btnPlayPause.addEventListener("click", togglePlayPause);
  elements.btnStep.addEventListener("click", () => {
    pauseSimulation();
    apiStep();
  });
  elements.btnReset.addEventListener("click", () => {
    pauseSimulation();
    apiReset(state.mode, state.seed);
  });

  // Speed slider
  elements.speedSlider.addEventListener("input", (e) => {
    state.stepDelay = parseInt(e.target.value, 10);
    elements.speedLabel.textContent = `${state.stepDelay}ms`;
    if (state.isRunning) {
      clearInterval(state.timerId);
      state.timerId = setInterval(() => {
        apiStep();
      }, state.stepDelay);
    }
  });

  // Seed Input & Randomizer
  elements.seedInput.addEventListener("change", (e) => {
    state.seed = parseInt(e.target.value, 10) || 42;
    pauseSimulation();
    apiReset(state.mode, state.seed);
  });

  elements.btnSeedRandom.addEventListener("click", () => {
    const randomSeed = Math.floor(Math.random() * 90000) + 1000;
    elements.seedInput.value = randomSeed;
    state.seed = randomSeed;
    pauseSimulation();
    apiReset(state.mode, state.seed);
  });

  // Console Clear
  elements.btnClearLog.addEventListener("click", () => {
    elements.consoleStream.innerHTML = "";
    appendLog("SYSTEM", "Incident stream log cleared.", "system");
  });
}

// 9. Startup Lifecycle
document.addEventListener("DOMContentLoaded", () => {
  initializeGrid(10);
  setupEventListeners();
  // Fetch initial world state
  apiReset(state.mode, state.seed);
});
