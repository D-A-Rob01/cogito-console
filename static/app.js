const els = {
  prompt: document.querySelector("#promptInput"),
  start: document.querySelector("#startButton"),
  evaporate: document.querySelector("#evaporateButton"),
  sessionId: document.querySelector("#sessionId"),
  socketState: document.querySelector("#socketState"),
  meshState: document.querySelector("#meshState"),
  providerState: document.querySelector("#providerState"),
  expertToggle: document.querySelector("#expertToggle"),
  tokenCount: document.querySelector("#tokenCount"),
  rewriteCount: document.querySelector("#rewriteCount"),
  graphCanvas: document.querySelector("#graphCanvas"),
  tokenStream: document.querySelector("#tokenStream"),
  eventLog: document.querySelector("#eventLog"),
  selectedToken: document.querySelector("#selectedToken"),
  alternatives: document.querySelector("#alternatives"),
  hiddenState: document.querySelector("#hiddenState"),
  comparePanel: document.querySelector("#comparePanel"),
};

const state = {
  socket: null,
  sessionId: crypto.randomUUID(),
  mesh: null,
  subContext: null,
  tokens: [],
  selectedTokenId: null,
  visualPaused: false,
  pausedGraphTokens: null,
  hoverGhost: null,
  pendingSteer: null,
  collapsedBranches: [],
  originTokenId: null,
  camera: { x: 50, y: 50, scale: 1 },
  rewrites: 0,
  events: [],
  provider: null,
  expertMode: false,
  lastComparison: null,
};

function connect() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  state.socket = new WebSocket(`${protocol}://${location.host}/ws/cogito/${state.sessionId}`);
  els.sessionId.textContent = state.sessionId.slice(0, 8);

  state.socket.addEventListener("open", () => {
    els.socketState.textContent = "open";
    addEvent("socket opened");
  });

  state.socket.addEventListener("close", () => {
    els.socketState.textContent = "closed";
    addEvent("socket closed");
  });

  state.socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    receive(message);
  });
}

function send(message) {
  if (!state.socket || state.socket.readyState !== WebSocket.OPEN) {
    addEvent("socket is not open");
    return;
  }
  state.socket.send(JSON.stringify(message));
}

function startStream() {
  state.tokens = [];
  state.mesh = null;
  state.subContext = null;
  state.selectedTokenId = null;
  state.visualPaused = false;
  state.pausedGraphTokens = null;
  state.hoverGhost = null;
  state.pendingSteer = null;
  state.collapsedBranches = [];
  state.originTokenId = null;
  state.lastComparison = null;
  state.camera = { x: 50, y: 50, scale: 1 };
  els.meshState.textContent = "routing";
  addEvent("prompt checks started");
  renderAll();
  send({ type: "start", prompt: els.prompt.value });
}

function receive(message) {
  switch (message.type) {
    case "session_open":
      state.provider = message.provider;
      els.providerState.textContent = providerLabel(message.provider);
      addEvent(`session ${message.session_id.slice(0, 8)} opened`);
      break;
    case "router_started":
      els.meshState.textContent = "routing";
      addEvent("prompt checks running");
      break;
    case "router_complete":
      state.subContext = message.sub_context;
      addEvent("prompt checks complete");
      break;
    case "mesh_manifested":
      state.mesh = message.mesh;
      els.meshState.textContent = "active";
      addEvent("session context manifested");
      break;
    case "token":
      upsertToken(message.packet);
      break;
    case "history_rewritten":
      applyHistoryRewrite(message);
      break;
    case "stream_cancelled":
      addEvent("stream cancelled for steering");
      break;
    case "stream_complete":
      addEvent("stream complete");
      break;
    case "provider_unavailable":
      els.providerState.textContent = "unavailable";
      addEvent(message.message || "provider unavailable");
      break;
    case "mesh_evaporated":
      els.meshState.textContent = message.mesh.status;
      addEvent(`mesh ${message.mesh.status}; released ${message.mesh.released_nodes} nodes`);
      break;
    case "error":
      addEvent(`error: ${message.message}`);
      break;
    default:
      addEvent(`unhandled event ${message.type}`);
  }
  renderAll();
}

function upsertToken(packet) {
  const existingIndex = state.tokens.findIndex((token) => token.token_id === packet.token_id);
  if (existingIndex >= 0) {
    state.tokens[existingIndex] = { ...packet, rewritten: state.tokens[existingIndex].rewritten };
  } else {
    state.tokens.push(packet);
  }
  state.tokens.sort((a, b) => a.token_id - b.token_id);
  state.selectedTokenId = packet.token_id;
}

function applyHistoryRewrite(message) {
  state.rewrites += 1;
  if (state.pendingSteer) {
    state.collapsedBranches.push({
      from: state.pendingSteer.tokenId,
      tokens: state.pendingSteer.oldTokens,
      origin: state.pendingSteer.origin,
      createdAt: performance.now(),
    });
    state.collapsedBranches = state.collapsedBranches.slice(-3);
  }

  const nextTokens = message.history.map((token, index) => {
    const prior = state.tokens.find((item) => item.token_id === index);
    return {
      ...(prior || {}),
      token_id: index,
      token,
      alternatives: prior?.alternatives || [],
      hidden_state: prior?.hidden_state || {},
      hidden_state_metrics: prior?.hidden_state_metrics || {},
      metrics: prior?.metrics || {},
      token_event: prior?.token_event || null,
      logprob: index === message.token_id ? message.alternative.logprob : prior?.logprob,
      rewritten: index === message.token_id,
    };
  });
  state.lastComparison = {
    tokenId: message.token_id,
    before: message.rewrite_event?.previous_text || state.pendingSteer?.oldTokens?.map((token) => token.token).join("") || "",
    after: message.rewrite_event?.rewritten_text || message.text,
    changedToken: message.replacement,
    oldRunId: message.rewrite_event?.old_run_id,
    newRunId: message.rewrite_event?.new_run_id || message.run_id,
    rationale: message.rewrite_event?.rationale || message.alternative?.rationale,
    alternative: message.alternative,
  };
  state.tokens = nextTokens;
  state.originTokenId = message.token_id;
  state.selectedTokenId = message.token_id;
  state.visualPaused = false;
  state.pausedGraphTokens = null;
  state.hoverGhost = null;
  state.pendingSteer = null;
  addEvent(`Path changed at token ${message.token_id}: ${JSON.stringify(message.replacement.trim())}`);
}

function handleAlternativeClick(tokenId, alternative) {
  queueSteer(tokenId, alternative, getTokenCoordinates(tokenId));
}

function queueSteer(tokenId, alternative, origin = null) {
  const oldTokens = state.tokens.filter((token) => token.token_id >= tokenId).map((token) => ({ ...token }));
  const focus = origin || getTokenCoordinates(tokenId) || { x: 50, y: 50 };
  state.pendingSteer = { tokenId, alternative, oldTokens, origin: focus };
  state.camera = focusCameraOn(focus.x, focus.y);
  state.visualPaused = false;
  state.pausedGraphTokens = null;
  state.hoverGhost = null;
  send({ type: "steer", token_id: tokenId, alternative });
  addEvent(`queued steering token ${JSON.stringify(alternative.token.trim())}`);
  renderAll();
}

function selectToken(tokenId) {
  state.selectedTokenId = tokenId;
  renderAll();
}

function evaporate() {
  send({ type: "evaporate" });
}

function renderAll() {
  els.tokenCount.textContent = state.tokens.length;
  els.rewriteCount.textContent = state.rewrites;
  if (state.visualPaused && state.hoverGhost) {
    els.graphCanvas.classList.add("paused");
    renderGhostDashboardOnly();
  } else {
    renderGraph();
  }
  renderTokenStream();
  renderInspector();
  renderEvents();
  renderCompare();
}

function renderGraph() {
  const graphTokens = getGraphTokens();
  const visible = graphTokens.slice(-7);
  const coordinates = visible.map((packet, index) => ({
    tokenId: packet.token_id,
    x: dominantPathX(index, visible.length),
    y: 50,
  }));
  const coordinateById = new Map(coordinates.map((coordinate) => [coordinate.tokenId, coordinate]));
  const tokenNodes = visible.map((packet, index) => tokenNode(packet, coordinates[index]));
  const ghostNodes = visible.flatMap((packet, index) =>
    (packet.alternatives || []).slice(0, 5).map((alternative, alternativeIndex) =>
      ghostNode(packet, alternative, alternativeIndex, coordinates[index]),
    ),
  );
  const collapsed = collapsedBranchNodes(coordinateById);
  const highwayLines = coordinates.slice(1).map((coordinate, index) =>
    weightedLine(coordinates[index].x, coordinates[index].y, coordinate.x, coordinate.y, 1, "highway-line"),
  );
  const ghostLines = visible.flatMap((packet, index) =>
    (packet.alternatives || []).slice(0, 5).map((alternative, alternativeIndex) => {
      const tokenCoordinate = coordinates[index];
      const ghostCoordinate = ghostCoordinates(tokenCoordinate, alternativeIndex);
      const weight = logprobWeight(alternative.logprob);
      return weightedLine(tokenCoordinate.x, tokenCoordinate.y, ghostCoordinate.x, ghostCoordinate.y, weight, "ghost-line");
    }),
  );
  const collapsedLines = collapsed.lines;
  const dashboard = state.hoverGhost ? ghostDashboard(state.hoverGhost) : "";
  const cameraStyle = `--camera-shift-x:${((50 - state.camera.x) * 0.3).toFixed(2)}%; --camera-shift-y:${((50 - state.camera.y) * 0.3).toFixed(2)}%; --camera-scale:${state.camera.scale};`;

  els.graphCanvas.classList.toggle("paused", state.visualPaused);

  const nodes = [
    node("Prompt", "ingest", 12, 50, "active system-node"),
    node("Prompt checks", contextVerdict("constraint_bounds"), 17, 28, state.subContext ? "active system-node" : "system-node"),
    node("Live path", "accepted tokens", 17, 72, visible.length ? "active system-node" : "system-node"),
    ...tokenNodes,
    ...ghostNodes,
    ...collapsed.nodes,
  ];

  els.graphCanvas.innerHTML = `
    <div class="graph-camera" style="${cameraStyle}">
      <svg class="graph-lines" viewBox="0 0 100 100" preserveAspectRatio="none">
        ${line(12, 50, 17, 28)}
        ${line(12, 50, 17, 72)}
        ${coordinates.length ? line(17, 72, coordinates[0].x, coordinates[0].y, "amber") : ""}
        ${highwayLines.join("")}
        ${ghostLines.join("")}
        ${collapsedLines.join("")}
      </svg>
      ${nodes.join("")}
    </div>
    ${dashboard}
  `;

  els.graphCanvas.querySelectorAll(".dominant-node[data-token-id]").forEach((item) => {
    item.addEventListener("click", () => selectToken(Number(item.dataset.tokenId)));
  });
  els.graphCanvas.querySelectorAll(".ghost-node").forEach((item) => {
    const tokenId = Number(item.dataset.tokenId);
    const alternativeIndex = Number(item.dataset.alternativeIndex);
    const packet = getGraphTokens().find((token) => token.token_id === tokenId);
    const alternative = packet?.alternatives?.[alternativeIndex];
    if (!packet || !alternative) return;
    const origin = { x: Number(item.dataset.x), y: Number(item.dataset.y) };
    item.addEventListener("mouseenter", () => handleGhostHover(packet, alternative, alternativeIndex, origin));
    item.addEventListener("mouseleave", clearGhostHover);
    item.addEventListener("click", (event) => {
      event.stopPropagation();
      handleGhostClick(packet, alternative, alternativeIndex, origin);
    });
  });
}

function node(title, subtitle, x, y, className = "", tokenId = null) {
  const tokenAttr = tokenId === null ? "" : ` data-token-id="${tokenId}"`;
  return `
    <button class="graph-node ${className}" style="left:${x}%; top:${y}%;"${tokenAttr}>
      <strong>${escapeHtml(title)}</strong>
      <span>${escapeHtml(subtitle)}</span>
    </button>
  `;
}

function tokenNode(packet, coordinate) {
  const logprob = tokenMetric(packet, "logprob");
  const probability = tokenMetric(packet, "probability");
  const weight = logprobWeight(metricValue(logprob, packet.logprob));
  const isPendingDiscard = state.pendingSteer && packet.token_id >= state.pendingSteer.tokenId;
  const className = [
    "dominant-node",
    packet.token_id === state.selectedTokenId ? "selected" : "",
    packet.token_id === state.originTokenId ? "origin" : "",
    packet.rewritten ? "rewritten" : "",
    isPendingDiscard ? "collapsing" : "",
  ].filter(Boolean).join(" ");
  return `
    <button class="graph-node ${className}" data-token-id="${packet.token_id}" style="left:${coordinate.x}%; top:${coordinate.y}%; --node-weight:${weight};">
      <strong>${escapeHtml(cleanToken(packet.token))}</strong>
      <span>${statusBadgeHtml(probability?.status || logprob?.status)} ${escapeHtml(metricSummary(probability, packet.probability, "probability"))}</span>
    </button>
  `;
}

function ghostNode(packet, alternative, alternativeIndex, tokenCoordinate) {
  const coordinate = ghostCoordinates(tokenCoordinate, alternativeIndex);
  const logprob = alternativeMetric(alternative, "logprob");
  const probability = alternativeMetric(alternative, "probability");
  const weight = logprobWeight(metricValue(logprob, alternative.logprob));
  const probabilityValue = metricValue(probability, alternative.probability ?? probabilityFromLogprob(alternative.logprob));
  return `
    <button
      class="ghost-node"
      data-token-id="${packet.token_id}"
      data-alternative-index="${alternativeIndex}"
      data-x="${coordinate.x}"
      data-y="${coordinate.y}"
      style="left:${coordinate.x}%; top:${coordinate.y}%; --ghost-opacity:${ghostOpacity(weight)}; --ghost-size:${ghostSize(weight)}px; --ghost-glow:${ghostGlow(weight)}px;"
      aria-label="Steer token ${packet.token_id} to ${escapeHtml(cleanToken(alternative.token))}"
    >
      <strong>${escapeHtml(cleanToken(alternative.token))}</strong>
      <span>${statusBadgeHtml(alternative.status || probability?.status)} ${formatPercent(probabilityValue)}</span>
    </button>
  `;
}

function ghostDashboard(ghost) {
  const logprob = alternativeMetric(ghost.alternative, "logprob");
  const probability = alternativeMetric(ghost.alternative, "probability");
  const probabilityValue = metricValue(probability, ghost.alternative.probability ?? probabilityFromLogprob(ghost.alternative.logprob));
  const left = Math.min(78, Math.max(12, ghost.origin.x + 4));
  const top = Math.min(78, Math.max(10, ghost.origin.y - 10));
  return `
    <aside class="ghost-dashboard" style="left:${left}%; top:${top}%;">
      <div>Alternative path ${statusBadgeHtml(ghost.alternative.status || probability?.status)}</div>
      <strong>${escapeHtml(JSON.stringify(ghost.alternative.token))}</strong>
      <dl>
        <dt>token_id</dt><dd>${ghost.packet.token_id}</dd>
        <dt>logprob</dt><dd>${escapeHtml(metricSummary(logprob, ghost.alternative.logprob, "logprob"))}</dd>
        <dt>probability</dt><dd>${escapeHtml(metricSummary(probability, probabilityValue, "probability"))}</dd>
        <dt>evidence</dt><dd>${escapeHtml(probability?.evidence_id || logprob?.evidence_id || "none")}</dd>
      </dl>
      <p><b>What it means:</b> ${escapeHtml(probability?.plain_explanation || "An alternate continuation available at this token.")}</p>
      <p>${escapeHtml(ghost.alternative.rationale || "Alternative semantic branch.")}</p>
      ${state.expertMode ? rawBlock("Alternative metric", ghost.alternative.metrics || {}) : ""}
    </aside>
  `;
}

function collapsedBranchNodes(coordinateById) {
  const branches = [];
  if (state.pendingSteer) {
    branches.push({
      from: state.pendingSteer.tokenId,
      tokens: state.pendingSteer.oldTokens,
      origin: state.pendingSteer.origin,
    });
  }
  branches.push(...state.collapsedBranches);

  const nodes = [];
  const lines = [];
  for (const branch of branches.slice(-2)) {
    const anchor = branch.origin || coordinateById.get(branch.from) || { x: 50, y: 50 };
    const tokens = (branch.tokens || []).slice(0, 6);
    tokens.forEach((packet, index) => {
      const x = Math.min(92, anchor.x + index * 3.2);
      const y = Math.min(93, anchor.y + 15 + index * 1.8);
      if (index === 0) {
        lines.push(weightedLine(anchor.x, anchor.y, x, y, 0.32, "discarded-line"));
      } else {
        const priorX = Math.min(92, anchor.x + (index - 1) * 3.2);
        const priorY = Math.min(93, anchor.y + 15 + (index - 1) * 1.8);
        lines.push(weightedLine(priorX, priorY, x, y, 0.22, "discarded-line"));
      }
      nodes.push(`
        <span class="discarded-node" style="left:${x}%; top:${y}%;">
          ${escapeHtml(cleanToken(packet.token))}
        </span>
      `);
    });
  }
  return { nodes, lines };
}

function line(x1, y1, x2, y2, className = "") {
  return `<line class="${className}" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" />`;
}

function weightedLine(x1, y1, x2, y2, weight, className = "") {
  return `<line class="${className}" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" style="--line-weight:${weight};" />`;
}

function renderTokenStream() {
  els.tokenStream.innerHTML = "";
  for (const packet of state.tokens) {
    const button = document.createElement("button");
    button.className = `token-pill${packet.token_id === state.selectedTokenId ? " selected" : ""}${packet.rewritten ? " rewritten" : ""}`;
    button.textContent = packet.token.replace(/\s/g, " ");
    button.addEventListener("click", () => selectToken(packet.token_id));
    els.tokenStream.append(button);
  }
}

function renderInspector() {
  const packet = state.tokens.find((token) => token.token_id === state.selectedTokenId);
  if (!packet) {
    els.selectedToken.className = "selected-token empty";
    els.selectedToken.textContent = "Click a streamed token to inspect its alternatives.";
    els.alternatives.innerHTML = "";
    els.hiddenState.innerHTML = "";
    return;
  }

  const logprob = tokenMetric(packet, "logprob");
  const probability = tokenMetric(packet, "probability");
  const latency = tokenMetric(packet, "latency_ms");
  els.selectedToken.className = "selected-token";
  els.selectedToken.innerHTML = `
    <strong>Token ${packet.token_id}</strong> ${statusBadgeHtml(probability?.status || logprob?.status)}<br />
    <code>${escapeHtml(JSON.stringify(packet.token))}</code>
    <dl class="metric-list">
      <dt>Probability</dt><dd>${escapeHtml(metricSummary(probability, packet.probability, "probability"))}</dd>
      <dt>Logprob</dt><dd>${escapeHtml(metricSummary(logprob, packet.logprob, "logprob"))}</dd>
      <dt>Latency</dt><dd>${escapeHtml(metricSummary(latency, null, "ms"))}</dd>
      <dt>Evidence</dt><dd>${escapeHtml(probability?.evidence_id || logprob?.evidence_id || "none")}</dd>
    </dl>
    <p><b>Why this number?</b> ${escapeHtml(probability?.plain_explanation || logprob?.plain_explanation || "No probability metric was returned for this token.")}</p>
    <p><b>Where did this come from?</b> ${escapeHtml(sourceLine(packet, probability || logprob))}</p>
    ${state.expertMode ? rawBlock("Raw token event", packet.token_event || packet) : ""}
  `;

  els.alternatives.innerHTML = "";
  for (const alternative of packet.alternatives || []) {
    const alternativeLogprob = alternativeMetric(alternative, "logprob");
    const alternativeProbability = alternativeMetric(alternative, "probability");
    const button = document.createElement("button");
    button.className = "alternative-button";
    button.innerHTML = `
      <code>${escapeHtml(JSON.stringify(alternative.token))}</code>
      <span>${statusBadgeHtml(alternative.status || alternativeProbability?.status)} ${escapeHtml(metricSummary(alternativeProbability, alternative.probability, "probability"))}</span>
      <small><b>Use this path</b> ${escapeHtml(alternative.rationale || "Rewrite model history from this point.")}</small>
      <small>Evidence: ${escapeHtml(alternativeProbability?.evidence_id || alternativeLogprob?.evidence_id || "none")}</small>
    `;
    button.addEventListener("click", () => handleAlternativeClick(packet.token_id, alternative));
    els.alternatives.append(button);
  }

  els.hiddenState.innerHTML = "";
  for (const [key, value] of Object.entries(packet.hidden_state || {})) {
    const metric = packet.hidden_state_metrics?.[key] || syntheticLegacySignal(key, value);
    const row = document.createElement("div");
    row.className = "state-row";
    row.innerHTML = `
      <span>${escapeHtml(metric.display_label || key)} ${statusBadgeHtml(metric.status)}</span>
      <span class="bar"><i style="--value:${Math.round(value * 100)}%"></i></span>
      <b title="${escapeHtml(metric.plain_explanation)}">${value}</b>
    `;
    els.hiddenState.append(row);
  }
}

function renderEvents() {
  els.eventLog.innerHTML = state.events
    .slice(-9)
    .map((event) => `<li>${escapeHtml(event)}</li>`)
    .join("");
}

function renderCompare() {
  if (!state.lastComparison) {
    els.comparePanel.className = "compare-panel empty";
    els.comparePanel.textContent = "Change a path to compare branches.";
    return;
  }
  const comparison = state.lastComparison;
  const probability = alternativeMetric(comparison.alternative || {}, "probability");
  const logprob = alternativeMetric(comparison.alternative || {}, "logprob");
  els.comparePanel.className = "compare-panel";
  els.comparePanel.innerHTML = `
    <div class="panel-title">Compare branches</div>
    <p>You changed the path at token ${comparison.tokenId}. The system restarted from that point with the selected token.</p>
    <dl class="metric-list">
      <dt>Changed token</dt><dd><code>${escapeHtml(JSON.stringify(comparison.changedToken))}</code></dd>
      <dt>Selected metric</dt><dd>${escapeHtml(metricSummary(probability || logprob, comparison.alternative?.logprob, "probability/logprob"))}</dd>
      <dt>Old run</dt><dd>${escapeHtml(comparison.oldRunId || "unknown")}</dd>
      <dt>New run</dt><dd>${escapeHtml(comparison.newRunId || "unknown")}</dd>
    </dl>
    <div class="branch-text">
      <section><b>Before</b><p>${escapeHtml(comparison.before || "(empty)")}</p></section>
      <section><b>After</b><p>${escapeHtml(comparison.after || "(empty)")}</p></section>
    </div>
    ${comparison.rationale ? `<p><b>Why it matters:</b> ${escapeHtml(comparison.rationale)}</p>` : ""}
  `;
}

function addEvent(text) {
  const timestamp = new Date().toLocaleTimeString([], { hour12: false });
  state.events.push(`${timestamp} ${text}`);
  renderEvents();
}

function contextVerdict(key) {
  const item = state.subContext?.[key];
  if (!item) return "pending";
  return `${Math.round(item.confidence * 100)}% SYNTHETIC ${item.signals?.[0] || "validated"}`;
}

function getGraphTokens() {
  return state.visualPaused && state.pausedGraphTokens ? state.pausedGraphTokens : state.tokens;
}

function getTokenCoordinates(tokenId) {
  const visible = getGraphTokens().slice(-7);
  const index = visible.findIndex((packet) => packet.token_id === tokenId);
  if (index === -1) return null;
  return {
    x: dominantPathX(index, visible.length),
    y: 50,
  };
}

function dominantPathX(index, length) {
  return length === 1 ? 52 : 24 + index * (54 / Math.max(1, length - 1));
}

function ghostCoordinates(tokenCoordinate, alternativeIndex) {
  const offsets = [
    { x: -3.5, y: -30 },
    { x: 4.5, y: -18 },
    { x: -4.5, y: 18 },
    { x: 4, y: 30 },
    { x: -1.5, y: 40 },
  ];
  const offset = offsets[alternativeIndex] || offsets[offsets.length - 1];
  return {
    x: Math.min(94, Math.max(6, tokenCoordinate.x + offset.x)),
    y: Math.min(94, Math.max(8, tokenCoordinate.y + offset.y)),
  };
}

function handleGhostHover(packet, alternative, alternativeIndex, origin) {
  if (!state.visualPaused) {
    state.pausedGraphTokens = state.tokens.map((token) => ({ ...token }));
  }
  state.visualPaused = true;
  state.hoverGhost = { packet, alternative, alternativeIndex, origin };
  els.graphCanvas.classList.add("paused");
  renderGhostDashboardOnly();
}

function clearGhostHover() {
  state.visualPaused = false;
  state.pausedGraphTokens = null;
  state.hoverGhost = null;
  renderGraph();
}

function renderGhostDashboardOnly() {
  els.graphCanvas.querySelector(".ghost-dashboard")?.remove();
  if (state.hoverGhost) {
    els.graphCanvas.insertAdjacentHTML("beforeend", ghostDashboard(state.hoverGhost));
  }
}

function handleGhostClick(packet, alternative, alternativeIndex, origin) {
  state.originTokenId = packet.token_id;
  queueSteer(packet.token_id, alternative, origin || ghostCoordinates(getTokenCoordinates(packet.token_id), alternativeIndex));
}

function focusCameraOn(x, y) {
  return { x, y, scale: 1.08 };
}

function tokenMetric(packet, name) {
  if (name === "logprob") return packet.token_event?.logprob || packet.metrics?.token_logprob || null;
  if (name === "probability") return packet.token_event?.probability || packet.metrics?.token_probability || null;
  if (name === "latency_ms") return packet.token_event?.latency_ms || packet.metrics?.token_latency_ms || null;
  if (name === "byte_length") return packet.token_event?.byte_length || packet.metrics?.byte_length || null;
  return packet.metrics?.[name] || null;
}

function alternativeMetric(alternative, name) {
  return alternative.metrics?.[name] || alternative.metrics?.[`alternative_${alternative.rank}_${name}`] || null;
}

function metricValue(metric, fallback = null) {
  const value = metric?.value ?? fallback;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : value;
}

function metricSummary(metric, fallback = null, unitHint = "") {
  if (!metric && fallback === null) return "not returned";
  const value = metricValue(metric, fallback);
  if (value === null || value === undefined) return "not returned";
  const unit = metric?.unit || unitHint;
  if (unit === "probability") return `${formatPercent(Number(value))} (${metric?.status || "unknown"})`;
  if (unit === "ms") return `${Number(value).toFixed(1)}ms (${metric?.status || "unknown"})`;
  if (typeof value === "number") return `${Number(value).toFixed(3)} ${unit}`.trim() + ` (${metric?.status || "unknown"})`;
  return `${value} (${metric?.status || "unknown"})`;
}

function statusBadgeHtml(status = "unknown") {
  return `<span class="status-badge ${escapeHtml(status)}">${escapeHtml(String(status).toUpperCase())}</span>`;
}

function sourceLine(packet, metric) {
  if (!metric) return "No provider or runtime source was attached.";
  const evidence = (packet.evidence || []).find((record) => record.evidence_id === metric.evidence_id);
  if (!evidence) return `Evidence record ${metric.evidence_id} is referenced but not in this packet.`;
  const field = evidence.source_field ? ` field ${evidence.source_field}` : "";
  const caveat = evidence.caveat ? ` Caveat: ${evidence.caveat}` : "";
  return `${evidence.source_kind.toUpperCase()} from ${evidence.source_name}${field}.${caveat}`;
}

function providerLabel(provider) {
  if (!provider) return "unknown";
  return provider.available === false ? `${provider.mode}: unavailable` : provider.mode;
}

function rawBlock(label, value) {
  return `
    <details class="raw-details">
      <summary>${escapeHtml(label)}</summary>
      <pre>${escapeHtml(JSON.stringify(value, null, 2))}</pre>
    </details>
  `;
}

function syntheticLegacySignal(key, value) {
  return {
    name: `legacy_${key}`,
    value,
    unit: "score",
    status: "synthetic",
    evidence_id: "legacy-unpersisted",
    display_label: key.replaceAll("_", " "),
    plain_explanation: "Legacy local signal; treated as synthetic because no provider evidence was attached.",
  };
}

function logprobWeight(logprob) {
  const value = Number(logprob);
  if (!Number.isFinite(value)) return 0.45;
  return Math.min(1, Math.max(0.08, (value + 5) / 5));
}

function probabilityFromLogprob(logprob) {
  const value = Number(logprob);
  if (!Number.isFinite(value)) return 0;
  return Math.exp(value);
}

function formatPercent(probability) {
  return `${(probability * 100).toFixed(probability > 0.1 ? 1 : 2)}%`;
}

function ghostOpacity(weight) {
  return (0.16 + weight * 0.84).toFixed(3);
}

function ghostSize(weight) {
  return Math.round(40 + weight * 34);
}

function ghostGlow(weight) {
  return Math.round(4 + weight * 28);
}

function cleanToken(token) {
  const value = (token || "").trim();
  return value || "space";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

els.start.addEventListener("click", startStream);
els.evaporate.addEventListener("click", evaporate);
els.expertToggle.addEventListener("change", () => {
  state.expertMode = els.expertToggle.checked;
  renderAll();
});
connect();
renderAll();

window.handleAlternativeClick = handleAlternativeClick;
