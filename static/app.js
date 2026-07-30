const $ = (selector) => document.querySelector(selector);

const els = {
  prompt: $("#promptInput"), provider: $("#providerSelect"), providerAvailability: $("#providerAvailability"),
  modelCard: $("#modelCard"), start: $("#startButton"), evaporate: $("#evaporateButton"),
  expert: $("#expertToggle"), socket: $("#socketState"), session: $("#sessionId"), run: $("#runId"),
  truthBanner: $("#truthBanner"), maxTokens: $("#maxTokens"), topAlternatives: $("#topAlternatives"),
  temperature: $("#temperature"), topK: $("#topK"), topP: $("#topP"), seed: $("#seed"),
  doSample: $("#doSample"), tokenBody: $("#tokenLaneBody"), tokenCount: $("#tokenCount"),
  streamState: $("#streamState"), transcript: $("#transcript"), layerChart: $("#layerChart"),
  mechanicsToken: $("#mechanicsToken"), causalPath: $("#causalPath"), resources: $("#resourceSamples"),
  capabilityMatrix: $("#capabilityMatrix"), ledger: $("#interventionLedger"), rewriteCount: $("#rewriteCount"),
  selectedMetric: $("#selectedMetric"), alternatives: $("#alternatives"), events: $("#eventLog"),
  exportLink: $("#exportLink"), modelRevision: $("#modelRevision"), tokenizerRevision: $("#tokenizerRevision"),
  promptTokens: $("#promptTokens"), ttft: $("#ttft"), throughput: $("#throughput"),
  processRam: $("#processRam"), cpuUse: $("#cpuUse"), fingerprint: $("#fingerprint"),
};

const state = {
  sessionId: crypto.randomUUID(), socket: null, providers: [], provider: null,
  capabilityRecord: null, tokens: [], selectedTokenId: null, selectedMetricName: "token_logprob",
  rewrites: [], events: [], runMetrics: {}, streaming: false,
};

const metricColumns = [
  ["token_logprob", "logprob"],
  ["token_probability", "probability"],
  ["token_entropy_bits", "entropy"],
  ["top1_top2_probability_margin", "margin"],
  ["token_latency_ms", "latency"],
];

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]);
}

function short(value, length = 9) {
  if (!value) return "—";
  const text = String(value);
  return text.length > length ? `${text.slice(0, length)}…` : text;
}

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  const number = Number(value);
  if (number !== 0 && Math.abs(number) < 0.0001) return number.toExponential(2);
  if (Math.abs(number) >= 1000) return number.toLocaleString(undefined, { maximumFractionDigits: 1 });
  return String(Number(number.toFixed(digits)));
}

function formatMetric(metric) {
  if (!metric || metric.status === "unavailable" || metric.value === null) return "Unavailable";
  const value = formatNumber(metric.value, metric.name?.includes("probability") ? 4 : 3);
  if (metric.unit === "ms") return `${value} ms`;
  if (metric.unit === "bits") return `${value} bits`;
  if (metric.unit === "bytes") return formatBytes(metric.value);
  if (metric.unit === "percent") return `${value}%`;
  return value;
}

function formatBytes(value) {
  if (value === null || value === undefined) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let amount = Number(value); let index = 0;
  while (amount >= 1024 && index < units.length - 1) { amount /= 1024; index += 1; }
  return `${amount.toFixed(index > 1 ? 2 : 0)} ${units[index]}`;
}

function addEvent(label, event = null) {
  state.events.unshift({ label, event, at: new Date() });
  state.events = state.events.slice(0, 24);
  renderEvents();
}

async function loadProviders() {
  try {
    const response = await fetch("/api/providers");
    if (!response.ok) throw new Error(`provider registry returned ${response.status}`);
    const payload = await response.json();
    state.providers = payload.available;
    els.provider.innerHTML = "";
    for (const provider of state.providers) {
      const option = document.createElement("option");
      option.value = provider.provider_id;
      option.textContent = `${provider.name}${provider.available ? "" : " · unavailable"}`;
      option.disabled = !provider.available;
      els.provider.append(option);
    }
    const reference = state.providers.find((provider) => provider.provider_id === "transformers" && provider.available);
    const active = state.providers.find((provider) => provider.provider_id === payload.active.provider_id && provider.available);
    els.provider.value = (reference || active || state.providers.find((provider) => provider.available))?.provider_id || "demo";
    selectProvider();
  } catch (error) {
    els.providerAvailability.textContent = `Provider registry unavailable: ${error.message}`;
    els.providerAvailability.className = "availability bad";
  }
}

function selectProvider() {
  state.provider = state.providers.find((provider) => provider.provider_id === els.provider.value) || null;
  const provider = state.provider;
  if (!provider) return;
  els.start.disabled = !provider.available || state.socket?.readyState !== WebSocket.OPEN;
  els.providerAvailability.textContent = provider.available ? "Ready for a local run" : provider.unavailable_reason || "Unavailable";
  els.providerAvailability.className = `availability ${provider.available ? "good" : "bad"}`;
  const model = provider.model;
  els.modelCard.innerHTML = model
    ? `<strong>${escapeHtml(model.model_id)}</strong><span>${formatNumber(model.parameter_count / 1e6, 0)}M parameters · ${escapeHtml(model.license)}</span><span>CPU-first · ${escapeHtml(model.dtype || model.torch_dtype)} · no quantization</span>`
    : `<strong>${escapeHtml(provider.name)}</strong><span>${provider.provider_id === "demo" ? "Deterministic synthetic fixture" : "Remote provider adapter"}</span>`;
  if (provider.generation_defaults) applyDefaults(provider.generation_defaults);
  renderTruthBanner(provider.provider_id === "demo" ? "synthetic" : "real");
  renderCapabilityMatrix(provider.capability_record);
}

function applyDefaults(settings) {
  els.maxTokens.value = Math.min(settings.max_new_tokens || 16, 32);
  els.topAlternatives.value = settings.top_alternatives || 5;
  els.temperature.value = settings.temperature || 1;
  els.topK.value = settings.top_k ?? 0;
  els.topP.value = settings.top_p ?? 1;
  els.seed.value = settings.seed ?? 1337;
  els.doSample.checked = Boolean(settings.do_sample);
}

function connect() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  state.socket = new WebSocket(`${protocol}://${location.host}/ws/cogito/${state.sessionId}`);
  els.session.textContent = state.sessionId.slice(0, 8);
  state.socket.addEventListener("open", () => {
    els.socket.textContent = "socket open";
    els.socket.classList.add("online");
    els.start.disabled = !state.provider?.available;
    addEvent("WebSocket opened");
  });
  state.socket.addEventListener("close", () => {
    els.socket.textContent = "socket closed";
    els.socket.classList.remove("online");
    els.start.disabled = true;
    addEvent("WebSocket closed");
  });
  state.socket.addEventListener("message", (event) => receive(JSON.parse(event.data)));
}

function send(payload) {
  if (state.socket?.readyState !== WebSocket.OPEN) return;
  state.socket.send(JSON.stringify(payload));
}

function settings() {
  return {
    max_new_tokens: Number(els.maxTokens.value), top_alternatives: Number(els.topAlternatives.value),
    temperature: Number(els.temperature.value), top_k: Number(els.topK.value), top_p: Number(els.topP.value),
    seed: Number(els.seed.value), do_sample: els.doSample.checked,
  };
}

function startExperiment() {
  state.tokens = []; state.rewrites = []; state.events = []; state.selectedTokenId = null;
  state.runMetrics = {}; state.capabilityRecord = null; state.streaming = true;
  els.streamState.textContent = "starting"; els.run.textContent = "pending";
  renderAll();
  send({ type: "start", prompt: els.prompt.value.trim(), provider_id: els.provider.value, generation_settings: settings() });
  addEvent(`Experiment requested with ${state.provider?.name || els.provider.value}`);
}

function receive(message) {
  switch (message.type) {
    case "session_open": addEvent(`Session ${message.session_id.slice(0, 8)} opened`); break;
    case "provider_capabilities":
      state.capabilityRecord = message.capability_record;
      state.provider = message.provider;
      renderCapabilityMatrix(message.capability_record);
      renderTruthBanner(message.provider.provider_id === "demo" ? "synthetic" : "real");
      appendTelemetry(message.telemetry_events);
      break;
    case "router_started": els.streamState.textContent = "encoding"; addEvent("Prompt routing started"); break;
    case "router_complete": addEvent("Prompt routing recorded"); break;
    case "mesh_manifested": addEvent("Temporary session context opened"); break;
    case "token": upsertToken(message.packet); appendTelemetry(message.packet.telemetry_events); break;
    case "history_rewritten": applyRewrite(message); appendTelemetry(message.telemetry_events); break;
    case "stream_cancelled": addEvent("Previous continuation cancelled for steering"); break;
    case "stream_complete": completeRun(message); appendTelemetry(message.telemetry_events); break;
    case "provider_unavailable":
      state.streaming = false; els.streamState.textContent = "unavailable"; addEvent(message.message || "Provider unavailable"); break;
    case "mesh_evaporated": addEvent("Temporary session context released"); break;
    case "error": state.streaming = false; els.streamState.textContent = "error"; addEvent(`Error: ${message.message}`); break;
    default: break;
  }
  renderAll();
}

function appendTelemetry(events = []) {
  for (const event of events || []) addEvent(`${event.event_type} · ${String(event.status).toUpperCase()}`, event);
}

function upsertToken(packet) {
  const index = state.tokens.findIndex((token) => token.token_id === packet.token_id);
  if (index >= 0) state.tokens[index] = packet; else state.tokens.push(packet);
  state.tokens.sort((a, b) => a.token_id - b.token_id);
  state.selectedTokenId = packet.token_id;
  state.selectedMetricName = "token_logprob";
  state.streaming = true; els.streamState.textContent = "streaming";
  els.run.textContent = short(packet.run_id, 8);
  updateRunIdentity(packet);
}

function applyRewrite(message) {
  state.rewrites.push(message.rewrite_event);
  const replacementIndex = message.token_id;
  state.tokens = state.tokens.filter((token) => token.token_id <= replacementIndex);
  const token = state.tokens.find((item) => item.token_id === replacementIndex);
  if (token) { token.token = message.replacement; token.rewritten = true; token.run_id = message.run_id; }
  state.selectedTokenId = replacementIndex;
  els.run.textContent = short(message.run_id, 8);
  addEvent(`${String(message.rewrite_event?.intervention_type || "intervention").replaceAll("_", " ")} at token ${replacementIndex}`);
}

function completeRun(message) {
  state.streaming = false; els.streamState.textContent = "complete";
  state.runMetrics = Object.fromEntries((message.metrics || []).map((metric) => [metric.name, metric]));
  els.throughput.textContent = state.runMetrics.tokens_per_second ? `${formatNumber(state.runMetrics.tokens_per_second.value, 2)} tok/s` : "—";
  addEvent("Run completed");
}

function updateRunIdentity(packet) {
  const raw = packet.raw_provider_payload || {};
  const identity = raw.model_identity || {};
  els.modelRevision.textContent = short(identity.model_revision, 10);
  els.modelRevision.title = identity.model_revision || "";
  els.tokenizerRevision.textContent = short(identity.tokenizer_revision, 10);
  els.tokenizerRevision.title = identity.tokenizer_revision || "";
  els.promptTokens.textContent = raw.prompt_tokens === undefined ? "— tokens" : `${raw.prompt_tokens} tokens`;
  els.fingerprint.textContent = short(raw.run_fingerprint, 10);
  els.fingerprint.title = raw.run_fingerprint || "";
  const metrics = packet.metrics || {};
  if (metrics.time_to_first_token_ms) els.ttft.textContent = formatMetric(metrics.time_to_first_token_ms);
  if (metrics.process_rss_bytes) els.processRam.textContent = formatBytes(metrics.process_rss_bytes.value);
  if (metrics.process_cpu_percent) els.cpuUse.textContent = `${formatNumber(metrics.process_cpu_percent.value, 1)}%`;
}

function renderTruthBanner(status) {
  const isReal = status === "real";
  els.truthBanner.className = `truth-banner ${isReal ? "real" : "synthetic"}`;
  els.truthBanner.innerHTML = isReal
    ? "<strong>MEASURED LOCALLY</strong><span>Values come from the selected model runtime or a named formula. Unsupported signals stay unavailable.</span>"
    : "<strong>DEMO DATA</strong><span>The demo is synthetic and visibly labeled. Choose Transformers for real model measurements.</span>";
}

function renderAll() {
  renderTokens(); renderTranscript(); renderSelected(); renderMechanics(); renderCausalPath(); renderLedger();
  els.tokenCount.textContent = `${state.tokens.length} token${state.tokens.length === 1 ? "" : "s"}`;
  els.rewriteCount.textContent = `${state.rewrites.length} change${state.rewrites.length === 1 ? "" : "s"}`;
  els.exportLink.href = `/api/sessions/${state.sessionId}/export.jsonl`;
  els.exportLink.classList.toggle("disabled", state.tokens.length === 0);
  els.exportLink.setAttribute("aria-disabled", String(state.tokens.length === 0));
}

function renderTokens() {
  if (!state.tokens.length) {
    els.tokenBody.innerHTML = '<tr class="empty-row"><td colspan="8">Run an experiment to populate measured token decisions.</td></tr>';
    return;
  }
  els.tokenBody.innerHTML = state.tokens.map((packet) => {
    const metrics = packet.metrics || {};
    const status = packet.token_event?.logprob?.status || metrics.token_logprob?.status || "unavailable";
    const cells = metricColumns.map(([name]) => metricCell(packet, metrics[name], name)).join("");
    return `<tr class="${packet.token_id === state.selectedTokenId ? "selected" : ""}" data-token-row="${packet.token_id}">
      <td>${packet.token_id}</td><td><button class="token-button" data-select-token="${packet.token_id}">${escapeHtml(visibleToken(packet.token))}</button></td>
      <td><span class="status ${escapeHtml(status)}">${escapeHtml(status.toUpperCase())}</span></td>${cells}</tr>`;
  }).join("");
}

function metricCell(packet, metric, name) {
  const status = metric?.status || "unavailable";
  return `<td><button class="metric-button ${escapeHtml(status)}" data-token="${packet.token_id}" data-metric="${escapeHtml(name)}"><span>${escapeHtml(formatMetric(metric))}</span><small>${escapeHtml(status.toUpperCase())}</small></button></td>`;
}

function visibleToken(token) {
  if (token === " ") return "␠";
  if (token === "\n") return "↵";
  return String(token ?? "").replaceAll("\n", "↵").replaceAll(" ", "·");
}

function renderTranscript() {
  if (!state.tokens.length) { els.transcript.textContent = "Awaiting a measured continuation."; return; }
  els.transcript.textContent = state.tokens.map((packet) => packet.token).join("");
}

function selectedPacket() {
  return state.tokens.find((token) => token.token_id === state.selectedTokenId) || null;
}

function renderSelected() {
  const packet = selectedPacket();
  if (!packet) {
    els.selectedMetric.className = "selected-metric empty-state";
    els.selectedMetric.textContent = "Select any displayed number to ask where it came from.";
    els.alternatives.className = "alternatives empty-state";
    els.alternatives.textContent = "No token selected.";
    return;
  }
  const metric = (packet.metrics || {})[state.selectedMetricName] || packet.token_event?.logprob;
  renderMetricEvidence(metric, packet);
  const alternatives = packet.alternatives || [];
  els.alternatives.className = "alternatives";
  els.alternatives.innerHTML = alternatives.length ? alternatives.map((alternative, index) => {
    const probability = alternative.metrics?.probability || null;
    return `<button class="alternative" data-alternative="${index}"><span><b>${escapeHtml(visibleToken(alternative.token))}</b><small>rank ${alternative.rank || index + 1}</small></span><span>${escapeHtml(formatMetric(probability))}<small>${escapeHtml(String(alternative.status || "unavailable").toUpperCase())}</small></span></button>`;
  }).join("") : '<div class="empty-state">This backend returned no alternatives.</div>';
}

function renderMetricEvidence(metric, packet) {
  if (!metric) {
    els.selectedMetric.className = "selected-metric unavailable";
    els.selectedMetric.innerHTML = '<span class="status unavailable">UNAVAILABLE</span><h3>No measurement emitted</h3><p>The backend did not provide this field; Cogito did not estimate it.</p>';
    return;
  }
  const evidence = (packet.evidence || []).find((record) => record.evidence_id === metric.evidence_id) || {};
  els.selectedMetric.className = `selected-metric ${metric.status}`;
  els.selectedMetric.innerHTML = `
    <div class="metric-heading"><span class="status ${escapeHtml(metric.status)}">${escapeHtml(metric.status.toUpperCase())}</span><code>token ${packet.token_id}</code></div>
    <h3>${escapeHtml(metric.display_label || metric.name)}</h3><div class="metric-value">${escapeHtml(formatMetric(metric))}</div>
    <dl><dt>Source</dt><dd>${escapeHtml(metric.source_name || evidence.source_name || "Not supplied")}</dd>
    <dt>Field</dt><dd>${escapeHtml(evidence.source_field || "Not supplied")}</dd>
    <dt>Capability</dt><dd>${escapeHtml(metric.capability || evidence.capability || "Not declared")}</dd>
    <dt>Formula</dt><dd>${escapeHtml(metric.formula || evidence.formula || "Direct measurement")}</dd>
    <dt>Collected</dt><dd>${escapeHtml(new Date((metric.collected_at || evidence.created_at || Date.now() / 1000) * 1000).toLocaleTimeString())}</dd></dl>
    <p>${escapeHtml(metric.caveat || evidence.caveat || metric.plain_explanation || "No additional caveat recorded.")}</p>`;
}

function renderMechanics() {
  const packet = selectedPacket(); const raw = packet?.raw_provider_payload || {};
  const layers = raw.layer_summaries || [];
  els.mechanicsToken.textContent = packet ? `token ${packet.token_id} · ${visibleToken(packet.token)}` : "select a token";
  if (!layers.length) {
    els.layerChart.className = "layer-chart empty-state";
    els.layerChart.textContent = "Layer summaries are unavailable for this token; no hidden-state value has been inferred.";
    return;
  }
  const maxNorm = Math.max(...layers.map((layer) => layer.l2_norm || 0), 1);
  els.layerChart.className = "layer-chart";
  els.layerChart.innerHTML = `<div class="layer-key"><span>L2 norm</span><span>${layers.length} retained layer summaries · DERIVED from real hidden tensors</span></div><div class="layer-bars">${layers.map((layer) => `<button title="Layer ${layer.layer_index}: norm ${formatNumber(layer.l2_norm)}, delta ${formatNumber(layer.delta_l2)}"><i style="height:${Math.max(5, (layer.l2_norm / maxNorm) * 100)}%"></i><span>${layer.layer_index}</span></button>`).join("")}</div>`;
}

function renderCausalPath() {
  if (!state.tokens.length) {
    els.causalPath.className = "causal-path empty-state"; els.causalPath.textContent = "Prompt → prefill → token decisions"; els.resources.innerHTML = ""; return;
  }
  const packet = selectedPacket() || state.tokens.at(-1);
  els.causalPath.className = "causal-path";
  els.causalPath.innerHTML = `<span>Prompt</span><i>→</i><span>Prefill</span><i>→</i>${state.tokens.slice(-5).map((item) => `<button data-select-token="${item.token_id}" class="${item.token_id === packet.token_id ? "active" : ""}">${escapeHtml(visibleToken(item.token))}</button>`).join('<i>→</i>')}`;
  const resource = packet.raw_provider_payload?.resource_sample || {};
  els.resources.innerHTML = `<div><span>Process RSS</span><strong>${formatBytes(resource.process_rss_bytes)}</strong></div><div><span>System memory</span><strong>${resource.system_memory_percent == null ? "Unavailable" : `${formatNumber(resource.system_memory_percent, 1)}%`}</strong></div><div><span>Process CPU</span><strong>${resource.process_cpu_percent == null ? "Unavailable" : `${formatNumber(resource.process_cpu_percent, 1)}%`}</strong></div>`;
}

function renderCapabilityMatrix(record) {
  const capabilities = record?.capabilities || {};
  const preferred = ["selected_token_logprobs", "top_k_alternatives", "raw_logits", "hidden_states", "attentions", "kv_cache_metrics", "system_resource_telemetry", "branch_replay"];
  els.capabilityMatrix.innerHTML = preferred.map((name) => {
    const capability = capabilities[name];
    if (!capability) return "";
    const visualStatus = capability.support === "supported" ? capability.status_when_present || "real" : "unavailable";
    return `<button data-capability="${escapeHtml(name)}" title="${escapeHtml(capability.limitation || capability.granularity)}"><span>${escapeHtml(name.replaceAll("_", " "))}</span><strong class="${escapeHtml(visualStatus)}">${escapeHtml(capability.support.toUpperCase())}</strong></button>`;
  }).join("");
}

function renderCapabilityEvidence(name) {
  const capability = state.capabilityRecord?.capabilities?.[name] || state.provider?.capability_record?.capabilities?.[name];
  if (!capability) return;
  const status = capability.support === "supported" ? capability.status_when_present || "real" : "unavailable";
  els.selectedMetric.className = `selected-metric ${status}`;
  els.selectedMetric.innerHTML = `<div class="metric-heading"><span class="status ${escapeHtml(status)}">${escapeHtml(status.toUpperCase())}</span><code>capability</code></div>
    <h3>${escapeHtml(name.replaceAll("_", " "))}</h3><div class="metric-value capability-value">${escapeHtml(capability.support.toUpperCase())}</div>
    <dl><dt>Backend</dt><dd>${escapeHtml(state.capabilityRecord?.provider_name || state.provider?.name || "Not selected")}</dd>
    <dt>Granularity</dt><dd>${escapeHtml(capability.granularity)}</dd>
    <dt>Status</dt><dd>${escapeHtml(status.toUpperCase())}</dd></dl>
    <p>${escapeHtml(capability.limitation || "The backend exposes this capability at the stated granularity.")}</p>`;
}

function renderLedger() {
  if (!state.rewrites.length) {
    els.ledger.className = "ledger empty-state";
    els.ledger.textContent = "Selecting an alternate token records the old run, new run, invalidated suffix, and replay method.";
    return;
  }
  els.ledger.className = "ledger";
  els.ledger.innerHTML = state.rewrites.map((entry) => `<article><div><span class="status ${escapeHtml(entry.status)}">${escapeHtml(String(entry.status).toUpperCase())}</span><strong>${escapeHtml(entry.intervention_type.replaceAll("_", " "))}</strong></div><p>${short(entry.old_run_id, 8)} → ${short(entry.new_run_id, 8)} · token ${entry.token_id}</p><small>${entry.reused_prefix_tokens} prefix tokens reused · ${entry.invalidated_token_count} invalidated · recompute from ${entry.recomputed_from_token}</small></article>`).join("");
}

function renderEvents() {
  els.events.innerHTML = "";
  for (const item of state.events.slice(0, 12)) {
    const li = document.createElement("li");
    const time = document.createElement("time"); time.textContent = item.at.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    const span = document.createElement("span"); span.textContent = item.label;
    li.append(time, span); els.events.append(li);
  }
}

document.addEventListener("click", (event) => {
  const capability = event.target.closest("[data-capability]");
  if (capability) { renderCapabilityEvidence(capability.dataset.capability); return; }
  const metric = event.target.closest("[data-metric]");
  if (metric) { state.selectedTokenId = Number(metric.dataset.token); state.selectedMetricName = metric.dataset.metric; renderAll(); return; }
  const token = event.target.closest("[data-select-token]");
  if (token) { state.selectedTokenId = Number(token.dataset.selectToken); state.selectedMetricName = "token_logprob"; renderAll(); return; }
  const alternativeButton = event.target.closest("[data-alternative]");
  if (alternativeButton) {
    const packet = selectedPacket(); const alternative = packet?.alternatives?.[Number(alternativeButton.dataset.alternative)];
    if (packet && alternative) { send({ type: "steer", token_id: packet.token_id, alternative }); addEvent(`Steering requested: ${visibleToken(alternative.token)}`); }
  }
});

els.start.addEventListener("click", startExperiment);
els.evaporate.addEventListener("click", () => { send({ type: "evaporate" }); state.tokens = []; state.rewrites = []; state.events = []; renderAll(); });
els.provider.addEventListener("change", selectProvider);
els.expert.addEventListener("change", () => document.body.classList.toggle("expert", els.expert.checked));
els.exportLink.addEventListener("click", (event) => { if (els.exportLink.classList.contains("disabled")) event.preventDefault(); });

await loadProviders();
connect();
renderAll();
