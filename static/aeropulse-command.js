(function () {
  'use strict';

  const byId = (id) => document.getElementById(id);
  const all = (selector) => Array.from(document.querySelectorAll(selector));
  const safeNumber = (value) => {
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  };

  const state = {
    workspace: 'overview',
    view: 'operator',
    streamStartedAt: null,
    lastSampleAt: null,
    latestMissionSeconds: null,
    missionComplete: false
  };

  const phaseOrder = ['PREFLIGHT', 'TAXI', 'TAKEOFF', 'CLIMB', 'CRUISE', 'LOITER', 'RECOVERY'];

  function normalizePhase(value) {
    const phase = String(value || 'PREFLIGHT').toUpperCase().replaceAll(' ', '_');
    if (phase === 'GROUND' || phase === 'READY') return 'PREFLIGHT';
    if (phase === 'ISR' || phase === 'PATROL' || phase === 'HIGH_ALTITUDE') return 'LOITER';
    if (phase === 'RETURN' || phase === 'DESCENT' || phase === 'LANDING' || phase === 'COMPLETE') return 'RECOVERY';
    return phaseOrder.includes(phase) ? phase : 'CRUISE';
  }

  function classifyHealth(value) {
    const n = safeNumber(value);
    if (n == null) return 'warning';
    if (n >= 75) return 'healthy';
    if (n >= 50) return 'warning';
    return 'critical';
  }

  function classifyRisk(level, score) {
    const normalized = String(level || '').toUpperCase();
    if (normalized === 'HIGH' || normalized === 'CRITICAL') return 'critical';
    if (normalized === 'MEDIUM') return 'warning';
    if (normalized === 'LOW') return 'healthy';
    const n = safeNumber(score);
    if (n == null) return 'warning';
    if (n >= 70) return 'critical';
    if (n >= 35) return 'warning';
    return 'healthy';
  }

  function setText(id, value, dataState) {
    const node = byId(id);
    if (!node) return;
    node.textContent = value;
    if (dataState) node.dataset.state = dataState;
  }

  function formatDuration(seconds) {
    const total = Math.max(0, Math.floor(Number(seconds) || 0));
    const hours = String(Math.floor(total / 3600)).padStart(2, '0');
    const minutes = String(Math.floor((total % 3600) / 60)).padStart(2, '0');
    const secs = String(total % 60).padStart(2, '0');
    return `T+${hours}:${minutes}:${secs}`;
  }

  function updatePhase(value) {
    const phase = normalizePhase(value);
    setText('apxMissionPhase', phase);
    const activeIndex = phaseOrder.indexOf(phase);
    all('[data-apx-phase]').forEach((node, index) => {
      node.classList.toggle('active', index === activeIndex && !state.missionComplete);
      node.classList.toggle('complete', state.missionComplete || index < activeIndex);
    });
  }

  function setView(view, persist = true) {
    const next = view === 'engineer' ? 'engineer' : 'operator';
    state.view = next;
    document.body.dataset.apxView = next;
    all('[data-apx-view-option]').forEach((button) => {
      const active = button.dataset.apxViewOption === next;
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', String(active));
    });
    if (persist) {
      try { localStorage.setItem('aeropulse-view', next); } catch (_) { /* storage can be disabled */ }
    }
  }

  function setWorkspace(workspace, persist = true) {
    const allowed = new Set(['overview', 'twin', 'thermal', 'mission', 'diagnostics', 'rul', 'replay', 'maintenance', 'reports']);
    const next = allowed.has(workspace) ? workspace : 'overview';
    state.workspace = next;
    document.body.dataset.apxWorkspace = next;

    all('[data-apx-workspace]').forEach((section) => {
      const targets = String(section.dataset.apxWorkspace || '').split(/\s+/);
      section.classList.toggle('apx-workspace-hidden', !targets.includes(next));
    });
    all('[data-apx-workspace-option]').forEach((button) => {
      const active = button.dataset.apxWorkspaceOption === next;
      button.classList.toggle('active', active);
      if (active) button.setAttribute('aria-current', 'page');
      else button.removeAttribute('aria-current');
    });

    if (next === 'mission' && typeof uavMap !== 'undefined' && uavMap && typeof uavMap.invalidateSize === 'function') {
      requestAnimationFrame(() => uavMap.invalidateSize());
    }
    if (next === 'thermal') {
      const thermalButton = document.querySelector('[data-engine-mode="thermal"]');
      if (thermalButton && !thermalButton.classList.contains('active')) thermalButton.click();
    }
    if ((next === 'maintenance' || next === 'reports') && byId('apxModelStatus')?.dataset.loaded !== 'true') {
      loadModelStatus();
    }
    if (persist) {
      try { localStorage.setItem('aeropulse-workspace', next); } catch (_) { /* storage can be disabled */ }
    }
  }

  function recommendationFor(level, score, advisory) {
    const directive = String(advisory || '').toUpperCase();
    if (directive.includes('[NO-GO FLIGHT HOLD]')) return { text: 'MISSION HOLD / INSPECT', state: 'critical' };
    if (directive.includes('[INSTRUMENTATION DIRECTIVE]')) return { text: 'VERIFY SENSOR TRUST', state: 'warning' };
    if (directive.includes('[CAUTION ADVISORY]')) return { text: 'MONITOR CLOSELY', state: 'warning' };
    if (directive.includes('[DISPATCH CLEARED]')) return { text: 'CONTINUE MISSION', state: 'healthy' };
    const riskState = classifyRisk(level, score);
    if (riskState === 'critical') return { text: 'RETURN TO BASE', state: 'critical' };
    if (riskState === 'warning') return { text: 'MONITOR CLOSELY', state: 'warning' };
    return { text: 'CONTINUE MISSION', state: 'healthy' };
  }

  function setDecision(recommendation, decisionState, title, detail) {
    setText('apxDecisionNow', recommendation);
    setText('apxRecommendation', recommendation, decisionState);
    setText('apxEvidenceTitle', title || 'Model result available');
    setText('apxEvidenceText', detail || 'Review the detailed evidence before making a maintenance decision.');
    const command = byId('apxCommandDecision');
    if (command) command.dataset.state = decisionState;
  }

  function updateEnvironmentSource(environment) {
    if (!environment) return;
    const source = String(environment.source || 'unknown').toLowerCase();
    const label = source === 'live'
      ? 'LIVE · OPEN-METEO'
      : source === 'manual_override'
        ? 'MANUAL SIMULATION OVERRIDE'
        : source === 'simulated'
          ? 'SIMULATED · ISA MODEL'
          : 'SOURCE UNKNOWN';
    ['envBadge', 'envSourceBadge'].forEach((id) => {
      const node = byId(id);
      if (!node) return;
      node.textContent = label;
      node.className = `${id === 'envBadge' ? 'badge' : 'pill'} ${source === 'live' ? 'live good' : 'warn'}`;
    });
  }

  function updateCore(payload) {
    if (!payload) return;
    state.lastSampleAt = performance.now();

    const health = safeNumber(payload.health_index);
    const risk = payload.mission_risk || {};
    const riskLevel = payload.risk_level || risk.level;
    const riskScore = safeNumber(payload.risk_score != null ? payload.risk_score : risk.score);
    const healthState = classifyHealth(health);
    const riskState = classifyRisk(riskLevel, riskScore);
    const rulData = payload.rul || {};
    const rul = safeNumber(rulData.rul_hours != null ? rulData.rul_hours : payload.rul_hours);
    const fault = payload.primary_fault || payload.fault_candidates?.[0]?.name || 'No dominant fault evidence';
    const confidence = safeNumber(payload.health_confidence);
    const anomaly = Boolean(payload.anomaly_flag);
    const recommendation = recommendationFor(riskLevel, riskScore, payload.maintenance_advisory);
    const phase = payload.uav?.mission_phase || payload.mission_phase;

    setText('apxHealthNow', health == null ? '—' : `${health.toFixed(1)} / 100`, healthState);
    setText('apxRiskNow', riskScore == null ? String(riskLevel || '—') : `${String(riskLevel || 'RISK')} · ${riskScore.toFixed(0)}`, riskState);
    setText('apxRulNow', rul == null ? 'NOT REPORTED' : `${rul.toFixed(1)} h`);

    const confidenceText = confidence == null ? 'confidence not reported' : `${(confidence * 100).toFixed(1)}% confidence`;
    const evidence = payload.maintenance_advisory
      || `${fault}; ${confidenceText}; ${anomaly ? 'anomaly evidence detected' : 'no anomaly flag in this sample'}.`;
    setDecision(recommendation.text, recommendation.state, fault, evidence);

    const faultIsActive = !['none', 'normal', 'healthy', 'no dominant fault evidence'].includes(String(fault).toLowerCase());
    const alertCount = Number(anomaly) + Number(riskState === 'critical') + Number(faultIsActive);
    setText('apxAlertCount', String(alertCount), alertCount > 0 ? (riskState === 'critical' ? 'critical' : 'warning') : 'healthy');

    if (phase) updatePhase(phase);
    if (payload.environment) updateEnvironmentSource(payload.environment);
    if (payload.time_min != null) state.latestMissionSeconds = Number(payload.time_min) * 60;
  }

  function updateFromAnalysis(result) {
    updateCore(result);
    setText('apxAircraftId', 'APX-MALE-01 · SIMULATION');
    setConnection('BACKEND READY', 'online');
  }

  function updateFromStream(packet) {
    if (!state.streamStartedAt) state.streamStartedAt = performance.now();
    state.missionComplete = false;
    updateCore(packet);
    setText('apxAircraftId', 'APX-MALE-01 · SIMULATION');
    setConnection('STREAM CONNECTED', 'online');
  }

  function setConnection(label, connectionState) {
    setText('apxConnection', label, connectionState || 'warning');
  }

  function markMissionComplete(summary) {
    state.missionComplete = true;
    updatePhase('RECOVERY');
    setConnection('MISSION COMPLETE', 'online');
    if (summary?.final_health_state) {
      setText('apxEvidenceTitle', `Mission complete · ${summary.final_health_state}`);
    }
  }

  function updateClocks() {
    const now = new Date();
    setText('apxUtcClock', now.toISOString().slice(11, 19) + 'Z');
    if (state.latestMissionSeconds != null) {
      setText('apxMissionElapsed', formatDuration(state.latestMissionSeconds));
    } else if (state.streamStartedAt) {
      setText('apxMissionElapsed', formatDuration((performance.now() - state.streamStartedAt) / 1000));
    }

    if (!state.lastSampleAt) {
      setText('apxFreshness', 'NO SAMPLE', 'warning');
      return;
    }
    const ageSeconds = (performance.now() - state.lastSampleAt) / 1000;
    setText('apxFreshness', ageSeconds < 1 ? '<1 s' : `${Math.floor(ageSeconds)} s`, ageSeconds < 3 ? 'healthy' : ageSeconds < 10 ? 'warning' : 'critical');
  }

  function addProvenanceLabels() {
    const labels = [
      ['healthIndex', 'MODEL-DERIVED', 'estimated'],
      ['missionRisk', 'MODEL-DERIVED', 'estimated'],
      ['rul', 'PROTOTYPE PREDICTION', 'predicted'],
      ['residual', 'PHYSICS MODEL', 'estimated']
    ];
    labels.forEach(([id, text, source]) => {
      const card = byId(id)?.closest('.card');
      const label = card?.querySelector('.kpi-label');
      if (!label || label.querySelector('.apx-provenance-tag')) return;
      const tag = document.createElement('span');
      tag.className = 'apx-provenance-tag';
      tag.dataset.source = source;
      tag.textContent = text;
      label.append(tag);
    });
  }

  function arrangeOperationalOverview() {
    const telemetryRail = document.querySelector('.grid.two[data-apx-workspace="overview"]');
    const engineTwin = document.querySelector('.cutaway-card[data-apx-workspace]');
    const evidenceRail = document.querySelector('.grid.kpis[data-apx-workspace]');
    if (!telemetryRail || !engineTwin || !evidenceRail || document.querySelector('.apx-operational-grid')) return;
    const cockpit = document.createElement('main');
    cockpit.className = 'apx-operational-grid';
    cockpit.setAttribute('aria-label', 'Propulsion intelligence workspace');
    telemetryRail.before(cockpit);
    cockpit.append(telemetryRail, engineTwin, evidenceRail);
  }

  function useTechnicalControlLabels() {
    const replacements = {
      btnModePlanner: 'Mission Design',
      btnModeExecution: 'Mission Execution',
      btnStartMission: 'Start Mission Simulation',
      btnPauseMission: 'Pause',
      btnResumeMission: 'Resume',
      btnRestartMission: 'Restart',
      btnAbortMission: 'Abort / Reset',
      btnLayerStandard: 'Standard',
      btnLayerDark: 'Dark',
      btnLayerSatellite: 'Satellite',
      btnLayerTerrain: 'Terrain',
      btnFollow: 'Follow UAV: ON',
      btnProxToggle: 'Proximity HUD',
      btnRouteToggle: 'Route',
      btnWpToggle: 'Waypoints'
    };
    Object.entries(replacements).forEach(([id, text]) => {
      const button = byId(id);
      if (button) button.textContent = text;
    });
  }

  async function requestJSON(url, options) {
    const response = await fetch(url, options);
    let payload;
    try { payload = await response.json(); } catch (_) { payload = { detail: 'Non-JSON response' }; }
    if (!response.ok) throw new Error(typeof payload.detail === 'string' ? payload.detail : JSON.stringify(payload.detail));
    return payload;
  }

  function scenarioFrom(prefix) {
    return {
      fault: 'none',
      severity: 0,
      altitude_ft: Number(byId(`${prefix}Altitude`).value),
      ambient_c: Number(byId(`${prefix}Ambient`).value),
      duration_h: Number(byId(`${prefix}Duration`).value),
      rapid_throttle: byId(`${prefix}Rapid`).checked,
      operating_state: 'CRUISE',
      simulation_mode: 'automatic'
    };
  }

  async function runWhatIfComparison() {
    const button = byId('apxRunWhatIf');
    if (!button) return;
    const original = button.textContent;
    button.disabled = true;
    button.textContent = 'Calculating…';
    try {
      const result = await requestJSON('/api/mission-whatif-rul', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ baseline: scenarioFrom('apxBase'), alternative: scenarioFrom('apxAlt') })
      });
      const baseRul = safeNumber(result.baseline?.rul?.rul_hours);
      const altRul = safeNumber(result.alternative?.rul?.rul_hours);
      const rulDelta = safeNumber(result.impact?.rul_hours ?? result.comparison?.rul_delta_hours);
      const healthDelta = safeNumber(result.impact?.health_index);
      setText('apxBaseRul', baseRul == null ? 'NOT REPORTED' : `${baseRul.toFixed(1)} h`);
      setText('apxAltRul', altRul == null ? 'NOT REPORTED' : `${altRul.toFixed(1)} h`);
      setText('apxDeltaRul', rulDelta == null ? 'NOT REPORTED' : `${rulDelta >= 0 ? '+' : ''}${rulDelta.toFixed(1)} h`, rulDelta != null && rulDelta < 0 ? 'warning' : 'healthy');
      setText('apxDeltaHealth', healthDelta == null ? 'NOT REPORTED' : `${healthDelta >= 0 ? '+' : ''}${healthDelta.toFixed(1)}`, healthDelta != null && healthDelta < 0 ? 'warning' : 'healthy');
      setText('apxWhatIfRecommendation', result.comparison?.recommendation || 'Comparison completed; inspect the reported deltas.');
      if (baseRul != null) setText('apxRulNow', `${baseRul.toFixed(1)} h`);
    } catch (error) {
      setText('apxWhatIfRecommendation', `Comparison unavailable: ${error.message}`);
    } finally {
      button.disabled = false;
      button.textContent = original;
    }
  }

  async function runReplayAnalysis() {
    const button = byId('apxRunReplay');
    if (!button) return;
    const original = button.textContent;
    button.disabled = true;
    button.textContent = 'Running replay…';
    setText('apxAircraftId', 'APX-MALE-01 · REPLAY');
    try {
      const baseScenario = typeof getScenarioPayload === 'function' ? getScenarioPayload() : {};
      const result = await requestJSON('/api/replay', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...baseScenario, steps: 48, step_minutes: 5, fault_onset_ratio: Number(byId('onset')?.value || 0.35) })
      });
      if (Array.isArray(result.timeline) && typeof drawTimeline === 'function') drawTimeline(result.timeline);
      const events = byId('events');
      if (events) {
        const summary = result.summary || {};
        const event = document.createElement('div');
        event.className = 'event';
        const title = document.createElement('strong');
        title.textContent = 'Replay analysis complete';
        event.append(title, document.createTextNode(` ${summary.steps ?? result.timeline?.length ?? 0} samples · fault onset ${summary.fault_onset_min == null ? 'not present' : `${summary.fault_onset_min} min`} · AI warning ${summary.ai_warning_min == null ? 'not triggered' : `${summary.ai_warning_min} min`}.`));
        events.replaceChildren(event);
      }
      const last = result.timeline?.at(-1);
      if (last) updateCore(last);
      setConnection('REPLAY COMPLETE', 'online');
    } catch (error) {
      const events = byId('events');
      if (events) events.textContent = `Replay unavailable: ${error.message}`;
      setConnection('REPLAY ERROR', 'offline');
    } finally {
      button.disabled = false;
      button.textContent = original;
    }
  }

  function statusCell(label, value) {
    const cell = document.createElement('div');
    const small = document.createElement('small');
    const strong = document.createElement('strong');
    small.textContent = label;
    strong.textContent = value;
    cell.append(small, strong);
    return cell;
  }

  async function loadModelStatus() {
    const container = byId('apxModelStatus');
    if (!container) return;
    container.dataset.loaded = 'loading';
    try {
      const [manifest, metrics] = await Promise.all([
        requestJSON('/api/model-manifest'),
        requestJSON('/api/metrics')
      ]);
      const accuracy = safeNumber(metrics.aces_health?.accuracy);
      const heldOut = manifest.aces?.held_out_flights?.length;
      container.replaceChildren(
        statusCell('Model Version', `${manifest.model_version || 'NOT REPORTED'} · scikit-learn ${manifest.scikit_learn || 'NOT REPORTED'}`),
        statusCell('Primary Dataset', `ACES · ${manifest.aces?.role || 'ROLE NOT REPORTED'}`),
        statusCell('Held-Out Evaluation', accuracy == null ? 'NOT REPORTED' : `${(accuracy * 100).toFixed(1)}% accuracy · ${heldOut ?? '—'} held-out flights`),
        statusCell('RUL Validation Scope', manifest.rul?.validation_gap || 'NOT REPORTED')
      );
      container.dataset.loaded = 'true';
    } catch (error) {
      container.replaceChildren(statusCell('System Status', `UNAVAILABLE · ${error.message}`));
      container.dataset.loaded = 'error';
    }
  }

  function mirrorBackendStatus() {
    const badge = byId('serverBadge');
    if (!badge) return;
    const sync = () => {
      const text = badge.textContent.replace(/^●\s*/, '').trim().toUpperCase();
      if (text.includes('OFFLINE')) setConnection('BACKEND OFFLINE', 'offline');
      else if (text.includes('READY')) setConnection('BACKEND READY', 'online');
      else if (text.includes('REQUIRED')) setConnection('SETUP REQUIRED', 'warning');
    };
    new MutationObserver(sync).observe(badge, { childList: true, characterData: true, subtree: true });
    sync();
  }

  function init() {
    all('[data-apx-view-option]').forEach((button) => button.addEventListener('click', () => setView(button.dataset.apxViewOption)));
    all('[data-apx-workspace-option]').forEach((button) => button.addEventListener('click', () => setWorkspace(button.dataset.apxWorkspaceOption)));

    let savedView = 'operator';
    let savedWorkspace = 'overview';
    try {
      savedView = localStorage.getItem('aeropulse-view') || savedView;
      savedWorkspace = localStorage.getItem('aeropulse-workspace') || savedWorkspace;
    } catch (_) { /* storage can be disabled */ }

    arrangeOperationalOverview();
    setView(savedView, false);
    setWorkspace(savedWorkspace, false);
    addProvenanceLabels();
    useTechnicalControlLabels();
    mirrorBackendStatus();
    byId('apxRunWhatIf')?.addEventListener('click', runWhatIfComparison);
    byId('apxRunReplay')?.addEventListener('click', runReplayAnalysis);
    byId('apxLoadModelStatus')?.addEventListener('click', loadModelStatus);
    byId('apxPrintReport')?.addEventListener('click', () => window.print());
    updatePhase(byId('autoPhasePill')?.textContent || 'PREFLIGHT');
    updateClocks();
    setInterval(updateClocks, 1000);

    document.addEventListener('keydown', (event) => {
      if (!event.altKey || event.ctrlKey || event.metaKey) return;
      const index = Number(event.key) - 1;
      const buttons = all('[data-apx-workspace-option]');
      if (index >= 0 && index < buttons.length) {
        event.preventDefault();
        buttons[index].click();
      }
    });
  }

  window.AeroPulseCommandUI = {
    updateFromAnalysis,
    updateFromStream,
    setConnection,
    markMissionComplete,
    setWorkspace,
    setView
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, { once: true });
  else init();
})();
