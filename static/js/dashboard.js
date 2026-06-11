/*
FIN.DR — Dashboard JavaScript Core
Handles: AJAX predictions, voice verification polling, Chart.js graphs, and live logs table.
*/

// ── Globals & State ───────────────────────────────────────────
let trendChart = null;
let pieChart = null;
let timelineChart = null;

let currentFilter = "ALL";
let currentSortCol = "created_at";
let currentSortOrder = "desc";
let transactionsData = [];

let voicePollInterval = null;
let activeVoiceTxnId = null;

// ── Initialisation ─────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  // Set current date/time on input
  const localTime = new Date();
  localTime.setMinutes(localTime.getMinutes() - localTime.getTimezoneOffset());
  document.getElementById("inTimestamp").value = localTime.toISOString().slice(0, 16);

  // Start clock
  setInterval(updateClock, 1000);
  updateClock();

  // Load Initial Data
  loadDashboardData();

  // Auto-refresh transaction logs & analytics charts every 5s
  setInterval(() => {
    if (!activeVoiceTxnId) {
      loadDashboardData();
    }
  }, 5000);
});

function updateClock() {
  const now = new Date();
  document.getElementById("topbarTime").textContent = now.toLocaleTimeString();
}

// ── Sidebar Navigation ─────────────────────────────────────────
function showPanel(panelId, element) {
  // Update nav link states
  document.querySelectorAll(".nav-item").forEach(item => item.classList.remove("active"));
  element.classList.add("active");

  // Show/Hide Panels
  document.querySelectorAll(".panel").forEach(panel => panel.classList.remove("active"));
  document.getElementById(`panel-${panelId}`).classList.add("active");

  // Update Title
  const titles = {
    analyze: "Transaction Analysis Center",
    predictions: "Neural Scoring Engine",
    voice: "Voice Risk Verification",
    analytics: "Financial Security Analytics",
    logs: "Audit & Transaction Logs",
    models: "AI Core Controller"
  };
  document.getElementById("pageTitle").textContent = titles[panelId] || "Dashboard";
}

function toggleSidebar() {
  const sidebar = document.getElementById("sidebar");
  sidebar.classList.toggle("collapsed");
}

// ── Prediction Pipeline (AJAX) ─────────────────────────────────
async function analyzeTransaction(event) {
  event.preventDefault();

  const analyzeBtn = document.getElementById("analyzeBtn");
  const analyzeBtnText = document.getElementById("analyzeBtnText");
  analyzeBtn.disabled = true;
  analyzeBtnText.textContent = "Analyzing transaction...";

  const payload = {
    amount:    parseFloat(document.getElementById("inAmount").value),
    user_id:   document.getElementById("inUserId").value,
    merchant:  document.getElementById("inMerchant").value,
    timestamp: document.getElementById("inTimestamp").value,
    card_type: document.getElementById("inCardType").value,
    txn_type:  document.getElementById("inTxnType").value,
    location:  document.getElementById("inLocation").value || "Unknown"
  };

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const data = await response.json();
    analyzeBtn.disabled = false;
    analyzeBtnText.textContent = "Analyze Transaction";

    if (data.success) {
      displayPredictionResult(data);
      loadDashboardData();
    } else {
      showToast("Prediction failed", "danger");
    }
  } catch (error) {
    analyzeBtn.disabled = false;
    analyzeBtnText.textContent = "Analyze Transaction";
    showToast("Server communication error", "danger");
  }
}

function displayPredictionResult(data) {
  const panel = document.getElementById("quickResultCard");
  panel.style.display = "block";

  // IDs
  document.getElementById("resultTxnId").textContent = `TXN ID: ${data.transaction_id}`;
  
  // Banner
  const banner = document.getElementById("decisionBanner");
  const bannerIcon = document.getElementById("decisionIcon");
  const bannerText = document.getElementById("decisionText");
  const meta = data.fusion.decision_meta;

  banner.style.backgroundColor = `${meta.color}15`;
  banner.style.border = `1px solid ${meta.color}30`;
  banner.style.color = meta.color;
  bannerIcon.innerHTML = meta.icon;
  bannerText.textContent = meta.label;

  // Progress Bars
  setProgressBar("barTf", "valTf", data.scores.tf_score, "#00d4ff");
  setProgressBar("barTr", "valTr", data.scores.transformer_score, "#7000ff");
  setProgressBar("barAn", "valAn", data.scores.anomaly_score, "#ffaa00");

  // Gauge Meter
  updateGauge(data.fusion.final_score);

  // Reason
  document.getElementById("reasonBox").innerHTML = `
    <div style="color: ${meta.color}; font-weight:700; margin-bottom: 8px;">
      ${meta.icon} ${meta.label} DECISION
    </div>
    <p>${data.fusion.reason}</p>
  `;

  // LLM Explanation
  const llmBox = document.getElementById("llmBox");
  if (data.llm_explanation) {
    llmBox.style.display = "block";
    document.getElementById("llmText").textContent = data.llm_explanation;
  } else {
    llmBox.style.display = "none";
  }

  // Voice CTA Trigger Visibility
  const voiceTriggerBtn = document.getElementById("voiceTriggerBtn");
  if (data.fusion.decision === "CALL_USER") {
    voiceTriggerBtn.style.display = "block";
    // Prepare Voice panel payload fields
    activeVoiceTxnId = data.transaction_id;
    document.getElementById("voiceTxnId").textContent = data.transaction_id;
    document.getElementById("voiceAmount").textContent = `₹${payloadAmountFormatted(data.transaction.amount)}`;
    document.getElementById("voiceMerchant").textContent = data.transaction.merchant;
  } else {
    voiceTriggerBtn.style.display = "none";
  }

  // Set KPIs on AI Scores Panel
  document.getElementById("kpiTf").textContent = `${(data.scores.tf_score * 100).toFixed(0)}%`;
  document.getElementById("kpiTr").textContent = `${(data.scores.transformer_score * 100).toFixed(0)}%`;
  document.getElementById("kpiAn").textContent = `${(data.scores.anomaly_score * 100).toFixed(0)}%`;
  
  const kpiFinal = document.getElementById("kpiFinal");
  kpiFinal.textContent = `${(data.fusion.final_score * 100).toFixed(0)}%`;
  kpiFinal.style.color = meta.color;
  document.getElementById("kpiFinalCard").style.borderColor = meta.color;

  // Auto trigger alarm toast if blocked
  if (data.fusion.decision === "BLOCK_TRANSACTION") {
    showToast(`SECURITY ALERT: Blocked transaction ${data.transaction_id}`, "danger");
  }
}

function payloadAmountFormatted(amt) {
  return parseFloat(amt).toLocaleString("en-IN", { minimumFractionDigits: 2 });
}

function setProgressBar(barId, valId, val, color) {
  const bar = document.getElementById(barId);
  const valLabel = document.getElementById(valId);
  
  bar.style.width = `${(val * 100).toFixed(0)}%`;
  bar.style.backgroundColor = color;
  bar.style.boxShadow = `0 0 10px ${color}50`;
  valLabel.textContent = `${(val * 100).toFixed(0)}%`;
}

function updateGauge(score) {
  const strokeLength = 251; // Circumference of semicircle (r=80)
  const strokeOffset = strokeLength - (score * strokeLength);
  const fill = document.getElementById("gaugeFill");
  
  fill.setAttribute("stroke-dasharray", `${strokeLength} ${strokeLength}`);
  fill.style.strokeDashoffset = strokeOffset;
  
  // Color transition based on score
  let col = "#00ffaa";
  if (score > 0.3 && score <= 0.7) col = "#ffaa00";
  if (score > 0.7) col = "#ff4455";
  fill.style.stroke = col;

  document.getElementById("gaugeValue").textContent = `${(score * 100).toFixed(0)}%`;
  document.getElementById("gaugeValue").style.fill = col;
}

// ── Voice Verification Live Loop ──────────────────────────────
function triggerVoiceVerification() {
  showPanel("voice", document.querySelector('[data-panel="voice"]'));
  startVoiceVerification();
}

async function startVoiceFromList(txnId, amount, merchant, timestamp) {
  activeVoiceTxnId = txnId;
  document.getElementById("voiceTxnId").textContent = txnId;
  document.getElementById("voiceAmount").textContent = `₹${parseFloat(amount).toLocaleString('en-IN', { minimumFractionDigits: 2 })}`;
  document.getElementById("voiceMerchant").textContent = merchant;
  
  // Switch to voice panel
  showPanel("voice", document.querySelector('[data-panel="voice"]'));
  
  // Start the verification
  await startVoiceVerificationFromTransaction(txnId, amount, merchant, timestamp);
}

async function startVoiceVerification() {
  if (!activeVoiceTxnId) {
    showToast("Please submit an analysis transaction that requires verification first.", "warning");
    return;
  }
  // Clear previous verifier UI and prepare voice UI
  document.getElementById("voiceResultBox").style.display = "none";
  document.getElementById("startVoiceBtn").disabled = true;
  document.getElementById("keyboardInputRow").style.display = "flex";

  const payload = {
    transaction_id: activeVoiceTxnId,
    amount: document.getElementById("inAmount").value,
    merchant: document.getElementById("inMerchant").value,
    timestamp: new Date().toLocaleTimeString()
  };

  try {
    const res = await fetch("/api/voice/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!data.success) throw new Error("start failed");
    showToast("Voice verification started", "info");
    // Start browser-driven speak->listen loop
    runVoiceLoop(activeVoiceTxnId, data.question);
  } catch (err) {
    console.error(err);
    showToast("Voice start API failed", "danger");
    document.getElementById("startVoiceBtn").disabled = false;
  }
}

async function startVoiceVerificationFromTransaction(txnId, amount, merchant, timestamp) {
  // Clear previous verifier UI and prepare voice UI
  document.getElementById("voiceResultBox").style.display = "none";
  document.getElementById("startVoiceBtn").disabled = true;
  document.getElementById("keyboardInputRow").style.display = "flex";

  const payload = {
    transaction_id: txnId,
    amount: amount,
    merchant: merchant,
    timestamp: timestamp
  };

  try {
    const res = await fetch(`/api/voice/verify/${txnId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!data.success) {
      showToast(`Failed to start verification: ${data.error}`, "danger");
      document.getElementById("startVoiceBtn").disabled = false;
      return;
    }
    showToast("Voice verification started for transaction", "info");
    // Start browser-driven speak->listen loop
    runVoiceLoop(txnId, data.question);
  } catch (err) {
    console.error(err);
    showToast("Voice API failed", "danger");
    document.getElementById("startVoiceBtn").disabled = false;
  }
}

// ── Browser Speech Helpers ─────────────────────────────────
function speakText(text) {
  return new Promise((resolve, reject) => {
    if (!('speechSynthesis' in window)) return reject(new Error('SpeechSynthesis unsupported'));
    try {
      const utter = new SpeechSynthesisUtterance(text);
      utter.lang = 'en-IN';
      utter.rate = 1.0;
      utter.onend = () => resolve();
      utter.onerror = (e) => reject(e);
      window.speechSynthesis.cancel(); // clear queue
      window.speechSynthesis.speak(utter);
    } catch (e) {
      reject(e);
    }
  });
}

function recogniseOnce() {
  return new Promise((resolve, reject) => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return reject(new Error('SpeechRecognition unsupported'));
    try {
      const rec = new SpeechRecognition();
      rec.lang = 'en-IN';
      rec.interimResults = false;
      rec.maxAlternatives = 1;
      let resultText = '';

      rec.onstart = () => {
        // UI handled outside
      };
      rec.onresult = (ev) => {
        if (ev.results && ev.results[0] && ev.results[0][0]) {
          resultText = ev.results[0][0].transcript;
        }
      };
      rec.onerror = (e) => {
        reject(e.error || e.message || e);
      };
      rec.onend = () => {
        resolve(resultText || '__timeout__');
      };
      rec.start();
    } catch (e) {
      reject(e);
    }
  });
}

async function runVoiceLoop(txnId, firstQuestion) {
  let question = firstQuestion;
  try {
    while (question) {
      // Speak
      _setVoicePhaseUI('SPEAKING', { question });
      try {
        await speakText(question);
      } catch (e) {
        console.warn('SpeechSynthesis failed', e);
      }

      // Listen
      _setVoicePhaseUI('LISTENING');
      let transcript = '';
      try {
        transcript = await recogniseOnce();
      } catch (e) {
        console.warn('Recognition error', e);
        transcript = '__timeout__';
      }

      // Send to backend
      _setVoicePhaseUI('PROCESSING');
      const payload = { transaction_id: txnId, response: transcript };
      const resp = await fetch('/api/voice/respond', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
      });
      const data = await resp.json();

      // If backend returns a new question, continue loop
      if (data.phase === 'QUESTION' || data.question) {
        question = data.question;
        continue;
      }

      // COMPLETE
      if (data.phase === 'COMPLETE' || data.status) {
        _setVoicePhaseUI('COMPLETE', data);
        // show results
        document.getElementById('startVoiceBtn').disabled = false;
        document.getElementById('keyboardInputRow').style.display = 'none';
        activeVoiceTxnId = null;
        loadDashboardData();
        return;
      }

      // Fallback break
      break;
    }
  } catch (err) {
    console.error('Voice loop error', err);
    showToast('Voice verification encountered an error', 'danger');
    document.getElementById('startVoiceBtn').disabled = false;
    document.getElementById('keyboardInputRow').style.display = 'none';
    activeVoiceTxnId = null;
  }
}

function _setVoicePhaseUI(phase, data = {}) {
  const ring = document.getElementById("voiceRing");
  const txt = document.getElementById("voiceStatusText");
  const badge = document.getElementById("voicePhaseBadge");
  const icon = document.getElementById("voiceStatusIcon");

  ring.className = "voice-status-ring";
  icon.className = "bi";
  badge.textContent = phase;

  if (phase === 'SPEAKING') {
    ring.classList.add('speaking');
    txt.textContent = data.question || 'Agent speaking...';
    icon.classList.add('bi-volume-up-fill');
  } else if (phase === 'LISTENING') {
    ring.classList.add('listening');
    txt.textContent = 'Listening to response...';
    icon.classList.add('bi-mic-fill');
  } else if (phase === 'PROCESSING') {
    txt.textContent = 'Processing voice patterns...';
    icon.classList.add('bi-hourglass-split');
  } else if (phase === 'COMPLETE') {
    txt.textContent = 'Verification Finished';
    icon.classList.add('bi-check-circle-fill');
    const resBox = document.getElementById('voiceResultBox');
    resBox.style.display = 'block';
    const vrStatus = document.getElementById('vrStatus');
    vrStatus.textContent = data.status || data.Status || 'COMPLETE';
    let color = '#00ffaa';
    if (data.status === 'BLOCKED') color = '#ff4455';
    if (data.status === 'FRAUD_REVIEW') color = '#ffaa00';
    vrStatus.style.color = color;
    document.getElementById('vrReason').textContent = data.reason ? `Outcome: ${data.reason}` : '';
    document.getElementById('vrFollowup').textContent = data.followup ? `Escalation: ${data.followup}` : '';
    showToast(`Voice Result: ${data.status || 'COMPLETE'}`, (data.status === 'APPROVED') ? 'success' : 'warning');
  }
}

function pollVoiceStatus() {
  if (voicePollInterval) clearInterval(voicePollInterval);

  voicePollInterval = setInterval(async () => {
    try {
      const response = await fetch(`/api/voice/status/${activeVoiceTxnId}`);
      const data = await response.json();

      updateVoiceStatusUI(data);

      if (data.phase === "COMPLETE") {
        clearInterval(voicePollInterval);
        document.getElementById("startVoiceBtn").disabled = false;
        document.getElementById("keyboardInputRow").style.display = "none";
        activeVoiceTxnId = null; // release lock
        loadDashboardData(); // update logs
      }
    } catch (e) {
      clearInterval(voicePollInterval);
      document.getElementById("startVoiceBtn").disabled = false;
    }
  }, 1500);
}

function updateVoiceStatusUI(data) {
  const ring = document.getElementById("voiceRing");
  const txt = document.getElementById("voiceStatusText");
  const badge = document.getElementById("voicePhaseBadge");
  const icon = document.getElementById("voiceStatusIcon");

  ring.className = "voice-status-ring"; // Reset classes
  icon.className = "bi"; // Reset

  badge.textContent = data.phase;

  if (data.phase === "SPEAKING") {
    ring.classList.add("speaking");
    txt.textContent = "Agent Speaking...";
    icon.classList.add("bi-volume-up-fill");
  } else if (data.phase === "LISTENING") {
    ring.classList.add("listening");
    txt.textContent = "Listening to response...";
    icon.classList.add("bi-mic-fill");
  } else if (data.phase === "PROCESSING") {
    txt.textContent = "Processing voice patterns...";
    icon.classList.add("bi-hourglass-split");
  } else if (data.phase === "COMPLETE") {
    txt.textContent = "Verification Finished";
    icon.classList.add("bi-check-circle-fill");

    // Display Voice Outcome Box
    const resBox = document.getElementById("voiceResultBox");
    resBox.style.display = "block";
    
    const vrStatus = document.getElementById("vrStatus");
    vrStatus.textContent = data.status;
    
    let color = "#00ffaa";
    if (data.status === "BLOCKED") color = "#ff4455";
    if (data.status === "FRAUD_REVIEW") color = "#ffaa00";

    vrStatus.style.color = color;
    document.getElementById("vrReason").textContent = `Outcome: ${data.reason}`;
    document.getElementById("vrFollowup").textContent = data.followup ? `Escalation: ${data.followup}` : "";
    
    showToast(`Voice Result: ${data.status}`, data.status === "APPROVED" ? "success" : "warning");
  }
}

async function sendKeyboardInput() {
  if (!activeVoiceTxnId) return;
  const inputEl = document.getElementById("keyboardInput");
  const text = inputEl.value.trim();
  if (!text) return;

  try {
    await fetch(`/api/voice/input/${activeVoiceTxnId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });
    inputEl.value = "";
    showToast("Manual response submitted", "success");
  } catch (e) {
    showToast("Could not send keyboard input", "danger");
  }
}

// ── Analytics & Charts (Chart.js) ──────────────────────────────
async function loadDashboardData() {
  try {
    // 1. Fetch Analytics
    const analyticsRes = await fetch("/api/analytics");
    const analyticsData = await analyticsRes.json();
    if (analyticsData.success) {
      updateAnalyticsWidgets(analyticsData);
      renderCharts(analyticsData);
    }

    // 2. Fetch Logs
    const logsRes = await fetch(`/api/transactions?limit=50&status=${currentFilter}`);
    const logsData = await logsRes.json();
    if (logsData.success) {
      transactionsData = logsData.transactions;
      renderLogsTable();
    }
  } catch (err) {
    console.error("Error refreshing dashboard data", err);
  }
}

function updateAnalyticsWidgets(data) {
  document.getElementById("aTotal").textContent = data.total;
  document.getElementById("aApprove").textContent = data.counts.APPROVE || 0;
  document.getElementById("aCallUser").textContent = data.counts.CALL_USER || 0;
  document.getElementById("aBlocked").textContent = data.counts.BLOCK_TRANSACTION || 0;
}

function renderCharts(data) {
  // Chart.js global settings
  Chart.defaults.color = "#8a99ad";
  Chart.defaults.font.family = "'Inter', sans-serif";

  // 1. Trend Chart
  const trendCtx = document.getElementById("trendChart").getContext("2d");
  const dates = data.trend.map(t => t.date);
  const totals = data.trend.map(t => t.total);
  const frauds = data.trend.map(t => t.fraud);

  if (trendChart) trendChart.destroy();
  trendChart = new Chart(trendCtx, {
    type: "line",
    data: {
      labels: dates,
      datasets: [
        {
          label: "Total Volume",
          data: totals,
          borderColor: "#00d4ff",
          backgroundColor: "rgba(0, 212, 255, 0.05)",
          fill: true,
          tension: 0.4
        },
        {
          label: "Fraud Blocked",
          data: frauds,
          borderColor: "#ff4455",
          backgroundColor: "rgba(255, 68, 85, 0.05)",
          fill: true,
          tension: 0.4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: "top" } },
      scales: {
        x: { grid: { color: "rgba(255,255,255,0.03)" } },
        y: { grid: { color: "rgba(255,255,255,0.03)" }, beginAtZero: true }
      }
    }
  });

  // 2. Pie Chart
  const pieCtx = document.getElementById("pieChart").getContext("2d");
  if (pieChart) pieChart.destroy();
  pieChart = new Chart(pieCtx, {
    type: "doughnut",
    data: {
      labels: ["Approved", "Review (Voice)", "Blocked"],
      datasets: [{
        data: [
          data.counts.APPROVE || 0,
          data.counts.CALL_USER || 0,
          data.counts.BLOCK_TRANSACTION || 0
        ],
        backgroundColor: ["#00ffaa", "#ffaa00", "#ff4455"],
        borderColor: "rgba(9, 14, 26, 0.95)",
        borderWidth: 2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: "bottom" } }
    }
  });

  // 3. Timeline Chart
  const timelineCtx = document.getElementById("timelineChart").getContext("2d");
  const timelineScores = data.timeline.map(t => t.final_score).reverse();
  const timelineLabels = data.timeline.map((t, idx) => `T-${20-idx}`).reverse();

  if (timelineChart) timelineChart.destroy();
  timelineChart = new Chart(timelineCtx, {
    type: "bar",
    data: {
      labels: timelineLabels,
      datasets: [{
        label: "Risk Score",
        data: timelineScores,
        backgroundColor: timelineScores.map(val => {
          if (val > 0.7) return "#ff4455";
          if (val >= 0.3) return "#ffaa00";
          return "#00ffaa";
        }),
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { max: 1.0, min: 0 }
      }
    }
  });
}

// ── Logs Filtering & Sorting ──────────────────────────────────
function renderLogsTable() {
  const tbody = document.getElementById("logsBody");
  tbody.innerHTML = "";

  if (transactionsData.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" class="empty-state">No transactions matching filter constraints.</td></tr>`;
    return;
  }

  transactionsData.forEach(txn => {
    let decMeta = { label: "Unknown", color: "#8a99ad" };
    if (txn.decision === "APPROVE") decMeta = { label: "Approved", color: "#00ffaa" };
    if (txn.decision === "CALL_USER") decMeta = { label: "Call User", color: "#ffaa00" };
    if (txn.decision === "BLOCK_TRANSACTION") decMeta = { label: "Blocked", color: "#ff4455" };

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td style="font-family: var(--font-mono); font-weight:700;">${txn.id}</td>
      <td>${txn.user_id}</td>
      <td style="font-weight:600;">₹${parseFloat(txn.amount).toFixed(2)}</td>
      <td>${txn.merchant}</td>
      <td>
        <div class="d-flex align-items-center gap-2">
          <span style="font-family: var(--font-mono);">${(txn.final_score * 100).toFixed(0)}%</span>
          <div class="score-bar-wrap" style="width: 40px; height: 4px; margin:0;">
            <div class="score-bar" style="width: ${txn.final_score * 100}%; background-color: ${decMeta.color}"></div>
          </div>
        </div>
      </td>
      <td>
        <span class="badge-decision" style="background-color: ${decMeta.color}15; color: ${decMeta.color}; border: 1px solid ${decMeta.color}30">
          ${decMeta.label}
        </span>
      </td>
      <td>
        ${txn.voice_status ? `<span class="badge-decision" style="background-color: #00ffaa15; color: #00ffaa;">${txn.voice_status}</span>` : `<button class="btn-voice-mini" onclick="startVoiceFromList('${txn.id}', '${txn.amount}', '${txn.merchant}', '${txn.timestamp}')" title="Start voice verification for this transaction"><i class="bi bi-mic-fill"></i></button>`}
      </td>
      <td style="color: var(--text-muted); font-size:12px;">${new Date(txn.created_at).toLocaleTimeString()}</td>
    `;
    tbody.appendChild(tr);
  });
}

function filterLogs() {
  const searchVal = document.getElementById("logSearch").value.toLowerCase().trim();
  if (!searchVal) {
    loadDashboardData();
    return;
  }
  const filtered = transactionsData.filter(txn => {
    return txn.id.toLowerCase().includes(searchVal) || 
           txn.merchant.toLowerCase().includes(searchVal) || 
           txn.user_id.toLowerCase().includes(searchVal);
  });
  transactionsData = filtered;
  renderLogsTable();
}

function setFilter(filter, el) {
  document.querySelectorAll(".filter-tab").forEach(tab => tab.classList.remove("active"));
  el.classList.add("active");
  currentFilter = filter;
  loadDashboardData();
}

function sortTable(col) {
  if (currentSortCol === col) {
    currentSortOrder = currentSortOrder === "asc" ? "desc" : "asc";
  } else {
    currentSortCol = col;
    currentSortOrder = "asc";
  }

  transactionsData.sort((a, b) => {
    let valA = a[col];
    let valB = b[col];

    if (col === "amount" || col === "final_score") {
      valA = parseFloat(valA);
      valB = parseFloat(valB);
    }

    if (valA < valB) return currentSortOrder === "asc" ? -1 : 1;
    if (valA > valB) return currentSortOrder === "asc" ? 1 : -1;
    return 0;
  });

  renderLogsTable();
}

// ── Retrain Controls & Reloads ───────────────────────────────
async function retrain(modelName) {
  const res = await fetch(`/api/retrain/${modelName}`, { method: "POST" });
  const data = await res.json();
  if (data.success) {
    showToast(`Retraining pipeline started for ${modelName}`, "success");
    checkRetrainStatus();
  } else {
    showToast(data.error, "danger");
  }
}

async function checkRetrainStatus() {
  const res = await fetch("/api/retrain/status");
  const data = await res.json();
  
  const tfStatus = document.getElementById("tfStatus");
  const trStatus = document.getElementById("trStatus");
  const isStatus = document.getElementById("isStatus");

  tfStatus.textContent = data.jobs.tensorflow || "Idle";
  trStatus.textContent = data.jobs.transformer || "Idle";
  isStatus.textContent = data.jobs.isolation || "Idle";
}

async function reloadModels() {
  const statusEl = document.getElementById("reloadStatus");
  statusEl.textContent = "Reloading models...";
  try {
    const res = await fetch("/api/reload_models", { method: "POST" });
    const data = await res.json();
    if (data.success) {
      statusEl.textContent = "All models reloaded successfully!";
      showToast("Models reloaded from disk", "success");
    } else {
      statusEl.textContent = `Error: ${data.error}`;
      showToast("Reload failed", "danger");
    }
  } catch (e) {
    statusEl.textContent = "Server offline or unavailable";
  }
}

// ── Toasts Utility ────────────────────────────────────────────
function showToast(message, type = "success") {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast-custom border-${type}`;
  
  let icon = "bi-check-circle-fill";
  let color = "var(--success)";
  
  if (type === "danger") { icon = "bi-x-circle-fill"; color = "var(--danger)"; }
  if (type === "warning") { icon = "bi-exclamation-triangle-fill"; color = "var(--warning)"; }
  if (type === "info") { icon = "bi-info-circle-fill"; color = "var(--info)"; }

  toast.innerHTML = `
    <i class="bi ${icon}" style="color: ${color}; font-size:20px;"></i>
    <span style="font-size:14px; font-weight:500;">${message}</span>
  `;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.animation = "slideIn 0.3s reverse forwards";
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}
