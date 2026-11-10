/**
 * GardenAI — Dashboard JS
 * Sensor cards, trend chart, relay pump control, ML assessment display
 */

// ── Chart ─────────────────────────────────────────────
let chart = null;

function initChart() {
  const ctx = document.getElementById("trendChart");
  if (!ctx) return;
  chart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        { label:"Moisture ADC", data:[], borderColor:"#3DBA82", backgroundColor:"#3DBA821A", fill:true,  tension:0.4, pointRadius:2, borderWidth:2 },
        { label:"Temp °C",      data:[], borderColor:"#E8A928", backgroundColor:"transparent", fill:false, tension:0.4, pointRadius:2, borderWidth:2 },
        { label:"Humidity %",   data:[], borderColor:"#5DB8E8", backgroundColor:"transparent", fill:false, tension:0.4, pointRadius:2, borderWidth:2 },
      ],
    },
    options: {
      responsive:true, maintainAspectRatio:false, animation:{duration:0},
      plugins:{ legend:{ display:false } },
      scales:{
        x:{ grid:{color:"rgba(61,186,130,0.07)"}, ticks:{color:"#4D7A60",font:{family:"DM Mono",size:10},maxTicksLimit:12} },
        y:{ grid:{color:"rgba(61,186,130,0.07)"}, ticks:{color:"#4D7A60",font:{family:"DM Mono",size:10}} },
      },
    },
  });
}

// ── Load plant list from DB into select ───────────────
async function loadPlantSelect() {
  const r = await fetch("/api/plants");
  const plants = await r.json();
  const sel = document.getElementById("plant-sel");
  if (!sel) return;
  sel.innerHTML = plants.map(p => `<option value="${p.name}">${p.name}</option>`).join("");
}

// ── Main update (called every 1 second by realtime.js) ─
window.onSensorUpdate = function(data) {
  const s = data.sensors;
  const h = data.history;

  // Moisture — raw ADC, inverted bar (high ADC = dry = short bar)
  updateCard("moisture", s.moisture,
    ((1023 - s.moisture) / 1023) * 100,
    s.moisture > 800 ? "danger" : s.moisture > 600 ? "warn" : "ok",
    s.moisture > 800 ? "⚠ Critically dry — relay activating"
      : s.moisture > 600 ? "⚠ Dry — pump triggered" : "✓ Good moisture"
  );

  // Temperature — raw °C
  updateCard("temp", s.temp.toFixed(2),
    (s.temp / 50) * 100,
    s.temp >= 38 ? "danger" : s.temp > 35 ? "warn" : "ok",
    s.temp >= 38 ? "⚠ Critical heat" : s.temp > 35 ? "⚠ High temp" : "✓ Normal"
  );

  // Humidity — raw %
  updateCard("humidity", s.humidity.toFixed(2),
    s.humidity,
    s.humidity < 15 ? "danger" : s.humidity < 30 ? "warn" : s.humidity > 92 ? "warn" : "ok",
    s.humidity < 15 ? "⚠ Critical low" : s.humidity < 30 ? "⚠ Low humidity" : "✓ Good"
  );

  // Pump relay
  const ring = document.getElementById("pump-ring");
  const lbl  = document.getElementById("pump-lbl");
  if (ring) ring.classList.toggle("active", s.pump);
  if (lbl)  lbl.textContent = s.pump ? "ON" : "OFF";

  // Meta chips
  set("port-chip", "Port: " + data.port);
  set("time-chip", s.last_update || "--");

  // Chart
  if (chart && h) {
    const labels = h.timestamps || [];
    set("dp-count", labels.length);
    chart.data.labels            = labels;
    chart.data.datasets[0].data  = h.moisture  || [];
    chart.data.datasets[1].data  = h.temp      || [];
    chart.data.datasets[2].data  = h.humidity  || [];
    chart.update("none");
  }

  renderAlerts(data.alerts || []);
};

function updateCard(key, displayVal, barPct, level, statusText) {
  const pct  = Math.min(100, Math.max(0, barPct));
  const vEl  = document.getElementById("v-"    + key);
  const fEl  = document.getElementById("f-"    + key);
  const stEl = document.getElementById("st-"   + key);
  const cEl  = document.getElementById("card-" + key);
  if (vEl)  vEl.textContent  = displayVal;
  if (fEl)  fEl.style.width  = pct.toFixed(1) + "%";
  if (stEl) { stEl.textContent = statusText; stEl.className = "sc-status " + level; }
  if (cEl)  cEl.className = "sensor-card" + (level !== "ok" ? " " + level : "");
}

// ── Alerts ────────────────────────────────────────────
let _lastAlertCount = -1;
function renderAlerts(alerts) {
  if (alerts.length === _lastAlertCount) return;
  _lastAlertCount = alerts.length;
  const list  = document.getElementById("alerts-list");
  const badge = document.getElementById("alert-badge");
  if (!list) return;
  if (badge) {
    badge.textContent      = alerts.length;
    badge.style.background = alerts.some(a => a.level === "danger") ? "#E8534A" : "#2A3E2C";
  }
  if (!alerts.length) { list.innerHTML = '<div class="alert-empty">No alerts.</div>'; return; }
  list.innerHTML = alerts.slice(0, 15).map(a =>
    `<div class="alert-item ${a.level}"><span class="alert-ts">${a.time}</span><span>${a.msg}</span></div>`
  ).join("");
}

// ── Relay pump command ────────────────────────────────
async function cmdPump(on) {
  const log = document.getElementById("relay-log");
  try {
    const r = await fetch("/api/pump", {
      method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ on })
    });
    const d = await r.json();
    if (log) log.textContent = d.msg + "  " + new Date().toLocaleTimeString();
  } catch(e) {
    if (log) log.textContent = "Error: " + e.message;
  }
}

// ── ML Assessment ─────────────────────────────────────
async function doAssess() {
  const plant = (document.getElementById("plant-sel")||{}).value || "Tomatoes";
  const btn   = document.getElementById("btn-assess");
  const box   = document.getElementById("ai-box");
  const row   = document.getElementById("ai-score-row");

  btn.textContent = "⏳ Analysing…"; btn.disabled = true;
  box.textContent = "Random Forest model reading live sensor data…";
  if (row) row.style.display = "none";

  try {
    const r = await fetch("/api/assess", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ plant })
    });
    const d = await r.json();

    // Score bar
    if (row && d.score !== undefined) {
      row.style.display = "block";
      const col = d.score >= 70 ? "var(--accent)" : d.score >= 40 ? "var(--amber)" : "var(--red)";
      set("ai-grade",     d.grade  || "?");
      set("ai-status",    d.label  || "");
      set("ai-score-lbl", `Health score: ${d.score} / 100`);
      set("ai-conf",      `ML confidence: ${d.confidence}%`);
      set("ai-time",      `Assessed at ${d.assessed_at}`);
      const bar = document.getElementById("ai-score-bar");
      const grade = document.getElementById("ai-grade");
      if (bar)   { bar.style.width = d.score + "%"; bar.style.background = col; }
      if (grade) grade.style.color = col;
      const issues = document.getElementById("ai-issues-wrap");
      if (issues) issues.textContent = d.issues && d.issues.length ? "⚠ " + d.issues.join("\n⚠ ") : "";
    }

    box.textContent = d.report || "No report returned.";
  } catch(e) {
    box.textContent = "ML error: " + e.message;
  }

  btn.textContent = "▶ Analyse"; btn.disabled = false;
}

function set(id, v) { const el = document.getElementById(id); if(el) el.textContent = v; }

document.addEventListener("DOMContentLoaded", () => { initChart(); loadPlantSelect(); });
