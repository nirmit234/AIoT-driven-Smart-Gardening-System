/**
 * GardenAI — Real-time Polling Engine
 * Polls GET /data every 1000 ms, updates navbar + footer, calls window.onSensorUpdate(data)
 */
const POLL_MS = 1000;
let failCount = 0;

document.addEventListener("DOMContentLoaded", () => {
  poll();
  setInterval(poll, POLL_MS);
  document.getElementById("hamburger")?.addEventListener("click", () =>
    document.getElementById("nav-links")?.classList.toggle("open"));
});

async function poll() {
  try {
    const resp = await fetch("/data", { cache: "no-store" });
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const data = await resp.json();
    failCount = 0;
    setNavStatus(data.sensors.connected, data.port);
    setUptime(data.uptime);
    if (typeof window.onSensorUpdate === "function") window.onSensorUpdate(data);
  } catch (err) {
    failCount++;
    if (failCount >= 5) setNavStatus(false, "");
    if (failCount === 1 || failCount % 10 === 0)
      console.warn("[GardenAI] /data failed:", err.message);
  }
}

function setNavStatus(connected, port) {
  const dot = document.getElementById("conn-dot");
  const lbl = document.getElementById("conn-label");
  if (!dot) return;
  dot.className   = "conn-dot" + (connected ? " live" : "");
  lbl.textContent = connected ? port + " · Live" : "Disconnected";
}

function setUptime(sec) {
  const el = document.getElementById("footer-uptime");
  if (!el) return;
  const h = String(Math.floor(sec/3600)).padStart(2,"0");
  const m = String(Math.floor((sec%3600)/60)).padStart(2,"0");
  const s = String(sec%60).padStart(2,"0");
  el.textContent = "Uptime: " + h + ":" + m + ":" + s;
}
