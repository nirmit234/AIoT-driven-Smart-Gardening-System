"""
GardenAI — AIoT Smart Gardening System
=======================================
PyCharm: right-click app.py → Run 'app'
Browser: http://127.0.0.1:5000

Install once:
    pip install flask pyserial scikit-learn numpy pandas joblib

First run (train ML model):
    python ml_model/train_model.py

ARDUINO SERIAL OUTPUT (two lines per cycle):
    DHT11# Humidity: 32.50%  |  Temperature: 29.50°C ~ 0.00°F
    Moisture: 294

PUMP RELAY (Python → Arduino over Serial):
    Sends "PUMP_ON\n"  → Arduino sets relay pin HIGH
    Sends "PUMP_OFF\n" → Arduino sets relay pin LOW
    Arduino echoes back "ACK:PUMP_ON" or "ACK:PUMP_OFF"
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template, jsonify, request
import serial
import serial.tools.list_ports
import threading
import time
import re
from datetime import datetime
from collections import deque

from database.db import (
    init_db, get_all_plants, get_plant, add_plant,
    update_plant, delete_plant, save_reading,
    get_recent_readings, save_assessment, get_recent_assessments
)
from ml_model.predictor import MLPredictor

# ══════════════════════════════════════════════════════
#  ▶  SET YOUR ARDUINO PORT BEFORE RUNNING
# ══════════════════════════════════════════════════════
ARDUINO_PORT = "COM3"    # Windows: "COM3" / "COM4"  |  Linux/Mac: "/dev/ttyACM0"
BAUD_RATE    = 9600

# ── Alert thresholds ────────────────────────────────────────────────────────
MOISTURE_DRY  = 42     # % below this → pump ON  (real data: pumps at ≤42%)
MOISTURE_WET  = 58     # % above this → pump OFF (post-irrigation)
MOISTURE_CRIT = 40     # % below this → critical alert
TEMP_HIGH     = 35     # °C
TEMP_CRIT     = 38     # °C
HUMIDITY_LOW  = 52     # %  (from real data baseline)
HUMIDITY_CRIT = 50     # %
HUMIDITY_HIGH = 78     # % (real data max 79%)

MAX_HISTORY   = 120    # data points kept in memory for charts

# ══════════════════════════════════════════════════════
#  SHARED STATE
# ══════════════════════════════════════════════════════
_lock = threading.Lock()

sensor_state = {
    "moisture":    0,
    "temp":        0.0,
    "humidity":    0.0,
    "pump":        False,
    "connected":   False,
    "last_update": "--",
    "error":       "",
}

history = {
    "moisture":   deque(maxlen=MAX_HISTORY),
    "temp":       deque(maxlen=MAX_HISTORY),
    "humidity":   deque(maxlen=MAX_HISTORY),
    "timestamps": deque(maxlen=MAX_HISTORY),
}

alerts     = deque(maxlen=300)
start_time = time.time()


def push_alert(msg, level="ok"):
    with _lock:
        alerts.appendleft({
            "msg":   msg,
            "level": level,
            "time":  datetime.now().strftime("%H:%M:%S"),
        })


# ══════════════════════════════════════════════════════
#  SERIAL READER THREAD — real Arduino only
# ══════════════════════════════════════════════════════
class SerialReader(threading.Thread):
    """
    Reads Arduino serial output continuously.
    Parses two line formats:
        DHT11# Humidity: 32.50%  |  Temperature: 29.50°C ~ 0.00°F
        Moisture: 294
    Sends PUMP_ON / PUMP_OFF commands back over the same serial port.
    Auto-reconnects if cable is unplugged.
    """
    def __init__(self):
        super().__init__(daemon=True, name="SerialReader")
        self.ser      = None
        self._running = True

    def run(self):
        print(f"[Serial] Connecting to Arduino on {ARDUINO_PORT} at {BAUD_RATE} baud …")
        self._loop()

    def _loop(self):
        while self._running:
            try:
                self.ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=2)
                time.sleep(2)   # wait for Arduino reset on connect
                with _lock:
                    sensor_state["connected"] = True
                    sensor_state["error"]     = ""
                print(f"[Serial] ✓ Connected to {ARDUINO_PORT}")
                push_alert(f"Arduino connected — {ARDUINO_PORT}", "ok")

                while self._running:
                    raw = self.ser.readline()
                    if raw:
                        line = raw.decode("utf-8", errors="ignore").strip()
                        if line:
                            self._parse(line)

            except serial.SerialException as exc:
                err = str(exc)
                print(f"[Serial] ✗ {err}")
                push_alert(f"Serial disconnected: {err}", "danger")
                with _lock:
                    sensor_state["connected"] = False
                    sensor_state["error"]     = err
                try:
                    self.ser.close()
                except Exception:
                    pass
                time.sleep(5)   # retry after 5 s

    def _parse(self, line):
        """
        FORMAT 1 — DHT line (values change every reading):
            DHT11# Humidity: 32.50%  |  Temperature: 29.50°C ~ 0.00°F

        FORMAT 2 — Moisture line (raw ADC, changes every reading):
            Moisture: 294

        FORMAT 3 — Pump echo (optional, if Arduino sends it):
            ACK:PUMP_ON  or  ACK:PUMP_OFF
        """
        line = line.strip()
        if not line:
            return
        try:
            # DHT line
            if "Humidity:" in line and "Temperature:" in line:
                hm = re.search(r'Humidity:\s*([\d.]+)\s*%', line)
                tm = re.search(r'Temperature:\s*([\d.]+)\s*°C', line)
                if hm and tm:
                    with _lock:
                        sensor_state["humidity"] = float(hm.group(1))
                        sensor_state["temp"]     = float(tm.group(1))
                    self._record()
                    print(f"[DHT]  T={sensor_state['temp']}°C  H={sensor_state['humidity']}%")

            # Moisture line
            elif re.match(r'(?i)^moisture:\s*\d+', line):
                m = re.search(r'(?i)moisture:\s*(\d+)', line)
                if m:
                    with _lock:
                        sensor_state["moisture"] = int(m.group(1))
                    self._record()
                    print(f"[Moist] ADC={sensor_state['moisture']}")

            # Pump ACK from Arduino
            elif "ACK:PUMP_ON" in line.upper():
                with _lock:
                    sensor_state["pump"] = True
                print("[Relay] Pump confirmed ON by Arduino")

            elif "ACK:PUMP_OFF" in line.upper():
                with _lock:
                    sensor_state["pump"] = False
                print("[Relay] Pump confirmed OFF by Arduino")

        except (ValueError, AttributeError):
            pass

    def _record(self):
        """Save reading to memory history, check thresholds, save to DB."""
        ts = datetime.now().strftime("%H:%M:%S")
        with _lock:
            s = sensor_state
            s["last_update"] = ts
            history["moisture"].append(s["moisture"])
            history["temp"].append(s["temp"])
            history["humidity"].append(s["humidity"])
            history["timestamps"].append(ts)

            # Auto-irrigation via relay
            # moisture is now % — pump triggers below 42% (real dataset threshold)
            if s["moisture"] < MOISTURE_DRY and not s["pump"]:
                s["pump"] = True
                self._send_relay("PUMP_ON")
            elif s["moisture"] > MOISTURE_WET and s["pump"]:
                s["pump"] = False
                self._send_relay("PUMP_OFF")

        # Save raw reading to SQLite
        s = sensor_state
        save_reading(s["moisture"], s["temp"], s["humidity"], s["pump"])

        # Threshold alerts
        m, t, h = s["moisture"], s["temp"], s["humidity"]
        if m <= MOISTURE_CRIT:
            push_alert(f"CRITICAL: Soil extremely dry — {m:.1f}%", "danger")
        elif m < MOISTURE_DRY:
            push_alert(f"Dry soil — {m:.1f}%, pump activating", "warn")
        if t >= TEMP_CRIT:
            push_alert(f"CRITICAL: Temperature {t:.2f}°C", "danger")
        elif t > TEMP_HIGH:
            push_alert(f"High temperature: {t:.2f}°C", "warn")
        if h < HUMIDITY_CRIT:
            push_alert(f"CRITICAL: Humidity {h:.2f}%", "danger")
        elif h < HUMIDITY_LOW:
            push_alert(f"Low humidity: {h:.2f}%", "warn")
        if h > HUMIDITY_HIGH:
            push_alert(f"High humidity — fungal risk: {h:.2f}%", "warn")

    def send_pump(self, on: bool):
        """Manual pump command from dashboard button."""
        with _lock:
            sensor_state["pump"] = on
        cmd = "PUMP_ON" if on else "PUMP_OFF"
        self._send_relay(cmd)
        push_alert(f"Pump {cmd.split('_')[1]} — manual command", "ok" if on else "warn")

    def _send_relay(self, cmd: str):
        """
        Send command over serial to Arduino relay.
        Arduino receives 'PUMP_ON\\n' or 'PUMP_OFF\\n' and switches relay pin.
        """
        if self.ser and self.ser.is_open:
            try:
                self.ser.write((cmd + "\n").encode())
                print(f"[Relay] Sent → {cmd}")
            except serial.SerialException as exc:
                push_alert(f"Relay command failed: {exc}", "danger")
                print(f"[Relay] ✗ Write error: {exc}")
        else:
            print(f"[Relay] ✗ Serial not open — cannot send {cmd}")
            push_alert("Serial not connected — pump command failed", "danger")


# ══════════════════════════════════════════════════════
#  FLASK APP
# ══════════════════════════════════════════════════════
app = Flask(__name__)

# Load ML predictor at startup
ml = MLPredictor()

_reader: SerialReader = None


def _snapshot():
    with _lock:
        s  = dict(sensor_state)
        h  = {k: list(v) for k, v in history.items()}
        al = list(alerts)
    return s, h, al


# ── Page routes ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/plants")
def plants():
    return render_template("plants.html")

@app.route("/analytics")
def analytics():
    return render_template("analytics.html")


# ── Live data (polled every 1 second by browser) ─────────────────────────────
@app.route("/data")
def data():
    s, h, al = _snapshot()
    return jsonify({
        "sensors":   s,
        "history":   h,
        "alerts":    al[:25],
        "uptime":    int(time.time() - start_time),
        "port":      ARDUINO_PORT,
    })


# ── Pump relay control ───────────────────────────────────────────────────────
@app.route("/api/pump", methods=["POST"])
def api_pump():
    body = request.get_json(force=True) or {}
    on   = bool(body.get("on", False))
    if _reader:
        _reader.send_pump(on)
        return jsonify({"pump": on, "ok": True,
                        "msg": f"Relay command PUMP_{'ON' if on else 'OFF'} sent to Arduino"})
    return jsonify({"ok": False, "msg": "Serial reader not running"}), 503


# ── ML health assessment ─────────────────────────────────────────────────────
@app.route("/api/assess", methods=["POST"])
def api_assess():
    body  = request.get_json(force=True) or {}
    plant_name = body.get("plant", "Tomatoes")
    s, h, _  = _snapshot()

    profile = get_plant(plant_name)   # fetch from SQLite

    result = ml.assess(
        plant_name    = plant_name,
        moisture      = s["moisture"],
        temp          = s["temp"],
        humidity      = s["humidity"],
        hist_moisture = h["moisture"],
        hist_temp     = h["temp"],
        plant_profile = profile,
    )

    # Persist assessment to DB
    save_assessment(
        plant_name   = plant_name,
        moisture     = s["moisture"],
        temp         = s["temp"],
        humidity     = s["humidity"],
        health_score = result["score"],
        health_label = result["label"],
        confidence   = result["confidence"],
        issues       = result["issues"],
        report       = result["report"],
    )

    push_alert(
        f"ML assessed {plant_name} — {result['score']}/100 ({result['grade']})",
        "ok" if result["score"] >= 60 else "warn"
    )
    return jsonify(result)


# ── Plant profile CRUD ────────────────────────────────────────────────────────
@app.route("/api/plants", methods=["GET"])
def api_plants_list():
    return jsonify(get_all_plants())


@app.route("/api/plants", methods=["POST"])
def api_plants_add():
    b = request.get_json(force=True) or {}
    result = add_plant(
        name          = b.get("name", ""),
        moisture_low  = int(b.get("moisture_low",  200)),
        moisture_high = int(b.get("moisture_high", 500)),
        temp_low      = float(b.get("temp_low",  18)),
        temp_high     = float(b.get("temp_high", 30)),
        humidity_low  = float(b.get("humidity_low",  45)),
        humidity_high = float(b.get("humidity_high", 75)),
        care_notes    = b.get("care_notes", ""),
    )
    return jsonify(result)


@app.route("/api/plants/<int:plant_id>", methods=["PUT"])
def api_plants_update(plant_id):
    b = request.get_json(force=True) or {}
    return jsonify(update_plant(plant_id, **b))


@app.route("/api/plants/<int:plant_id>", methods=["DELETE"])
def api_plants_delete(plant_id):
    return jsonify(delete_plant(plant_id))


# ── Assessment history ────────────────────────────────────────────────────────
@app.route("/api/assessments")
def api_assessments():
    return jsonify(get_recent_assessments(20))


# ── Sensor reading history from DB ───────────────────────────────────────────
@app.route("/api/readings")
def api_readings():
    limit = int(request.args.get("limit", 100))
    return jsonify(get_recent_readings(limit))


# ── Serial ports list (debug) ─────────────────────────────────────────────────
@app.route("/api/ports")
def api_ports():
    ports = [p.device for p in serial.tools.list_ports.comports()]
    return jsonify({"ports": ports, "configured": ARDUINO_PORT})


# ── ML model info ─────────────────────────────────────────────────────────────
@app.route("/api/model-info")
def api_model_info():
    import json, os
    info_path = os.path.join("ml_model", "feature_info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Model not trained yet. Run: python ml_model/train_model.py"})


# ══════════════════════════════════════════════════════
#  STARTUP
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    # Initialise SQLite database and seed plant profiles
    init_db()

    # Start Arduino serial reader
    _reader = SerialReader()
    _reader.start()

    print(f"""
╔══════════════════════════════════════════════╗
║    GardenAI — AIoT Web Server                ║
╠══════════════════════════════════════════════╣
║  URL    :  http://127.0.0.1:5000             ║
║  Port   :  {ARDUINO_PORT:<34}║
║  Baud   :  {BAUD_RATE:<34}║
║  ML     :  Random Forest (local, no API key) ║
║  DB     :  SQLite  database/gardenai.db      ║
╚══════════════════════════════════════════════╝

  Arduino must be connected on {ARDUINO_PORT}
  Change ARDUINO_PORT at the top of app.py if needed.
  Available ports:  http://127.0.0.1:5000/api/ports

  If ML model missing:  python ml_model/train_model.py
""")

    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
