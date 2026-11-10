"""
GardenAI — Database Manager
=============================
SQLite database for:
  - plant_profiles   : user-defined plants with ideal sensor ranges
  - sensor_readings  : full history of every Arduino reading
  - ai_assessments   : ML assessment results per reading

All access goes through this module — never query SQLite directly from app.py.
"""

import sqlite3
import os
import json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "gardenai.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # safe for multi-thread Flask
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create all tables and seed default plant profiles."""
    conn = get_conn()
    c = conn.cursor()

    # ── Plant profiles ───────────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS plant_profiles (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT    NOT NULL UNIQUE,
            moisture_low  INTEGER NOT NULL,   -- ideal ADC lower bound
            moisture_high INTEGER NOT NULL,   -- ideal ADC upper bound
            temp_low      REAL    NOT NULL,   -- ideal °C lower bound
            temp_high     REAL    NOT NULL,   -- ideal °C upper bound
            humidity_low  REAL    NOT NULL,   -- ideal % lower bound
            humidity_high REAL    NOT NULL,   -- ideal % upper bound
            care_notes    TEXT    DEFAULT '',
            created_at    TEXT    DEFAULT (datetime('now')),
            updated_at    TEXT    DEFAULT (datetime('now'))
        )
    """)

    # ── Sensor readings (every Arduino reading stored) ──────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            moisture   INTEGER NOT NULL,
            temp       REAL    NOT NULL,
            humidity   REAL    NOT NULL,
            pump       INTEGER NOT NULL DEFAULT 0,
            recorded_at TEXT   DEFAULT (datetime('now'))
        )
    """)

    # ── AI assessment results ────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS ai_assessments (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            plant_name    TEXT    NOT NULL,
            moisture      INTEGER NOT NULL,
            temp          REAL    NOT NULL,
            humidity      REAL    NOT NULL,
            health_score  INTEGER NOT NULL,
            health_label  TEXT    NOT NULL,
            confidence    REAL    NOT NULL,
            issues        TEXT    NOT NULL DEFAULT '[]',
            report        TEXT    NOT NULL DEFAULT '',
            assessed_at   TEXT    DEFAULT (datetime('now'))
        )
    """)

    conn.commit()

    # ── Seed default plant profiles if table is empty ────────────────────────
    count = c.execute("SELECT COUNT(*) FROM plant_profiles").fetchone()[0]
    if count == 0:
        defaults = [
            ("Tomatoes",  250, 500, 20, 30, 50, 80,  "Full sun. Water deeply 2-3x/week. Stake when tall."),
            ("Basil",     200, 450, 18, 28, 45, 75,  "Water at base. Pinch flowers. Hates cold drafts."),
            ("Marigold",  300, 600, 18, 30, 35, 65,  "Drought tolerant. Deadhead spent blooms. Natural pest repellent."),
            ("Spinach",   200, 450, 10, 22, 50, 80,  "Shade in heat. Harvest outer leaves first. Bolt-prone above 25°C."),
            ("Rose",      280, 520, 15, 28, 45, 70,  "Deep watering. Prune early morning. Mulch base."),
            ("Mint",      150, 380, 15, 25, 50, 80,  "Keep contained — spreads fast. Trim regularly."),
            ("Chili",     300, 580, 22, 35, 40, 70,  "Loves heat. Slight drought stress increases heat. Pinch tips early."),
            ("Cucumber",  200, 450, 18, 30, 60, 85,  "Consistent moisture essential. Train vertically. Harvest often."),
            ("Coriander", 220, 470, 17, 27, 45, 75,  "Bolts in heat — keep cool. Harvest leaves before flowering."),
            ("Pepper",    280, 550, 20, 32, 50, 80,  "Warm soil needed. Consistent watering. Support heavy fruit."),
            ("Lettuce",   180, 400, 10, 22, 55, 85,  "Partial shade in summer. Harvest outer leaves. Keep moist."),
            ("Aloe Vera", 500, 850, 18, 35, 20, 50,  "Very drought tolerant. Well-draining soil. Avoid overwatering."),
            ("Fern",      150, 350, 15, 25, 60, 90,  "Keep consistently moist. High humidity needed. Indirect light."),
            ("Sunflower", 300, 580, 20, 32, 40, 70,  "Deep roots — water less often but deeply. Full sun essential."),
            ("Orchid",    300, 550, 18, 28, 55, 80,  "Water weekly, let drain fully. Indirect bright light. High humidity."),
        ]
        c.executemany("""
            INSERT OR IGNORE INTO plant_profiles
            (name, moisture_low, moisture_high, temp_low, temp_high, humidity_low, humidity_high, care_notes)
            VALUES (?,?,?,?,?,?,?,?)
        """, defaults)
        conn.commit()
        print(f"[DB] Seeded {len(defaults)} default plant profiles")

    conn.close()
    print(f"[DB] Initialised → {DB_PATH}")


# ── Plant profile CRUD ───────────────────────────────────────────────────────

def get_all_plants():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM plant_profiles ORDER BY name").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_plant(name):
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM plant_profiles WHERE name=?", (name,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def add_plant(name, moisture_low, moisture_high, temp_low, temp_high,
              humidity_low, humidity_high, care_notes=""):
    conn = get_conn()
    try:
        conn.execute("""
            INSERT INTO plant_profiles
            (name,moisture_low,moisture_high,temp_low,temp_high,humidity_low,humidity_high,care_notes)
            VALUES (?,?,?,?,?,?,?,?)
        """, (name, moisture_low, moisture_high, temp_low, temp_high,
              humidity_low, humidity_high, care_notes))
        conn.commit()
        return {"ok": True, "msg": f"Plant '{name}' added."}
    except sqlite3.IntegrityError:
        return {"ok": False, "msg": f"Plant '{name}' already exists."}
    finally:
        conn.close()


def update_plant(plant_id, **kwargs):
    allowed = {"name","moisture_low","moisture_high","temp_low","temp_high",
               "humidity_low","humidity_high","care_notes"}
    fields  = {k: v for k, v in kwargs.items() if k in allowed}
    if not fields:
        return {"ok": False, "msg": "No valid fields to update."}
    fields["updated_at"] = datetime.now().isoformat()
    sets = ", ".join(f"{k}=?" for k in fields)
    vals = list(fields.values()) + [plant_id]
    conn = get_conn()
    conn.execute(f"UPDATE plant_profiles SET {sets} WHERE id=?", vals)
    conn.commit()
    conn.close()
    return {"ok": True, "msg": "Plant profile updated."}


def delete_plant(plant_id):
    conn = get_conn()
    conn.execute("DELETE FROM plant_profiles WHERE id=?", (plant_id,))
    conn.commit()
    conn.close()
    return {"ok": True, "msg": "Plant profile deleted."}


# ── Sensor reading history ────────────────────────────────────────────────────

def save_reading(moisture, temp, humidity, pump):
    conn = get_conn()
    conn.execute("""
        INSERT INTO sensor_readings (moisture, temp, humidity, pump)
        VALUES (?,?,?,?)
    """, (moisture, temp, humidity, int(pump)))
    conn.commit()
    conn.close()


def get_recent_readings(limit=200):
    conn = get_conn()
    rows = conn.execute("""
        SELECT * FROM sensor_readings ORDER BY id DESC LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in reversed(rows)]


# ── AI assessment history ─────────────────────────────────────────────────────

def save_assessment(plant_name, moisture, temp, humidity,
                    health_score, health_label, confidence, issues, report):
    conn = get_conn()
    conn.execute("""
        INSERT INTO ai_assessments
        (plant_name, moisture, temp, humidity, health_score,
         health_label, confidence, issues, report)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (plant_name, moisture, temp, humidity, health_score,
          health_label, confidence, json.dumps(issues), report))
    conn.commit()
    conn.close()


def get_recent_assessments(limit=20):
    conn = get_conn()
    rows = conn.execute("""
        SELECT * FROM ai_assessments ORDER BY id DESC LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["issues"] = json.loads(d["issues"])
        result.append(d)
    return result
