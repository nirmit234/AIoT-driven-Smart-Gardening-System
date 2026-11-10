"""
GardenAI — ML Inference Engine
================================
Loads the Random Forest model trained on REAL sensor data
(12,180 readings · 12 weeks · 84 days · 5-min intervals).

IMPORTANT — moisture scale:
  This model uses moisture as a PERCENTAGE (0-100),
  NOT raw ADC. The dataset uses percentage directly.
  If your Arduino sends raw ADC, convert before calling:
      moisture_pct = (1023 - raw_adc) / 1023 * 100

Includes three models:
  1. classifier    — health label  (Excellent/Good/Fair/Poor/Critical)
  2. regressor     — health score  (0-100, MAE ≈ 0.44 pts)
  3. pump_predictor— irrigation needed? (97%+ accuracy)
"""

import os, json
import joblib
import numpy as np
from datetime import datetime

MODEL_PATH = os.path.join(os.path.dirname(__file__), "health_model.pkl")
INFO_PATH  = os.path.join(os.path.dirname(__file__), "feature_info.json")


class MLPredictor:
    def __init__(self):
        self._bundle = None
        self._info   = None
        self._load()

    def _load(self):
        if not os.path.exists(MODEL_PATH):
            print(f"[ML] Model not found — run: python ml_model/train_model.py")
            return
        self._bundle = joblib.load(MODEL_PATH)
        with open(INFO_PATH) as f:
            self._info = json.load(f)
        src = self._info.get("data_source","?")
        acc = self._info.get("classifier_accuracy","?")
        mae = self._info.get("regressor_mae","?")
        print(f"[ML] Loaded: {src}")
        print(f"[ML] Accuracy {acc}%  |  Score MAE {mae} pts")

    def is_ready(self):
        return self._bundle is not None

    def assess(self, plant_name: str,
               moisture: float,   # percentage 0-100
               temp: float,       # °C
               humidity: float,   # %
               hist_moisture: list = None,
               hist_temp:     list = None,
               plant_profile: dict = None) -> dict:
        """
        Run ML prediction.

        moisture  — percentage (0-100). If your Arduino sends raw ADC,
                    convert first: pct = (1023 - raw_adc) / 1023 * 100
        temp      — Celsius
        humidity  — percent
        plant_profile — dict from SQLite with moisture_low/high, temp_low/high,
                        humidity_low/high  (all in % for moisture)
        """
        if not self.is_ready():
            return self._fallback(plant_name, moisture, temp, humidity)

        hist_moisture = list(hist_moisture) if hist_moisture else [moisture]
        hist_temp     = list(hist_temp)     if hist_temp     else [temp]

        # ── Rolling statistics ─────────────────────────────────────────────
        m_arr = np.array(hist_moisture[-12:] if len(hist_moisture) >= 12 else hist_moisture)
        t_arr = np.array(hist_temp[-12:]     if len(hist_temp) >= 12     else hist_temp)

        m_mean_3  = float(np.mean(m_arr[-3:]))  if len(m_arr) >= 3  else moisture
        m_mean_12 = float(np.mean(m_arr))
        m_std_12  = float(np.std(m_arr))
        m_trend_5 = float(m_arr[-1] - m_arr[0]) if len(m_arr) >= 5 else 0.0
        m_min_12  = float(np.min(m_arr))
        m_max_12  = float(np.max(m_arr))

        t_mean_3  = float(np.mean(t_arr[-3:])) if len(t_arr) >= 3 else temp
        t_mean_12 = float(np.mean(t_arr))
        t_trend_5 = float(t_arr[-1] - t_arr[0]) if len(t_arr) >= 5 else 0.0

        # Use same humidity window if available
        h_hist    = [humidity] * len(hist_moisture)
        h_arr     = np.array(h_hist[-12:])
        h_mean_3  = humidity
        h_mean_12 = humidity
        h_trend_5 = 0.0

        # ── Time features ─────────────────────────────────────────────────
        now     = datetime.now()
        hour    = now.hour
        dow     = now.weekday()
        hs      = np.sin(2 * np.pi * hour / 24)
        hc      = np.cos(2 * np.pi * hour / 24)
        ds      = np.sin(2 * np.pi * dow  / 7)
        dc      = np.cos(2 * np.pi * dow  / 7)

        # ── Plant profile distances ────────────────────────────────────────
        if plant_profile:
            m_mid = (plant_profile["moisture_low"]  + plant_profile["moisture_high"]) / 2
            t_mid = (plant_profile["temp_low"]       + plant_profile["temp_high"])     / 2
            h_mid = (plant_profile["humidity_low"]   + plant_profile["humidity_high"]) / 2
            m_in  = int(plant_profile["moisture_low"]  <= moisture  <= plant_profile["moisture_high"])
            t_in  = int(plant_profile["temp_low"]       <= temp      <= plant_profile["temp_high"])
            h_in  = int(plant_profile["humidity_low"]   <= humidity  <= plant_profile["humidity_high"])
        else:
            m_mid, t_mid, h_mid = 65, 27, 65
            m_in = t_in = h_in = 0

        m_vs = moisture - m_mid
        t_vs = temp     - t_mid
        h_vs = humidity - h_mid
        all_in = int(m_in and t_in and h_in)

        # ── Interaction features ──────────────────────────────────────────
        heat_dry  = temp * (100 - moisture) / 100
        vpd       = temp * (1 - humidity / 100)
        m_x_hum   = moisture * humidity / 100

        # ── Build feature vector (must match training order) ──────────────
        X = np.array([[
            moisture, temp, humidity,
            m_mean_3, m_mean_12, m_std_12, m_trend_5, m_min_12, m_max_12,
            t_mean_3, t_mean_12, t_trend_5,
            h_mean_3, h_mean_12, h_trend_5,
            hs, hc, ds, dc,
            m_vs, t_vs, h_vs,
            m_in, t_in, h_in, all_in,
            heat_dry, vpd, m_x_hum,
        ]])

        clf       = self._bundle["classifier"]
        reg       = self._bundle["regressor"]
        pump_clf  = self._bundle.get("pump_predictor")
        le        = self._bundle["label_encoder"]

        # ── Predictions ───────────────────────────────────────────────────
        label_enc  = clf.predict(X)[0]
        label      = le.inverse_transform([label_enc])[0]
        proba      = clf.predict_proba(X)[0]
        confidence = round(float(np.max(proba)) * 100, 1)
        score      = int(np.clip(reg.predict(X)[0], 0, 100))

        pump_needed     = False
        pump_confidence = 0.0
        if pump_clf is not None:
            pump_needed     = bool(pump_clf.predict(X)[0])
            pump_confidence = round(float(np.max(pump_clf.predict_proba(X)[0])) * 100, 1)

        grade = {"Excellent":"A","Good":"B","Fair":"C","Poor":"D","Critical":"F"}.get(label,"?")
        color = "green" if score >= 70 else ("amber" if score >= 40 else "red")

        importances = self._info.get("importances", {})
        issues      = self._detect_issues(moisture, temp, humidity,
                                           m_trend_5, t_trend_5, plant_profile, score)
        trend_msg   = self._trend_narrative(hist_moisture, hist_temp, m_trend_5, t_trend_5)

        report = self._build_report(
            plant_name, score, grade, label, confidence,
            moisture, temp, humidity, pump_needed, pump_confidence,
            issues, trend_msg, importances, plant_profile,
            m_vs, t_vs, h_vs, m_mean_12, t_mean_12
        )

        return {
            "score":            score,
            "grade":            grade,
            "label":            label,
            "confidence":       confidence,
            "color":            color,
            "pump_needed":      pump_needed,
            "pump_confidence":  pump_confidence,
            "issues":           issues,
            "trend":            trend_msg,
            "importances":      importances,
            "report":           report,
            "assessed_at":      datetime.now().strftime("%H:%M:%S"),
        }

    def _detect_issues(self, m, t, h, m_tr, t_tr, profile, score):
        issues = []
        # Real thresholds from dataset analysis
        if   m <= 40:  issues.append(f"Critically dry — moisture {m:.1f}% (pump trigger ≤40%)")
        elif m <= 45:  issues.append(f"Very dry — moisture {m:.1f}%")
        elif m <= 55:  issues.append(f"Soil drying — moisture {m:.1f}%")
        elif m >= 83:  issues.append(f"Overwatered — moisture {m:.1f}%")
        if   t >= 39:  issues.append(f"Critical heat stress — {t:.1f}°C")
        elif t >= 37:  issues.append(f"High temperature — {t:.1f}°C")
        elif t >= 35:  issues.append(f"Warm — monitor for heat stress")
        if   h <= 52:  issues.append(f"Very low humidity — {h:.1f}%")
        elif h <= 57:  issues.append(f"Low humidity — {h:.1f}%")
        elif h >= 78:  issues.append(f"High humidity — fungal risk")
        if m_tr < -2:  issues.append(f"Moisture dropping fast ({m_tr:.1f}%/5 readings)")
        if t_tr >  1:  issues.append(f"Temperature rising ({t_tr:.2f}°C/5 readings)")
        if profile:
            if not (profile["moisture_low"] <= m <= profile["moisture_high"]):
                issues.append(f"Outside ideal moisture range ({profile['moisture_low']}–{profile['moisture_high']}%)")
            if not (profile["temp_low"] <= t <= profile["temp_high"]):
                issues.append(f"Outside ideal temp range ({profile['temp_low']}–{profile['temp_high']}°C)")
        return issues

    def _trend_narrative(self, hist_m, hist_t, m_tr, t_tr):
        parts = []
        if len(hist_m) >= 5:
            if   m_tr < -3: parts.append(f"Moisture dropping rapidly ({m_tr:.1f}% over last 5 readings). Water soon.")
            elif m_tr < -1: parts.append(f"Moisture trending lower ({m_tr:.1f}%). Monitor closely.")
            elif m_tr >  3: parts.append(f"Moisture rising (+{m_tr:.1f}%). Irrigation effective.")
            else:            parts.append("Moisture is stable.")
        if len(hist_t) >= 5:
            if   t_tr > 1:  parts.append(f"Temperature rising +{t_tr:.1f}°C — watch for heat stress.")
            elif t_tr < -1: parts.append(f"Temperature cooling ({t_tr:.1f}°C) — conditions improving.")
        return " ".join(parts) if parts else "Insufficient history for trend analysis."

    def _build_report(self, plant, score, grade, label, conf,
                      m, t, h, pump, pump_conf,
                      issues, trend, importances, profile,
                      m_vs, t_vs, h_vs, m_avg, t_avg):
        lines = [
            f"=== {plant} Health Report ===",
            f"Score: {score}/100  |  Grade: {grade}  |  Status: {label}",
            f"ML Confidence: {conf}%  |  Assessed: {datetime.now().strftime('%d %b %Y  %H:%M:%S')}",
            "",
            "LIVE SENSOR READINGS",
            f"  Soil Moisture  : {m:.1f} %",
            f"  Temperature    : {t:.2f} °C",
            f"  Humidity       : {h:.2f} %",
            f"  12-reading avg moisture: {m_avg:.1f}%",
            f"  12-reading avg temp    : {t_avg:.2f}°C",
        ]
        if profile:
            lines += [
                "",
                f"IDEAL RANGES FOR {plant.upper()}",
                f"  Moisture  : {profile['moisture_low']}–{profile['moisture_high']}%"
                f"   offset from ideal: {m_vs:+.1f}%",
                f"  Temp      : {profile['temp_low']}–{profile['temp_high']}°C"
                f"   offset: {t_vs:+.2f}°C",
                f"  Humidity  : {profile['humidity_low']}–{profile['humidity_high']}%"
                f"   offset: {h_vs:+.2f}%",
            ]
        lines += ["", "TREND ANALYSIS", f"  {trend}"]
        lines += ["",
            f"IRRIGATION PREDICTION",
            f"  Pump needed : {'YES' if pump else 'NO'}  (model confidence {pump_conf}%)"]
        if issues:
            lines += ["", "ISSUES DETECTED"]
            for i in issues:
                lines.append(f"  ⚠  {i}")
        else:
            lines += ["", "✅ No issues — all conditions optimal."]
        if importances:
            top3 = sorted(importances.items(), key=lambda x: -x[1])[:3]
            lines += ["", "KEY PREDICTION FACTORS"]
            for feat, imp in top3:
                lines.append(f"  {feat.replace('_',' ').title():<28}  {imp*100:.1f}%")
        lines += ["",
            "NOTE: Model trained on 12,180 real sensor readings",
            "      (12 weeks × 84 days × 5-min intervals)"]
        return "\n".join(lines)

    def _fallback(self, plant, m, t, h):
        score = 100
        if m <= 40: score -= 40
        elif m <= 45: score -= 25
        elif m <= 55: score -= 12
        if t >= 37: score -= 18
        elif t >= 35: score -= 8
        if h <= 52: score -= 20
        elif h <= 57: score -= 10
        score = max(0, min(100, score))
        label = ("Excellent" if score>=85 else "Good" if score>=70 else
                 "Fair" if score>=50 else "Poor" if score>=30 else "Critical")
        return {"score":score,"grade":"?","label":label,"confidence":0,"color":"amber",
                "pump_needed":m<=42,"pump_confidence":0,
                "issues":["ML model not loaded — run train_model.py"],
                "trend":"No history.","importances":{},
                "report":f"Fallback. Score:{score}\nRun: python ml_model/train_model.py",
                "assessed_at":datetime.now().strftime("%H:%M:%S")}
