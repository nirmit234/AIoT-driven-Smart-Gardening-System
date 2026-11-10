"""
GardenAI — ML Model Trainer (Real Dataset)
===========================================
Trains on DOC-20260223-WA0001.tsv — 12,180 real readings,
12 weeks, 84 days, 5-minute intervals.
"""

import os, sys, json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, mean_absolute_error, accuracy_score

MODEL_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "health_model.pkl")
INFO_PATH  = os.path.join(MODEL_DIR, "feature_info.json")
DATA_PATH  = "/mnt/user-data/uploads/DOC-20260223-WA0001.tsv"

# Plant profiles — moisture in % (0-100) to match the dataset
PLANT_PROFILES = {
    "Tomatoes":  {"m_lo":55,"m_hi":75,"t_lo":20,"t_hi":30,"h_lo":50,"h_hi":80},
    "Basil":     {"m_lo":50,"m_hi":70,"t_lo":18,"t_hi":28,"h_lo":45,"h_hi":75},
    "Marigold":  {"m_lo":45,"m_hi":65,"t_lo":18,"t_hi":30,"h_lo":35,"h_hi":65},
    "Spinach":   {"m_lo":55,"m_hi":75,"t_lo":10,"t_hi":22,"h_lo":50,"h_hi":80},
    "Rose":      {"m_lo":50,"m_hi":70,"t_lo":15,"t_hi":28,"h_lo":45,"h_hi":70},
    "Mint":      {"m_lo":60,"m_hi":80,"t_lo":15,"t_hi":25,"h_lo":50,"h_hi":80},
    "Chili":     {"m_lo":45,"m_hi":65,"t_lo":22,"t_hi":35,"h_lo":40,"h_hi":70},
    "Cucumber":  {"m_lo":55,"m_hi":75,"t_lo":18,"t_hi":30,"h_lo":60,"h_hi":85},
    "Coriander": {"m_lo":50,"m_hi":70,"t_lo":17,"t_hi":27,"h_lo":45,"h_hi":75},
    "Pepper":    {"m_lo":50,"m_hi":72,"t_lo":20,"t_hi":32,"h_lo":50,"h_hi":80},
    "Lettuce":   {"m_lo":60,"m_hi":80,"t_lo":10,"t_hi":22,"h_lo":55,"h_hi":85},
    "Aloe Vera": {"m_lo":30,"m_hi":50,"t_lo":18,"t_hi":35,"h_lo":20,"h_hi":50},
    "Fern":      {"m_lo":65,"m_hi":84,"t_lo":15,"t_hi":25,"h_lo":60,"h_hi":90},
    "Sunflower": {"m_lo":45,"m_hi":65,"t_lo":20,"t_hi":32,"h_lo":40,"h_hi":70},
    "Orchid":    {"m_lo":50,"m_hi":70,"t_lo":18,"t_hi":28,"h_lo":55,"h_hi":80},
}


def compute_health_score(m, t, h, m_trend, profile):
    score = 100
    # Moisture — calibrated from real pump-trigger at <=42%
    if   m <= 40:  score -= 40
    elif m <= 45:  score -= 25
    elif m <= 55:  score -= 12
    elif m >= 83:  score -= 15
    elif m >= 78:  score -= 6
    # Temperature
    if   t >= 39:  score -= 35
    elif t >= 37:  score -= 18
    elif t >= 35:  score -= 8
    elif t <= 27:  score -= 3
    # Humidity
    if   h <= 52:  score -= 20
    elif h <= 57:  score -= 10
    elif h <= 59:  score -= 5
    elif h >= 78:  score -= 5
    # Trend penalty
    if m_trend < -2.0: score -= 8
    elif m_trend < -1.0: score -= 4
    # Plant profile fit
    if profile:
        m_mid = (profile["m_lo"] + profile["m_hi"]) / 2
        t_mid = (profile["t_lo"] + profile["t_hi"]) / 2
        h_mid = (profile["h_lo"] + profile["h_hi"]) / 2
        m_in = profile["m_lo"] <= m <= profile["m_hi"]
        t_in = profile["t_lo"] <= t <= profile["t_hi"]
        h_in = profile["h_lo"] <= h <= profile["h_hi"]
        if m_in and t_in and h_in:
            score += 5
        if not m_in: score -= int(min(15, abs(m - m_mid) / 5))
        if not t_in: score -= int(min(10, abs(t - t_mid) / 2))
        if not h_in: score -= int(min(8,  abs(h - h_mid) / 5))
    return max(0, min(100, int(score)))


def score_to_label(s):
    if s >= 85: return "Excellent"
    if s >= 70: return "Good"
    if s >= 50: return "Fair"
    if s >= 30: return "Poor"
    return "Critical"


def load_and_engineer(path):
    print(f"[Data] Loading {path}")
    df = pd.read_csv(path, sep="\t")
    df = df.rename(columns={
        "Temperature_degC":          "temp",
        "soil_moisture_percent":     "moisture",
        "relative_humidity_percent": "humidity",
        "Water Pump":                "pump",
    })
    df = df.dropna(subset=["temp","moisture","humidity"])
    df["pump"]    = df["pump"].notna().astype(int)
    df["hour"]    = df["time"].str.split(":").str[0].astype(int)
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    df["dow"]     = (df["day"]-1) % 7
    df["dow_sin"] = np.sin(2*np.pi*df["dow"]/7)
    df["dow_cos"] = np.cos(2*np.pi*df["dow"]/7)

    for col, c in [("moisture","m"),("temp","t"),("humidity","h")]:
        df[f"{c}_mean_3"]  = df[col].rolling(3,  min_periods=1).mean().round(2)
        df[f"{c}_mean_12"] = df[col].rolling(12, min_periods=1).mean().round(2)
        df[f"{c}_std_12"]  = df[col].rolling(12, min_periods=1).std().fillna(0).round(2)
        df[f"{c}_trend_5"] = df[col].diff(5).fillna(0).round(2)
        df[f"{c}_min_12"]  = df[col].rolling(12, min_periods=1).min()
        df[f"{c}_max_12"]  = df[col].rolling(12, min_periods=1).max()

    df["heat_dryness"]   = (df["temp"] * (100 - df["moisture"]) / 100).round(2)
    df["vpd_proxy"]      = (df["temp"] * (1 - df["humidity"]/100)).round(2)
    df["moisture_x_hum"] = (df["moisture"] * df["humidity"] / 100).round(2)
    df["pump_needed"]    = df["pump"].shift(-1).fillna(0).astype(int)

    print(f"[Data] {len(df)} rows, {df.shape[1]} columns after feature engineering")
    return df


def build_dataset(df_base):
    rows = []
    for plant_name, profile in PLANT_PROFILES.items():
        chunk = df_base.copy()
        chunk["plant"] = plant_name
        m_mid = (profile["m_lo"]+profile["m_hi"])/2
        t_mid = (profile["t_lo"]+profile["t_hi"])/2
        h_mid = (profile["h_lo"]+profile["h_hi"])/2
        chunk["m_vs_ideal"]  = (chunk["moisture"] - m_mid).round(2)
        chunk["t_vs_ideal"]  = (chunk["temp"]     - t_mid).round(2)
        chunk["h_vs_ideal"]  = (chunk["humidity"] - h_mid).round(2)
        chunk["m_in_range"]  = ((chunk["moisture"]>=profile["m_lo"])&(chunk["moisture"]<=profile["m_hi"])).astype(int)
        chunk["t_in_range"]  = ((chunk["temp"]    >=profile["t_lo"])&(chunk["temp"]    <=profile["t_hi"])).astype(int)
        chunk["h_in_range"]  = ((chunk["humidity"]>=profile["h_lo"])&(chunk["humidity"]<=profile["h_hi"])).astype(int)
        chunk["all_in_range"]= (chunk["m_in_range"]&chunk["t_in_range"]&chunk["h_in_range"]).astype(int)
        chunk["health_score"] = chunk.apply(
            lambda r: compute_health_score(r["moisture"],r["temp"],r["humidity"],r["m_trend_5"],profile), axis=1)
        chunk["health_label"] = chunk["health_score"].apply(score_to_label)
        rows.append(chunk)
    full = pd.concat(rows, ignore_index=True)
    print(f"[Data] Multi-plant: {len(full):,} rows × {len(PLANT_PROFILES)} plants")
    return full


def train():
    print("\n" + "="*60)
    print("  GardenAI ML Trainer — Real Sensor Dataset")
    print("="*60 + "\n")

    df_base = load_and_engineer(DATA_PATH)
    df      = build_dataset(df_base)

    print(f"\n[Data] Label distribution:")
    for lbl, cnt in df["health_label"].value_counts().items():
        pct = cnt/len(df)*100
        print(f"  {lbl:<12} {'█'*int(pct/2):<30}  {cnt:>7}  ({pct:.1f}%)")

    FEATURES = [
        "moisture","temp","humidity",
        "m_mean_3","m_mean_12","m_std_12","m_trend_5","m_min_12","m_max_12",
        "t_mean_3","t_mean_12","t_trend_5",
        "h_mean_3","h_mean_12","h_trend_5",
        "hour_sin","hour_cos","dow_sin","dow_cos",
        "m_vs_ideal","t_vs_ideal","h_vs_ideal",
        "m_in_range","t_in_range","h_in_range","all_in_range",
        "heat_dryness","vpd_proxy","moisture_x_hum",
    ]

    X      = df[FEATURES].values
    y_enc  = LabelEncoder().fit_transform(df["health_label"].values)
    y_score= df["health_score"].values
    y_pump = df["pump_needed"].values
    le     = LabelEncoder().fit(df["health_label"].values)
    y_enc  = le.transform(df["health_label"].values)

    X_tr, X_te, yl_tr, yl_te, ys_tr, ys_te, yp_tr, yp_te = train_test_split(
        X, y_enc, y_score, y_pump, test_size=0.15, random_state=42, stratify=y_enc)
    print(f"\n[Train] {len(X_tr):,} train  |  {len(X_te):,} test  |  {len(FEATURES)} features")

    # 1. Health classifier
    print("\n[Train] Fitting Random Forest Classifier …")
    clf = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_leaf=2,
        max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(X_tr, yl_tr)
    acc = accuracy_score(yl_te, clf.predict(X_te))
    print(f"[Train] Classifier accuracy : {acc*100:.2f}%")
    print(classification_report(yl_te, clf.predict(X_te), target_names=le.classes_))

    # 2. Score regressor
    print("[Train] Fitting Random Forest Regressor …")
    reg = RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_leaf=2,
        max_features="sqrt", random_state=42, n_jobs=-1)
    reg.fit(X_tr, ys_tr)
    mae = mean_absolute_error(ys_te, reg.predict(X_te))
    print(f"[Train] Score MAE : {mae:.2f} pts")

    # 3. Pump predictor
    print("[Train] Fitting Pump Predictor …")
    pump_clf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1)
    pump_clf.fit(X_tr, yp_tr)
    pump_acc = accuracy_score(yp_te, pump_clf.predict(X_te))
    print(f"[Train] Pump accuracy : {pump_acc*100:.2f}%")

    # 4. Cross-validation
    print("[Train] Cross-validating (5-fold) …")
    cv = cross_val_score(clf, X, y_enc, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"[Train] CV : {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%")

    # Feature importances
    importances = {f: round(float(v),5) for f,v in zip(FEATURES, clf.feature_importances_)}
    print("\n[Train] Top 10 features:")
    for f,v in sorted(importances.items(), key=lambda x:-x[1])[:10]:
        print(f"  {f:<25} {'█'*int(v*60)}  {v:.4f}")

    bundle = {
        "classifier": clf, "regressor": reg, "pump_predictor": pump_clf,
        "label_encoder": le, "features": FEATURES,
        "plant_profiles": PLANT_PROFILES,
        "trained_at": datetime.now().isoformat(),
        "training_samples": len(X_tr), "test_samples": len(X_te),
        "classifier_accuracy": round(acc*100,2), "regressor_mae": round(mae,2),
        "pump_accuracy": round(pump_acc*100,2),
        "cv_mean": round(cv.mean()*100,2), "cv_std": round(cv.std()*100,2),
        "data_source": "real_12weeks_84days_12180_readings",
        "moisture_scale": "percentage_0_100",
    }
    joblib.dump(bundle, MODEL_PATH, compress=3)

    info = {
        "features": FEATURES, "classes": list(le.classes_),
        "plants": list(PLANT_PROFILES.keys()),
        "classifier_accuracy": round(acc*100,2), "regressor_mae": round(mae,2),
        "pump_accuracy": round(pump_acc*100,2),
        "cv_accuracy": round(cv.mean()*100,2), "cv_std": round(cv.std()*100,2),
        "trained_at": datetime.now().isoformat(),
        "training_rows": len(X_tr),
        "data_source": "real_12weeks_84days_12180_readings",
        "moisture_scale": "percentage_0_100",
        "importances": importances,
    }
    with open(INFO_PATH,"w") as f:
        json.dump(info, f, indent=2)

    print(f"\n[Train] ✓ Model  → {MODEL_PATH}")
    print(f"[Train] ✓ Info   → {INFO_PATH}")
    print(f"\n{'='*60}")
    print(f"  Classifier  : {acc*100:.2f}%  (CV {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%)")
    print(f"  Score MAE   : {mae:.2f} pts")
    print(f"  Pump model  : {pump_acc*100:.2f}%")
    print(f"  Samples     : {len(X_tr):,} train / {len(X_te):,} test")
    print(f"  Plants      : {len(PLANT_PROFILES)}")
    print("="*60 + "\n")
    return bundle

if __name__ == "__main__":
    train()
