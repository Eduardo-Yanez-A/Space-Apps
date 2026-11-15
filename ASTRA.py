"""ASTRA: offline synthetic-data research prototype; Python 3.10+.
Install once: python -m pip install numpy pandas scikit-learn openpyxl
Run: python ASTRA.py (or VS Code Run Python File).
Only ASTRA_Database.xlsx beside this script is required. No files are written.
Illustrative screening rules below are inherited research heuristics, not
NASA thresholds or validated clinical guidance. No confirmed diagnoses.
All model preprocessing and statistical references use training rows only.
"""
from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime, timezone
try:
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import IsolationForest
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, RobustScaler
    from sklearn.model_selection import GroupShuffleSplit
except ImportError as exc:
    raise SystemExit("Missing dependency: " + str(exc) + "\nInstall: python -m pip install numpy pandas scikit-learn openpyxl")

METRICS = [
    "Hemoglobin (g/dl)", "Hematocrit (%)", "Leukocytes (k/uL)",
    "Calcium (mg/dL)", "Vitamin D (ng/mL)", "alkaline phosphatase (U/L)",
    "Calcium excretion (mg/24hr)", "metabolites(ng/mL)", "Heart Rate (Bpm)",
    "HRV (ms)", "Oxygen saturation (%)", "SOS bone density (m/s)",
    "Muscle Mass (kg)", "Radiation per day(millirads)",
    "Cabin Pressure/atm (kPa)", "Space Suit Pressure (kPa)",
    "EVA profile | Role",
    "Retinal & Optic Nerve Thickening(um) minimum rim width;",
    "Sodium (mmol/L)", "Sex", "Age", "Aerobic Capacity ml⁄min/kg",
    "Work W", "Height",
]

CAT = ["EVA profile | Role", "Sex"]

NUM = [c for c in METRICS if c not in CAT]

MISSING = {"undefined", "unknown", "nan", "none", "null", "", "n/a", "na", "-"}

LEVELS = {"NORMAL": 0, "MONITOR": 1, "REVIEW": 2, "URGENT": 3}

def _as_float(value: Any, spo2=False):
    if pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False:
        return np.nan
    if isinstance(value, str):
        s = value.strip()
        if s.lower() in MISSING:
            return np.nan
        if s.endswith("%"):
            if not spo2:
                return np.nan
            s = s[:-1].strip()
        s = s.replace(",", ".")
        try:
            value = float(s)
        except ValueError:
            return np.nan
    try:
        number = float(value)
        if not np.isfinite(number) or abs(number) > 1e12:
            return np.nan
        return number
    except (ValueError, TypeError, OverflowError):
        return np.nan

def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Strict schema, robust missing-value interpretation, safe numeric parsing."""
    missing = [c for c in METRICS if c not in frame.columns]
    extra = [c for c in frame.columns if c not in METRICS]
    if missing:
        raise ValueError(f"Missing required input metrics: {missing}")
    if extra:
        raise ValueError(f"Unexpected model input columns: {extra}. Use a separate metadata table.")
    out = frame[METRICS].copy()
    for col in NUM:
        out[col] = out[col].map(lambda value: _as_float(value, spo2=(col == "Oxygen saturation (%)")))
    for col in CAT:
        out[col] = out[col].map(lambda v: "Unknown" if pd.isna(v) or str(v).strip().lower() in MISSING else str(v).strip())
        if col == "Sex":
            out[col] = out[col].map(lambda v: {"f": "F", "female": "F", "m": "M", "male": "M"}.get(v.lower(), v))
        else:
            out[col] = out[col].map(lambda v: {"iva": "IVA", "eva": "EVA", "rover": "Rover", "base": "Base"}.get(v.lower(), v))
    return out

def build_pipeline(contamination: float = 0.20, random_state: int = 26) -> Pipeline:
    return Pipeline(steps=[
        ("prepare", ColumnTransformer(transformers=[
            ("numbers", Pipeline([
                ("impute", SimpleImputer(strategy="median", keep_empty_features=False)),
                ("scale", RobustScaler()),
            ]), NUM),
            ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
        ], remainder="drop", sparse_threshold=0.0)),
        ("anomaly", IsolationForest(
            n_estimators=250, max_samples=256, contamination=contamination,
            random_state=random_state, n_jobs=-1,
        )),
    ])

def train_unlabeled(cases: pd.DataFrame, *, contamination: float = 0.20) -> dict:
    X = normalize(cases)
    if len(X) < 50:
        raise ValueError("At least 50 unlabeled examples are required for this prototype.")
    usable = [c for c in NUM if X[c].notna().any()]
    if not usable:
        raise ValueError("Training requires observed numeric measurements.")
    pipeline = build_pipeline(contamination=contamination)
    pipeline.named_steps["prepare"].transformers[0] = ("numbers", pipeline.named_steps["prepare"].transformers[0][1], usable)
    pipeline.fit(X)  # NO diagnostic/label column is used here
    references = {column: {
        "median": float(X[column].median()) if X[column].notna().any() else None,
        "q25": float(X[column].quantile(.25)) if X[column].notna().any() else None,
        "q75": float(X[column].quantile(.75)) if X[column].notna().any() else None,
    } for column in NUM}
    return {"pipeline": pipeline, "reference": references,
            "schema": METRICS[:], "model": "IsolationForest unsupervised",
            "n_train": len(X), "contamination": contamination,
            "clinical_certified": False}

def _flag(flags: list, metric: str, observed: Any, level: str, issue: str, action: str):
    flags.append({"metric": metric, "value": float(observed), "level": level,
                  "finding": issue, "action": action})

def screening_rules(values: dict) -> list[dict]:
    """Research-only conservative screening, NOT astronaut medical flight rules."""
    f = []

    def check(metric, moderate, urgent, concern, action):
        v = values.get(metric)
        if v is None or pd.isna(v):
            return
        if urgent(v):
            _flag(f, metric, v, "URGENT", concern, action)
        elif moderate(v):
            _flag(f, metric, v, "REVIEW", concern, action)

    oxygen_action = "Pause exertion, secure yourself as required, and check the onboard sensor setup. If the reading remains low or you feel unwell, follow the mission medical procedure and contact the crew medical officer."
    sodium_action = "Compare the value with mission lab ranges and your flight baseline. Do not adjust salt or fluids yourself; follow the assigned hydration plan and consult the crew medical officer."
    calcium_action = "Review the confirmed value against mission lab ranges and your flight baseline with crew medical staff. Follow the assigned bone-health countermeasure plan; do not change supplements without flight-medical direction."
    hemoglobin_action = "Review the onboard result with the crew medical officer against your flight baseline and fluid-shift context before changing iron intake or supplements; report any symptoms."
    leukocyte_action = "Have crew medical staff review the result and trend; a white-cell count alone cannot identify infection or an immune cause. Follow the mission medical plan if symptoms are present."
    heart_rate_action = "Stop exertion, secure yourself as required, rest for 5 minutes, and check the onboard monitor. Contact the crew medical officer if the reading stays outside your baseline or you have symptoms."
    vitamin_d_action = "Review the result with the flight surgeon against the onboard nutrition and bone-health plan. Use only prescribed supplements and do not start or increase a dose based on this screening result."
    calcium_excretion_action = "Check the timed urine collection start/end times and completeness using the onboard lab procedure; review the trend with crew medical staff alongside calcium, vitamin D, and your flight baseline."
    alkaline_phosphatase_action = "Review the result and trend with crew medical staff alongside calcium, vitamin D, your flight baseline, and the onboard lab method; do not change supplements or countermeasures based on this value alone."
    check("Oxygen saturation (%)", lambda x: x < 94, lambda x: x < 90,
          "Low measured oxygen saturation; check signal quality and clinical context.", oxygen_action)
    check("Sodium (mmol/L)", lambda x: x < 135 or x > 145, lambda x: x < 130 or x > 150,
          "Sodium is outside an illustrative reference interval; interpret it using mission laboratory ranges and crew medical context.", sodium_action)
    check("Calcium (mg/dL)", lambda x: x < 8.5 or x > 10.5, lambda x: x < 7.5 or x > 12,
          "Serum calcium is outside an illustrative reference interval; confirm it using mission laboratory ranges and review it in the crew member's bone-health context.", calcium_action)
    check("Hemoglobin (g/dl)", lambda x: x < 12 or x > 18, lambda x: x < 8 or x > 20,
          "Hemoglobin merits medical review; interpret against sex and individual baseline.", hemoglobin_action)
    check("Leukocytes (k/uL)", lambda x: x < 4 or x > 11, lambda x: x < 2 or x > 20,
          "White blood cell count requires verification; this alone does not establish infection.", leukocyte_action)
    check("Heart Rate (Bpm)", lambda x: x < 50 or x > 110, lambda x: x < 40 or x > 140,
          "Heart rate is outside illustrative resting values; activity level and context matter.", heart_rate_action)
    check("Vitamin D (ng/mL)", lambda x: x < 20 or x > 80, lambda x: x < 12 or x > 150,
          "Vitamin D requires review with the onboard nutrition and bone-health plan; this result is not a diagnosis or a supplement dose.", vitamin_d_action)
    check("Calcium excretion (mg/24hr)", lambda x: x > 300 or x < 40, lambda x: x > 500,
          "24-hour calcium excretion is outside illustrative values; verify the timed collection and review calcium balance and bone-health trends in the microgravity mission context.", calcium_excretion_action)
    check("alkaline phosphatase (U/L)", lambda x: x < 35 or x > 135, lambda x: x > 250,
          "Alkaline phosphatase merits review alongside calcium, vitamin D, the flight baseline, and the onboard laboratory method; this value alone does not determine a supplement or countermeasure change.", alkaline_phosphatase_action)
    # Environmental measurements require vehicle-specific protocols, not a clinical range.
    c = values.get("Cabin Pressure/atm (kPa)")
    if c is not None and pd.notna(c):
        d = abs(c - 101.34) / 101.34
        if d > .05:
            _flag(f, "Cabin Pressure/atm (kPa)", c, "URGENT" if d > .10 else "REVIEW",
                  "Cabin pressure differs from the example 101.34 kPa setpoint; vehicle configuration may differ.",
                  "Check the cabin instrumentation, verify the local pressure reference, and follow the vehicle's approved environmental emergency procedures.")
    role = str(values.get("EVA profile | Role", "")).strip().upper()
    suit = values.get("Space Suit Pressure (kPa)")
    if role == "EVA" and suit is not None and pd.notna(suit):
        deviation = abs(suit - 29.5) / 29.5
        if deviation > .05:
            _flag(f, "Space Suit Pressure (kPa)", suit,
                  "URGENT" if deviation > .10 else "REVIEW",
                  "Suit pressure deviates from the example 29.5 kPa EMU reference; this does not apply to every suit.",
                  "Verify the suit configuration, check the actual pressure sensor reading, and follow the approved EVA procedure for the specific suit system before any action.")
    rad = values.get("Radiation per day(millirads)")
    if rad is not None and pd.notna(rad) and rad > 350:
        _flag(f, "Radiation per day(millirads)", rad, "REVIEW",
              "Dose rate is higher than the example 175 mrad/day reference; this is not an official exposure threshold.",
              "Verify the dosimeter reading, review the mission exposure log, and follow spacecraft radiological safety, shelter, and flight-surgeon procedures.")
    # Explicit validation of impossible measurements for data quality.
    for col in ["Oxygen saturation (%)", "Heart Rate (Bpm)", "Sodium (mmol/L)", "Age", "Height"]:
        v = values.get(col)
        if v is None or pd.isna(v):
            continue
        impossible = (v < 0 or (col == "Oxygen saturation (%)" and v > 100)
                      or (col == "Age" and v > 120) or (col == "Height" and v > 250))
        if impossible:
            _flag(f, col, v, "URGENT", "Implausible input; likely instrument, unit, or transcription error.",
                  "Verify the onboard sensor or lab source, units, and data entry; do not rely on this result until it has been checked and accurately re-entered.")
    return sorted(f, key=lambda x: LEVELS[x["level"]], reverse=True)

def personalized_trends(history: pd.DataFrame, current: dict) -> list[dict]:
    """Compare recent reading with first 3-5 personal pre-change observations.

    Sequence is determined by row order (time/astronaut ID are metadata, never features).
    """
    if history is None or len(history) < 3:
        return []
    X = normalize(history)
    baseline = X.iloc[:min(5, len(X))]
    config = {
        "Hemoglobin (g/dl)": (.15, "decrease"),
        "Hematocrit (%)": (.15, "decrease"),
        "HRV (ms)": (.30, "decrease"),
        "SOS bone density (m/s)": (.08, "decrease"),
        "Muscle Mass (kg)": (.08, "decrease"),
        "Retinal & Optic Nerve Thickening(um) minimum rim width;": (.12, "both"),
        "Aerobic Capacity ml⁄min/kg": (.15, "decrease"),
        "Work W": (.20, "decrease"),
        "Radiation per day(millirads)": (1.0, "increase"),
    }
    findings = []
    for col, (cutoff, direction) in config.items():
        base = baseline[col].median()
        observed = current.get(col)
        if pd.isna(base) or observed is None or pd.isna(observed) or abs(base) < 1e-9:
            continue
        fraction = (observed - base) / abs(base)
        triggered = ((direction == "decrease" and fraction <= -cutoff) or
                     (direction == "increase" and fraction >= cutoff) or
                     (direction == "both" and abs(fraction) >= cutoff))
        if triggered:
            findings.append({
                "metric": col, "value": float(observed), "baseline": round(float(base), 2),
                "change_pct": round(100 * fraction, 1), "level": "REVIEW",
                "finding": "Important change relative to this person's early measurements; review trend and instrument consistency.",
                "action": "Check the sensor setup and mission measurement protocol; discuss the trend with flight medicine before changing the countermeasure plan.",
            })
    return findings

def robust_population_outliers(values: dict, reference: dict, multiplier: float = 1.5) -> list[dict]:
    """Unsupervised Tukey-fence novelty detector; 1.5*IQR beyond Q1/Q3.

    It complements Isolation Forest for isolated, strong univariate changes.
    This is statistical outlier detection, NOT a clinical normal range.
    """
    unusual = []
    for col in NUM:
        x = values.get(col)
        r = reference.get(col, {})
        if x is None or pd.isna(x) or r.get("q25") is None or r.get("q75") is None:
            continue
        q1, q3 = r["q25"], r["q75"]
        iqr = q3 - q1
        if iqr <= 1e-7:
            continue
        lo, hi = q1 - multiplier * iqr, q3 + multiplier * iqr
        if x < lo or x > hi:
            unusual.append({"metric": col, "value": float(x),
                            "population_reference_lower": round(float(lo), 3),
                            "population_reference_upper": round(float(hi), 3)})
    return unusual

def _driver_scores(values: dict, reference: dict, n=5) -> list[dict]:
    scored = []
    for col in NUM:
        x = values.get(col)
        r = reference.get(col, {})
        if x is None or pd.isna(x) or r.get("median") is None:
            continue
        width = (r.get("q75") or 0) - (r.get("q25") or 0)
        if width <= 1e-7:  # constant/undefined feature is not clinically interpretable
            continue
        score = abs(x - r["median"]) / width
        scored.append({"metric": col, "value": round(float(x), 3),
                       "reference_median": round(r["median"], 3),
                       "distance_IQR": round(float(score), 2)})
    return sorted(scored, key=lambda d: d["distance_IQR"], reverse=True)[:n]

def _patterns(values: dict, flagged: list) -> list[str]:
    def val(col):
        v = values.get(col)
        return float(v) if v is not None and pd.notna(v) else np.nan
    patterns = []
    if val("Hemoglobin (g/dl)") < 12 and val("Hematocrit (%)") < 36:
        patterns.append("Possible anemia-pattern change (not a diagnosis); confirm the onboard sample and compare with the crew member's flight baseline and fluid-shift context.")
    if val("Leukocytes (k/uL)") < 4 or val("Leukocytes (k/uL)") > 11:
        patterns.append("White-cell alteration; review the onboard result and trend with crew medical staff. Infection or immune cause cannot be inferred from this count alone.")
    if val("Oxygen saturation (%)") < 94 and val("Heart Rate (Bpm)") > 105:
        patterns.append("Combined cardiorespiratory warning: follow onboard medical procedures and review sensor quality, recent exercise, and mission context.")
    if val("Vitamin D (ng/mL)") < 20 and val("Calcium excretion (mg/24hr)") > 300:
        patterns.append("Possible microgravity bone-health pattern (not a diagnosis); review calcium excretion, vitamin D, onboard nutrition, and assigned countermeasures with crew medical staff.")
    if any(x["metric"] == "Radiation per day(millirads)" for x in flagged):
        patterns.append("Radiation-monitoring event; no radiation disease can be diagnosed from this daily reading.")
    if any(x["metric"] in ("Space Suit Pressure (kPa)", "Cabin Pressure/atm (kPa)") for x in flagged):
        patterns.append("Life-support environment deviation; evaluate using mission-specific operating limits.")
    return patterns

def assess(bundle: dict, record: dict | pd.Series,
           history: pd.DataFrame | None = None) -> dict:
    raw = pd.DataFrame([dict(record)])
    X = normalize(raw)
    values = X.iloc[0].to_dict()
    available = sum(pd.notna(values[c]) for c in NUM)
    prediction = int(bundle["pipeline"].predict(X)[0]) if available >= 3 else 1
    decision = float(bundle["pipeline"].decision_function(X)[0]) if available >= 3 else None
    flags = screening_rules(values)
    quality = quality_issues(raw.iloc[0].to_dict(), values)
    encoder = bundle["pipeline"].named_steps["prepare"].named_transformers_["categories"]
    for c, known in zip(CAT, encoder.categories_):
        if values[c] not in known and values[c] != "Unknown":
            quality.append(f"{c}: unseen category {values[c]!r}; model has no learned category reference.")
    trends = personalized_trends(history, values)
    for issue in trends:
        if not any(f["metric"] == issue["metric"] for f in flags):
            flags.append(issue)
    outliers = robust_population_outliers(values, bundle["reference"])
    # Two independently unsupervised novelty detectors; no case labels used.
    ml_anomaly = prediction == -1 or len(outliers) > 0
    missing_count = sum(pd.isna(values[c]) for c in NUM)
    # A statistical outlier is a prompt to investigate, not a medical diagnosis.
    level = max((LEVELS[x["level"]] for x in flags), default=0)
    if ml_anomaly and level < 1:
        level = 1
    if (missing_count >= 7 or quality) and level < 2:
        level = 2
    severity = next(label for label, i in LEVELS.items() if i == level)
    return {
        "status": severity,
        "quality_issues": quality,
        "personal_trends": trends,
        "ml_anomaly": ml_anomaly,
        "isolation_forest_anomaly": prediction == -1,
        "iqr_outliers": outliers,
        "anomaly_margin": round(decision, 5) if decision is not None else None,
        "ml_withheld": available < 3,
        "missing_metrics": [c for c in METRICS if pd.isna(values[c]) or values[c] == "Unknown"],
        "nonmedical_score_explanation": "decision_function < 0 means statistical anomaly; not a disease probability",
        "flags": sorted(flags, key=lambda f: LEVELS[f["level"]], reverse=True),
        "statistical_drivers": _driver_scores(values, bundle["reference"]),
        "possible_patterns": _patterns(values, flags),
        "numeric_measurements_available": len(NUM) - missing_count,
        "numeric_measurements_total": len(NUM),
        "notice": "Educational research prototype. Neither the ML output nor these heuristics diagnose illness, prescribe treatment, or authorize EVA/vehicle actions. Mission-approved procedures and flight surgeons take precedence.",
    }

def evaluate_unseen(model: dict, cases: pd.DataFrame, scenario_labels: list[str]) -> dict:
    """External synthetic holdout evaluation. Labels are not model features."""
    X = normalize(cases)
    forest = model["pipeline"].predict(X) == -1
    iqr = np.array([bool(robust_population_outliers(row.to_dict(), model["reference"]))
                    for _, row in X.iterrows()], dtype=bool)
    predictions = forest | iqr
    actual = np.array([s != "Routine baseline" for s in scenario_labels], dtype=bool)
    tp = int(np.sum(predictions & actual))
    fp = int(np.sum(predictions & ~actual))
    fn = int(np.sum(~predictions & actual))
    tn = int(np.sum(~predictions & ~actual))
    forest_tp = int(np.sum(forest & actual))
    forest_fp = int(np.sum(forest & ~actual))
    return {"holdout": len(X), "isolation_forest_only_precision": round(forest_tp / max(forest_tp + forest_fp, 1), 3),
            "isolation_forest_only_recall": round(forest_tp / max(int(np.sum(actual)), 1), 3),
            "detector": "IsolationForest OR unsupervised 1.5 IQR population outlier",
            "true_positive": tp, "false_positive": fp,
            "false_negative": fn, "true_negative": tn,
            "precision": round(tp / max(tp + fp, 1), 3),
            "recall": round(tp / max(tp + fn, 1), 3),
            "warning": "Only synthetic scenarios were evaluated: NOT clinical validity or flight qualification."}

DATABASE = Path(__file__).resolve().with_name("ASTRA_Database.xlsx")
NOTICE = "This is a research prototype, not a validated medical diagnostic system. Synthetic interpretations are not confirmed diagnoses."


def quality_issues(raw, values):
    issues = []
    for c in NUM:
        v = raw.get(c)
        absent = pd.isna(v) or str(v).strip().lower() in MISSING
        if not absent and pd.isna(values[c]):
            issues.append(f"{c}: invalid/nonfinite or excessive input {v!r}; unavailable for ML.")
        if pd.notna(values[c]) and (values[c] < 0 or
                (c in ("Oxygen saturation (%)", "Hematocrit (%)") and values[c] > 100) or
                (c == "Age" and values[c] > 120) or (c == "Height" and values[c] > 250)):
            issues.append(f"{c}: implausible value {values[c]}; check units and entry.")
    return issues


def load_database(path=DATABASE):
    """Only this workbook is read. Labels are kept outside the feature table."""
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    cases = sheets["Cases_24_metrics"]
    if cases.columns.tolist() != METRICS or len(METRICS) != 24:
        raise ValueError("Cases_24_metrics must contain exactly the original 24 metric names in order.")
    meta = sheets["Case_Metadata"]
    if len(meta) != len(cases) or meta.Case_ID.isna().any() or meta.Case_ID.duplicated().any():
        raise ValueError("Case metadata must provide one unique ID per measurement row.")
    if meta.Data_Row_in_Cases_24_metrics.tolist() != list(range(2, len(cases)+2)):
        raise ValueError("Case metadata row links are not aligned.")
    for name in ("Case_Diagnoses", "Case_Classifications"):
        labels = sheets[name]
        if labels.Case_ID.duplicated().any() or not set(labels.Case_ID).issubset(set(meta.Case_ID)):
            raise ValueError(f"Invalid case links in {name}.")
    sheets["X"] = normalize(cases)
    return sheets


def train_and_evaluate(db):
    # Eligibility is provenance-based; no scenario/diagnosis is inspected here.
    eligible = db["Case_Metadata"].index[db["Case_Metadata"].Training_Eligible.eq(True)].to_numpy()
    X = db["X"].iloc[eligible]
    # Keep identical feature rows in the same partition, preventing duplicate leakage.
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    if len(np.unique(groups)) < 60:
        raise ValueError("Insufficient independent synthetic cases for train/holdout split.")
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=2026).split(X, groups=groups))
    bundle = train_unlabeled(X.iloc[train], contamination=.22)
    bundle["train_indices"] = eligible[train]
    bundle["test_indices"] = eligible[test]
    # Retrieve simulated labels only AFTER fitting. Never refit on the holdout.
    ids = db["Case_Metadata"].iloc[eligible[test]].Case_ID
    classification = db["Case_Classifications"].set_index("Case_ID")
    scenarios = classification.reindex(ids)["Scenario"]
    if scenarios.isna().any():
        raise ValueError("Missing synthetic validation classifications.")
    report = evaluate_unseen(bundle, X.iloc[test], scenarios.tolist())
    print("\nTRAINING AND SYNTHETIC HOLDOUT EVALUATION")
    print(f"Training rows: {len(train)} | Held out: {len(test)} | Original input metrics: 24")
    print("Isolation Forest and IQR use training-only imputers, encoders, scaler and quartiles.")
    print("Split is label-blind, grouped by identical measurements; no independent-person clinical validation.")
    for k, v in report.items():
        print(f"{k}: {v}")
    print("All models remain in memory. No model or result files are saved.")
    return bundle


def previous_measurements(db, idx):
    meta = db["Case_Metadata"]
    row = meta.iloc[idx]
    person, stamp = row.Astronaut_ID, row.Timestamp_UTC
    if pd.isna(person) or pd.isna(stamp):
        return None
    times = pd.to_datetime(meta.Timestamp_UTC, utc=True, errors="coerce")
    t = pd.to_datetime(stamp, utc=True, errors="coerce")
    prior = meta.index[meta.Astronaut_ID.eq(person) & times.lt(t)]
    prior = sorted(prior, key=lambda i: times.iloc[i])
    # A trend needs >=3 distinct earlier timestamps and explicit baseline provenance.
    baseline = [i for i in prior if meta.iloc[i].Baseline_Eligible == True]
    if len(baseline) < 3 or times.iloc[baseline].nunique() < 3:
        return None
    # Only documented early reference observations, never current/future readings.
    return db["X"].iloc[baseline[:5]]


def print_assessment(result, label="", partition="", history_available=False):
    print("\nASTRONAUT HEALTH ASSESSMENT", label)
    status = {"REVIEW": "MEDICAL REVIEW REQUIRED", "URGENT": "MEDICAL REVIEW REQUIRED - PRIORITY SAFETY ALERT"}.get(result["status"], result["status"])
    print("\nOverall Status:", status)
    print("Machine Learning:", "WITHHELD - insufficient observations" if result["ml_withheld"] else "Anomaly Detected" if result["ml_anomaly"] else "No anomaly detected")
    print("Isolation Forest anomaly margin:", result["anomaly_margin"], "(negative = anomaly; not disease probability)")
    print("Isolation Forest:", result["isolation_forest_anomaly"], "| Complementary IQR:", bool(result["iqr_outliers"]))
    print("Numeric observations:", result["numeric_measurements_available"], "/", result["numeric_measurements_total"])
    if partition:
        print("Dataset partition:", partition)
    print("\nAbnormal Indicators:")
    for f in result["flags"]:
        print(f"- {f['metric']}: {f['value']} [{f['level']}] {f['finding']}")
    for f in result["iqr_outliers"]:
        print(f"- {f['metric']}: {f['value']} [statistical] training IQR fence {f['population_reference_lower']} to {f['population_reference_upper']}")
    if not result["flags"] and not result["iqr_outliers"]:
        print("- None under the defined demonstration detectors; illness is not ruled out.")
    print("\nMost unusual measurements (training median/IQR distances; not causal model attribution):")
    for f in result["statistical_drivers"]:
        print(f"- {f['metric']}: {f['value']}; median {f['reference_median']}; distance {f['distance_IQR']} IQR")
    print("\nPossible Health Concerns (simulated rule interpretations):")
    for p in result["possible_patterns"] or ["No specific pattern identified; this cannot establish or exclude disease."]:
        print("-", p)
    if any(f["metric"].startswith("Retinal") for f in result["iqr_outliers"]):
        print("- Ocular/fluid-shift related change may merit same-eye OCT and expert review; no SANS diagnosis.")
    if any(f["metric"] in ("SOS bone density (m/s)", "Muscle Mass (kg)", "Aerobic Capacity ml⁄min/kg") for f in result["iqr_outliers"]):
        print("- Musculoskeletal/fitness variation may merit review in altered gravity; protocol and baseline are required.")
    print("\nPersonal baseline:", "Earlier same-astronaut reference observations available." if history_available else "Unavailable: no verified earlier reference series; population comparison only.")
    for f in result["personal_trends"]:
        print(f"- {f['metric']}: {f['value']}, early median {f['baseline']}, change {f['change_pct']}%; possible deterioration signal.")
    if history_available and not result["personal_trends"]:
        print("No configured personal change threshold exceeded; stability is not clinically established.")
    print("\nData Quality:")
    print("Unavailable metrics:", ", ".join(result["missing_metrics"]) or "none")
    for issue in result["quality_issues"]:
        print("-", issue)
    print("Missing values are imputed with TRAINING medians for statistics only; imputed values are never called measurements.")
    print("\nRecommended Actions:")
    actions = list(dict.fromkeys(f["action"] for f in result["flags"]))
    for action in actions or ["Review unusual measurements, sensor quality, units, and vehicle-specific conditions with the crew medical team before changing care or countermeasures."]:
        print("-", action)
    print("Safety alerts are independent of ML. Terrestrial screening and cabin/suit setpoints are illustrative, not astronaut operating limits.")
    print("\nImportant:\n" + NOTICE)


def select_case(db):
    query = input("Case ID, Excel data row (2+), or scenario text to list matches [blank cancels]: ").strip()
    if not query:
        return None
    meta = db["Case_Metadata"]
    if query.isdigit():
        idx = int(query)-2
        if 0 <= idx < len(meta):
            return idx
    found = meta.index[meta.Case_ID.str.casefold().eq(query.casefold())]
    if len(found):
        return int(found[0])
    labels = db["Case_Classifications"]
    matches = labels[labels.Scenario.fillna("").str.contains(query, case=False, regex=False)]
    for _, row in matches.head(20).iterrows():
        print(row.Case_ID, "|", row.Scenario)
    print("Enter an exact Case ID using option 2. Shown labels are synthetic context only.")
    return None


def manual_entry(db):
    print("Enter values in the displayed units; blank/unknown = unavailable. :cancel aborts.")
    print("SpO2 uses percent units (97 or 97%); bare 0.97 means 0.97%, not 97%.")
    values = {}
    for c in METRICS:
        while True:
            v = input(c + ": ").strip()
            if v.lower() == ":cancel":
                return None
            if c in CAT or v.lower() in MISSING:
                values[c] = v
                break
            n = _as_float(v, spo2=c == "Oxygen saturation (%)")
            if pd.isna(n):
                print("Invalid or excessive number; re-enter or leave blank. No zero will be substituted.")
                continue
            values[c] = v
            break
    person = input("Astronaut ID for optional comparison [blank = no personal baseline]: ").strip()
    history = None
    if person:
        meta = db["Case_Metadata"]
        times = pd.to_datetime(meta.Timestamp_UTC, utc=True, errors="coerce")
        rows = meta.index[meta.Astronaut_ID.eq(person) & meta.Baseline_Eligible.eq(True) & times.lt(pd.Timestamp.now(tz="UTC"))]
        rows = sorted(rows, key=lambda i: times.iloc[i])
        if len(rows) >= 3 and times.iloc[rows].nunique() >= 3:
            history = db["X"].iloc[rows[:5]]
    return values, history


def show_trends(db, bundle):
    meta = db["Case_Metadata"]
    people = meta.Astronaut_ID.dropna().unique().tolist()
    print("Available longitudinal series:", ", ".join(people) or "none")
    person = input("Astronaut ID [blank cancels]: ").strip()
    if not person:
        return
    rows = meta.index[meta.Astronaut_ID.eq(person)]
    times = pd.to_datetime(meta.Timestamp_UTC, utc=True, errors="coerce")
    rows = [i for i in rows if pd.notna(times.iloc[i])]
    rows.sort(key=lambda i: times.iloc[i])
    if not rows:
        print("No dated history for this astronaut; no personal trend inferred.")
        return
    print("Trends use the first 3-5 explicitly documented reference readings, not a clinically validated healthy baseline.")
    for idx in rows:
        history = previous_measurements(db, idx)
        findings = personalized_trends(history, db["X"].iloc[idx].to_dict()) if history is not None else []
        print(str(meta.iloc[idx].Timestamp_UTC), "|", meta.iloc[idx].Case_ID, "|", "; ".join(f"{f['metric']}: {f['change_pct']}% vs reference" for f in findings) or ("No configured change" if history is not None else "Insufficient earlier reference observations"))
    idx = rows[-1]
    history = previous_measurements(db, idx)
    print_assessment(assess(bundle, db["Cases_24_metrics"].iloc[idx], history), meta.iloc[idx].Case_ID, history_available=history is not None)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    try:
        db = load_database()
    except (OSError, ValueError, KeyError, ImportError) as exc:
        print("Cannot load ASTRA_Database.xlsx beside ASTRA.py:", exc)
        return 1
    bundle = None
    print(NOTICE)
    while True:
        print("\n========================================\n ASTRA - ASTRONAUT HEALTH MONITORING\n Machine Learning Health Assessment\n========================================\n\n[1] Train and evaluate the ML model\n[2] Analyze an astronaut from the database\n[3] Enter new astronaut health measurements\n[4] Review available health metrics\n[5] Analyze astronaut health trends\n[6] Exit\n")
        try:
            option = input("Select an option: ").strip()
            if option == "6":
                print("ASTRA closed. No persistent outputs were created.")
                return 0
            if option == "4":
                for _, row in db["Dictionary_24_metrics"].iterrows():
                    print(f"{row.iloc[0]} | {row.iloc[1]} | {row.iloc[2]} | {row.iloc[4]}")
                print("metabolites: source zero placeholders remain in raw sheets; canonical values are unavailable. No analyte is specified.")
            elif option in ("1", "2", "3", "5"):
                if bundle is None or option == "1":
                    print("Training unsupervised model in memory...")
                    bundle = train_and_evaluate(db)
                if option == "2":
                    idx = select_case(db)
                    if idx is not None:
                        h = previous_measurements(db, idx)
                        partition = "training (not an unbiased evaluation)" if idx in bundle["train_indices"] else "held-out synthetic case" if idx in bundle["test_indices"] else "excluded from training and holdout"
                        print_assessment(assess(bundle, db["Cases_24_metrics"].iloc[idx], h), db["Case_Metadata"].iloc[idx].Case_ID, partition, h is not None)
                elif option == "3":
                    entered = manual_entry(db)
                    if entered is not None:
                        values, h = entered
                        print_assessment(assess(bundle, values, h), "manual entry (session only)", history_available=h is not None)
                elif option == "5":
                    show_trends(db, bundle)
            else:
                print("Invalid option. Choose 1-6.")
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended; no data written.")
            return 0
        except (ValueError, KeyError, TypeError, OSError) as exc:
            print("Unable to complete this operation:", exc)


if __name__ == "__main__":
    raise SystemExit(main())
