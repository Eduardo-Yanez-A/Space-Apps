"""Local server to connect the HTML interface with the ASTRA engine."""
from __future__ import annotations

from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
import pandas as pd
import ASTRA as astra

BASE_DIR = Path(__file__).resolve().parent
DATABASE = BASE_DIR / "ASTRA_Database.xlsx"
app = Flask(__name__, static_folder=None)

_db = None
_bundle = None
_training_report = None


def initialize():
    """Load the database and train the detector in memory a single time."""
    global _db, _bundle, _training_report
    if _bundle is not None:
        return
    if not DATABASE.exists():
        raise FileNotFoundError(f"Could not find {DATABASE.name} next to app.py")
    _db = astra.load_database(DATABASE)
    # train_and_evaluate prints evaluation metrics on the synthetic holdout.
    # The evaluation does not equate to clinical validation.
    _bundle = astra.train_and_evaluate(_db)


@app.get("/")
def home():
    return send_from_directory(BASE_DIR, "interfaz.html")


@app.get("/api/status")
def status():
    return jsonify({
        "ready": _bundle is not None,
        "database_found": DATABASE.exists(),
        "engine": "ASTRA / IsolationForest + IQR",
        "clinical_validation": False,
    })


@app.get("/api/metrics")
def metrics():
    try:
        initialize()
        dictionary = _db["Dictionary_24_metrics"]
        rows = []
        for _, row in dictionary.iterrows():
            rows.append({
                "name": str(row.iloc[0]),
                "type": str(row.iloc[1]),
                "unit": str(row.iloc[2]),
                "caution": str(row.iloc[4]),
            })
        return jsonify(rows)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/api/assess")
def assess():
    try:
        initialize()
        body = request.get_json(silent=True) or {}
        submitted = body.get("measurements", {})
        if not isinstance(submitted, dict):
            return jsonify({"error": "The measurements field must be a JSON object."}), 400

        # The engine requires exactly the original names of the 24 variables.
        record = {}
        for name in astra.METRICS:
            value = submitted.get(name)
            if name in astra.CAT:
                record[name] = str(value).strip() if value not in (None, "") else "Unknown"
            elif value in (None, ""):
                record[name] = None
            else:
                try:
                    record[name] = float(str(value).replace(",", "."))
                except (TypeError, ValueError):
                    return jsonify({"error": f"Invalid numeric value for: {name}"}), 400

        result = astra.assess(_bundle, record, history=None)
        return jsonify(result)
    except (ValueError, KeyError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Error in ASTRA")
        return jsonify({"error": f"Unable to complete the analysis: {exc}"}), 500


if __name__ == "__main__":
    print("Initializing ASTRA. Training may take a few seconds...")
    initialize()
    print("Open http://127.0.0.1:5000 in your browser.")
    app.run(host="127.0.0.1", port=5000, debug=False)
