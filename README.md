# ASTRA integrated: web interface + Python engine

## Files
- `interfaz.html`: dark web panel with Roboto typography.
- `app.py`: Flask server and API that connects the page to ASTRA.
- `ASTRA.py`: original analysis engine.
- `ASTRA_Database.xlsx`: database used by the engine.
- `requirements.txt`: Python dependencies.

## Requirements
Python 3.10 or later.

## Installation
Open a terminal inside this folder and run:

```bash
python -m venv .venv
```

Activate the virtual environment:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run
```bash
python app.py
```

Then open `http://127.0.0.1:5000` in the browser. Do not open `interfaz.html` directly: the page must communicate with the local server.

## How they connect
1. The page collects values for the 24 original variables.
2. JavaScript sends a JSON payload to `POST /api/assess`.
3. Flask validates and adapts the data to the schema of `ASTRA.py`.
4. ASTRA returns assessment status, indicators, outliers, and statistical explanations.
5. The page displays the result and generates a report.

## Important notes
- The names of the 24 variables are preserved exactly as they appear in the engine and Excel sheet.
- Weight, name, photo, and notes are presentation data only; they are not sent as model variables.
- The medication status display is a separate illustrative estimate: it decreases by 0.478 percentage points per month from 100%, floored at 0%. It is not drug-specific, does not measure potency or expiration, and is not sent to ASTRA.
- The chart shows only measurements aggregated during the current browser session. It does not classify values as normal or abnormal.
- The database contains synthetic scenarios. The holdout evaluation is statistical and synthetic, not clinical validation.
- ASTRA is a research prototype: it does not diagnose, prescribe treatment, or replace medical or operational decision-making.
- The application is served locally at `127.0.0.1`; it is not configured to expose data to the internet.
