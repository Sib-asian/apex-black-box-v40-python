# Apex Black Box v4.0

Advanced live football betting analysis engine.
JavaScript Oracle Engine running inside Streamlit, with an optional Python backend.

## Setup

### 1. Clone
```bash
git clone https://github.com/Sib-asian/apex-black-box-v40-python.git
cd apex-black-box-v40-python
```

### 2. Virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 3. Run
```bash
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501`.  
A local Flask API is automatically started in the background (port range 5050-5100).  
Console output: `[apex-api] Oracle Engine API listening on http://127.0.0.1:XXXX`

---

## Engine modes

| Engine | Default | Description |
|--------|---------|-------------|
| JS | YES | JavaScript engine in-browser (unchanged behaviour) |
| PY | No | Python port running on local Flask API |

### Activate the Python engine
1. Click **⚙️ Engine: JS** button (next to the scan button) — toggles to **⚙️ Engine: PY**
2. Press **🔮 ESEGUI V40 SCAN** — calculation runs on Python backend

The toggle is per-session and resets on page reload.

### Automatic fallback
If the Python API is unreachable, the engine falls back to JS automatically and shows:
*"Backend Python non raggiungibile — calcolo eseguito con motore JS"*

---

## Architecture

```
streamlit_app.py / main.py  <- entry point; starts Flask API, injects port into HTML
apex_black_box/
  engine.py     <- Python Oracle Engine (pure function, no I/O)
  api.py        <- Flask REST API: POST /api/scan, GET /api/health
  steam.py      <- Steam movement utilities
  calibration.py
  utils.py
static/js/V40.html  <- full UI + JS Oracle Engine (default, unchanged)
```

### API reference

POST `/api/scan` — payload fields (all optional, default 0):

| Field | Type | Description |
|-------|------|-------------|
| min, rec | int | Minute, added time |
| hg, ag | int | Goals home/away |
| lastGoal | int | Minute of last goal |
| sotH/A, misH/A, corH/A, daH/A | int | Shots, corners, attacks |
| rcH/A | int | Red cards |
| tC, sC, tO, sO | float | Total/spread current+open |
| isKnockout | bool | Knockout match |
| possH, possA | int | Possession % |
| prevScans | list | Previous scan raw data for trend/confidence |

---

## Troubleshooting

- **Port conflict**: app tries 5050-5100 automatically; stop other servers if needed
- **CORS errors**: Flask-CORS is enabled; API binds to 127.0.0.1 only
- **PY vs JS differences**: minor floating-point deltas are normal; JS is the reference

---

## Logging & Evaluation

### Where logs are stored

When the **Python engine** (`⚙️ Engine: PY`) is active, every scan and final result
is automatically appended to:

```
data/logs/<match_id>.jsonl
```

Each file is a newline-delimited JSON (JSONL) stream.  One file is created per
match (the match name is normalised to a safe file-system identifier).

**Scan entries** (`type: "scan"`) are written at:

| Tag | When written |
|-----|-------------|
| `snap_20` | First scan at or after minute 20 |
| `snap_40` | First scan at or after minute 40 |
| `snap_60` | First scan at or after minute 60 |
| `snap_80` | First scan at or after minute 80 |
| `goal_event` | Any scan where the score changed since the last logged scan |
| `red_event` | Any scan where the red-card count changed since the last logged scan |

**Final score entries** (`type: "final"`) are appended when you press the
**📊 RIS.** button and the backend is reachable.

### How to record a final score

1. At full-time, click **📊 RIS.** in the memory bar.
2. Enter the final score in `H-A` format (e.g. `2-1`).
3. Click **OK** — the result is saved in localStorage *and* sent to the backend.

The backend endpoint is `POST /api/final` (body: `{ matchName, hgFT, agFT }`).
If the backend is unavailable the call fails silently and only localStorage is updated.

### How to run the evaluation script

```bash
python tools/evaluate_logs.py
# optional: specify a custom log directory
python tools/evaluate_logs.py --log-dir data/logs
```

The script requires **no external dependencies** (pure Python 3.9+).

Output includes:

- Mean Brier score per market (`1X2`, `Over2.5`, `Over1.5`, `BTTS`) broken down
  by snapshot tag and overall.
- ECE-like 10-bin calibration table for binary markets.

> **Note**: only matches that have *both* at least one scan *and* a final score
> recorded contribute to the evaluation metrics.
