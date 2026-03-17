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
