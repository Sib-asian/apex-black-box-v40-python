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

---

## Streamlit Cloud deployment

Push to your GitHub repository and connect it to
[Streamlit Community Cloud](https://streamlit.io/cloud).

**No custom port configuration is required.**
The Python Oracle Engine communicates with the frontend via the Streamlit
Component bidirectional protocol (postMessage), so it works out-of-the-box
on `https://<your-app>.streamlit.app` without any additional server setup.

> **Legacy note:** `main.py` is kept as a backward-compatible entry point.
> Streamlit Cloud deployments that already point to `main.py` will continue
> to work — it simply delegates all logic to `streamlit_app.py`.

---

## Engine modes

| Engine | Default | Description |
|--------|---------|-------------|
| JS | YES | JavaScript engine in-browser (unchanged behaviour) |
| PY | No | Python Oracle Engine running server-side via Streamlit Component protocol |

### Activate the Python engine
1. Click **⚙️ Engine: JS** button (next to the scan button) — toggles to **⚙️ Engine: PY**
2. Press **🔮 ESEGUI V40 SCAN** — calculation runs on Python backend

The toggle is per-session and resets on page reload.

### Automatic fallback
If the Python engine encounters an error, the engine falls back to JS automatically and shows:
*"Backend Python non raggiungibile — calcolo eseguito con motore JS"*

---

## Architecture

```
streamlit_app.py        <- entry point; declares Streamlit Component, handles scan/final
main.py                 <- legacy alias for streamlit_app.py
apex_black_box/
  engine.py     <- Python Oracle Engine (pure function, no I/O)
  api.py        <- Flask REST API (optional; for standalone local use)
  steam.py      <- Steam movement utilities
  calibration.py
  utils.py
static/js/V40.html  <- full UI + JS Oracle Engine + Streamlit Component bridge
```

### How the Python engine works on Streamlit Cloud

1. The frontend (`static/js/V40.html`) is served as a **Streamlit Component** via
   `streamlit.components.v1.declare_component`.
2. When the user activates **Engine: PY** and runs a scan, the JS sends the payload
   to the Streamlit Python backend via `window.parent.postMessage` (Streamlit
   Component value protocol) — **no custom port or HTTP call needed**.
3. The Python backend calls `apex_black_box.engine.scan(payload)`, serialises the
   result as JSON, and passes it back to the component as a render prop.
4. The JS receives the result and renders it exactly as it would for the JS engine.

### API reference (Flask — local use only)

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

- **Engine: PY toggle shows alert**: you must be running the app inside Streamlit
  (locally or on Streamlit Cloud). Opening `V40.html` directly in a browser without
  Streamlit will limit you to the JS engine only.
- **CORS errors (local Flask)**: Flask-CORS is enabled; `api.py` binds to 127.0.0.1.
  The Flask server is **not** used by the browser on Streamlit Cloud.
- **PY vs JS differences**: minor floating-point deltas are normal; JS is the reference.
- **No logs in Streamlit Cloud "Manage app → Logs"**: see the dedicated section below.

### Verifying the Streamlit component bridge (Manage app → Logs)

When the component bridge is working correctly you should see log lines like:

```
[Apex] app started engine_version=4.0
[Apex] component request received: action=scan reqId=r1_... match=...
[Apex] component request received: action=final reqId=r2_... match=...
```

**Steps to verify:**

1. Open your Streamlit Cloud app and click **Manage app** (bottom-right) → **Logs**.
2. In the UI, enter a match name, fill in the live stats, and click **ESEGUI V40 SCAN**.
3. In Logs you should see a line containing `action=scan`.
4. Submit the final score via **📊 RIS.** — you should see `action=final`.

**If you see the red banner** _"Backend Streamlit non connesso"_ in the UI, or no
log lines appear:

- Make sure you are accessing the app via the Streamlit Cloud URL (not by opening
  `V40.html` directly in a browser).
- Check the Logs for Python startup errors (`[Apex] app started` should appear).
- Reload the page; Streamlit Cloud can occasionally take a few seconds to wake up
  after inactivity.

---

## Supabase persistence

When deployed on **Streamlit Cloud**, FINAL-score events are automatically
persisted to a [Supabase](https://supabase.com) table.

### Required secrets

Add the following secrets in your Streamlit Cloud app settings
(**Settings → Secrets**):

```toml
SUPABASE_URL = "https://<your-project>.supabase.co"
SUPABASE_KEY = "<your-anon-or-service-role-key>"
```

For local development you can set the same values as environment variables:

```bash
export SUPABASE_URL="https://<your-project>.supabase.co"
export SUPABASE_KEY="<your-key>"
```

### Table schema

Create the following table in your Supabase project (`SQL editor → New query`):

```sql
create table if not exists public.match_logs (
  id         bigserial primary key,
  match_id   text        not null,
  entry      jsonb       not null,
  created_at timestamptz not null default now()
);
```

RLS can remain disabled for private projects, or you may enable it and add
an appropriate policy for your use-case.

### What is logged

| Event | When |
|-------|------|
| `type: "final"` | User submits the full-time score via **📊 RIS.** |

Each row contains a `match_id` (derived from the match name) and an `entry`
JSON object with: `type`, `match_id`, `matchName`, `engine_version`, `ts`
(UTC ISO-8601), `hg_ft`, `ag_ft`, and a sanitized subset of the live payload.

> **Note:** if Supabase credentials are absent or the insert fails, the app
> continues normally — only a warning is printed to stderr.

---

## Local file logging (Evaluation)

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

The backend receives `POST /api/final` semantics via the component bridge
(body: `{ matchName, hgFT, agFT }`).
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

