# Apex Black Box V40 — Python Edition

> Production-ready in-play football probabilistic engine with Poisson modelling,
> Dixon-Coles correction, xG blending, Oracle verdicts, and SQLite persistence.

---

## Features

- **Poisson Distribution** with log-space computation for numerical stability
- **Expected Goals (xG)** calculation with Dangerous Attacks soft cap (35 DA → 0.15× decay)
- **Lambda Blending** — sigmoid weight between pre-match Bayesian prior and live xG rate
- **Dixon-Coles Correction** — negative correlation between low-scoring outcomes
- **Score Effects** — garbage time, post-goal shock, red card reductions, knockout mode
- **VIX (Volatility Index)** — proprietary uncertainty measure (0–100)
- **Steam Analysis** — pre-match line movement detection with lambda modifiers
- **Oracle Verdicts** — GREEN / WATCH / SPEC / SKIP pick classification with edge calculation
- **Kelly Criterion** — fractional Kelly staking with VIX and confidence adjustments
- **SQLAlchemy ORM** — SQLite persistence for match history and scan results
- **Calibration Dashboard** — Brier score, per-market bias analysis, and recommendations
- **Input Validation** — guard conditions for all input parameters
- **Complete Type Hints** — fully mypy-compatible
- **Pytest Coverage** — 62+ unit tests

---

## Installation

```bash
git clone https://github.com/Sib-asian/apex-black-box-v40-python.git
cd apex-black-box-v40-python
pip install -r requirements.txt
pip install -e .
```

---

## Quick Start

```python
from apex_black_box import OracleEngineV40, MatchScore, MatchStats, PreMatchData

# Pre-match market data
pre = PreMatchData(
    spread_open=-0.5, spread_curr=-0.5,
    total_open=2.5,   total_curr=2.75,
    is_knockout=False,
)

# Live score at 35'
score = MatchScore(min=35, rec=0, hg=1, ag=0, last_goal=32)

# Live stats snapshot
stats = MatchStats(
    sot_h=3, mis_h=2, cor_h=2, da_h=8,  poss_h=55.0,
    sot_a=1, mis_a=1, cor_a=1, da_a=4,  poss_a=45.0,
)

engine = OracleEngineV40(pre, score, stats)
output = engine.run()

print(f"P(1):     {output.probs['1']*100:.1f}%")
print(f"P(X):     {output.probs['X']*100:.1f}%")
print(f"P(2):     {output.probs['2']*100:.1f}%")
print(f"Over 2.5: {output.probs['Over25']*100:.1f}%")
print(f"BTTS:     {output.probs['BTTS']*100:.1f}%")
print(f"Confidence: {output.confidence:.0f}%")
print(f"VIX:        {output.vix:.0f}%")
```

---

## Kelly Criterion Example

```python
from apex_black_box import KellyCriterion

kelly = KellyCriterion(bankroll=1000.0, max_kelly=0.25, fraction=0.5)

result = kelly.calculate(
    prob=0.65,       # Model probability
    odds=2.10,       # Bookie decimal odds
    vix=45.0,        # VIX from oracle output
    confidence=70.0, # Confidence from oracle output
)

print(f"Edge:             {result['edge']*100:.1f}%")
print(f"EV:               {result['ev']*100:.1f}%")
print(f"Full Kelly:       {result['kelly_full']*100:.1f}%")
print(f"Fractional Kelly: {result['kelly_fractional']*100:.1f}%")
print(f"Stake:            {result['stake']:.2f}")
```

---

## Oracle Verdicts Example

```python
from apex_black_box import OracleVerdictGenerator

generator = OracleVerdictGenerator(output)

bookie_odds = {
    "1": 1.55, "X": 3.80, "2": 5.50,
    "Over25": 2.10, "Under25": 1.75,
    "BTTS": 2.20,
}

verdicts = generator.generate_verdicts(bookie_odds)
for v in verdicts:
    print(f"[{v['classification']}] {v['market']} "
          f"@ {v['odds']} | edge={v['edge']*100:.1f}%")

best = generator.get_best_pick(bookie_odds)
if best:
    print(f"Best pick: {best['classification']} on {best['market']}")
```

---

## Steam Analysis

```python
from apex_black_box import SteamAnalyzer

steam = SteamAnalyzer(
    spread_open=-0.5, spread_curr=-1.0,
    total_open=2.5,   total_curr=2.75,
)

spread_info = steam.detect_spread_movement()
total_info  = steam.detect_total_movement()
modifiers   = steam.get_lambda_modifiers()
advice      = steam.generate_advice()

print(f"Spread direction: {spread_info['direction']}")
print(f"Lambda modifiers: {modifiers}")
for tip in advice:
    print(f"  * {tip}")
```

---

## Database Persistence

```python
from apex_black_box import DatabaseManager

db = DatabaseManager("sqlite:///my_bets.db")
db.create_tables()

match_id = db.save_match(
    home="Arsenal", away="Chelsea",
    date="2025-01-15",
    pre_data={"spread_open": -0.5, "total_open": 2.5},
)

scan_id = db.save_scan(match_id=match_id, output=output)

recent = db.get_recent_matches(limit=10)
```

---

## Calibration Dashboard

```python
from apex_black_box import CalibrationDashboard

dash = CalibrationDashboard()
dash.add_bet_data({"market": "1",      "prediction": 0.65, "outcome": 1, "odds": 1.80, "stake": 50.0})
dash.add_bet_data({"market": "Over25", "prediction": 0.55, "outcome": 0, "odds": 1.90, "stake": 30.0})

accuracy = dash.analyze_accuracy()
biases   = dash.analyze_biases()
recs     = dash.generate_recommendations()

print(f"Mean Brier: {accuracy['mean_brier']:.4f}")
for rec in recs:
    print(f"  * {rec}")
```

---

## Mathematical Model

### Poisson with Dixon-Coles Correction

```
P(H=h, A=a) = Poisson(h|lam_h) * Poisson(a|lam_a) * DC(h, a)

DC = 1 + rho * f(h, a)
rho        = -0.10 * dc_scale * time_decay
dc_scale   = 1 / (1 + max(0, lam_h * lam_a - 1) * 0.35)
time_decay = max(0.05, 1 - min/90)
```

### Expected Goals

```
xG_shots = SoT * (0.10 + conv_rate * 0.04) + Miss * (0.02 + conv_rate * 0.02)
xG_DA    = DA_eff * 0.003
DA_eff   = min(DA, 35) + max(0, DA - 35) * 0.15   <- soft cap

lambda_remain = blended_rate * (minutes_remaining / 90)
```

### Lambda Blending (Sigmoid)

```
w       = 1 / (1 + exp(-0.10 * (min - 45)))
blended = w * xG_rate + (1 - w) * prior
lambda  = blended * (minutes_remaining / 90)
```

### Score Effects

```
Garbage Time: |goal_diff| >= 3  OR  (|goal_diff| >= 2  AND  min > 80)
  -> lam_leading  * 0.55
  -> lam_trailing * 1.05  (chasing)

Post-goal Shock (within 6 min of goal):
  shock_mult = 1.0 + 0.32 * exp(-0.22 * delta_min)
  both teams: lam / shock_mult

Red Card: reduction = 1 - min(rc, 2) * 0.40 * rd
  where rd = max(0.15, (90 - min) / 90)
```

### Kelly Criterion

```
f* = (b * p - q) / b    where b = odds - 1, q = 1 - p
Adjustments:
  * (confidence / 100)
  * 0.80  if VIX > 60
  = 0     if VIX < 30
  * fraction  (default 0.5 = half-Kelly)
  stake = min(f* * bankroll, bankroll * max_kelly)
```

---

## Project Structure

```
apex-black-box-v40-python/
+-- README.md
+-- requirements.txt
+-- setup.py
+-- .gitignore
|
+-- apex_black_box/
|   +-- __init__.py          Package exports (v4.0.0)
|   +-- core.py              OracleEngineV40 - Poisson, xG, blend, score effects
|   +-- steam.py             SteamAnalyzer - line movement detection
|   +-- oracle.py            OracleVerdictGenerator - GREEN/WATCH/SPEC/SKIP
|   +-- kelly.py             KellyCriterion - fractional Kelly staking
|   +-- database.py          DatabaseManager - SQLAlchemy ORM (SQLite)
|   +-- calibration.py       CalibrationDashboard - Brier score, bias analysis
|   +-- validators.py        InputValidator - guard conditions
|   +-- utils.py             Helper functions
|
+-- tests/
    +-- __init__.py
    +-- test_core.py          42 unit tests (Poisson, xG, blend, score effects)
    +-- test_kelly.py         20 unit tests (Kelly criterion)
```

---

## Running Tests

```bash
pytest tests/ -v
pytest tests/ --cov=apex_black_box --cov-report=term-missing
```

---

## Requirements

| Package    | Version | Purpose                    |
|------------|---------|----------------------------|
| numpy      | >=1.24  | Numerical arrays, Poisson  |
| scipy      | >=1.10  | Statistical functions       |
| pandas     | >=2.0   | Data manipulation           |
| SQLAlchemy | >=2.0   | ORM and SQLite persistence |
| pydantic   | >=2.0   | Data validation             |
| pytest     | >=7.0   | Testing framework           |
| click      | >=8.1   | CLI interface               |

---

## License

MIT
