"""
Apex Black Box – Oracle Engine (Python port, v40 parity)

Pure-logic module: no I/O, no side-effects, no localStorage access.
Call  scan(payload: dict) -> dict  from the Flask API layer.

Ported faithfully from the JavaScript runOracleEngine() in
static/js/V40.html (V40 with V40-Shrink anti-overconfidence patch).
"""

from __future__ import annotations
import math
import random
import threading
from typing import Any

# ─────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────
MEAN_GOALS_PER_MIN: float = 1.35 / 90
DC_RHO: float = -0.10
RHYTHM_NORM: float = (15 * 0.80 + 15 * 0.90 + 35 * 0.95 + 15 * 1.10 + 10 * 1.30) / 90

# B1: Cumulative multiplier guard bounds (vs post-blend base_lambda)
MULT_GUARD_MAX: float = 1.80  # total modifiers may not exceed ×1.80
MULT_GUARD_MIN: float = 0.35  # total modifiers may not fall below ×0.35

# log-factorial cache (up to 20)
_LFC: list[float] = [0.0]
_LFC_LOCK = threading.Lock()  # F5: thread-safe cache


# ─────────────────────────────────────────────────────────────────
#  MATH HELPERS
# ─────────────────────────────────────────────────────────────────

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def clamp01(v: float) -> float:
    if not math.isfinite(v):
        return 0.0
    return max(0.0, min(1.0, v))


def safe_num(v: Any, fallback: float = 0.0) -> float:
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v)
    return float(fallback)


def renormalize_1x2(p1: float, px: float, p2: float) -> tuple[float, float, float]:
    s = p1 + px + p2
    if not math.isfinite(s) or s <= 0:
        return (1 / 3, 1 / 3, 1 / 3)
    return (p1 / s, px / s, p2 / s)


def shrink_to_base(p: float, base: float, strength: float) -> float:
    return clamp01((1 - strength) * clamp01(p) + strength * base)


def compute_data_quality(d_s: float, t_s: float, st_q: float, conf: float) -> float:
    return clamp01(
        safe_num(d_s) * 0.45
        + safe_num(t_s) * 0.30
        + safe_num(st_q, 0.5) * 0.15
        + safe_num(conf / 88) * 0.10
    )


def sigmoid_weight(minute: float) -> float:
    return 1.0 / (1.0 + math.exp(-0.10 * (minute - 45)))


def rhythm_curve(m: float) -> float:
    if m < 15:
        r = 0.80
    elif m < 30:
        r = 0.90
    elif m < 65:
        r = 0.95
    elif m < 80:
        r = 1.10
    elif m < 90:
        r = 1.30
    else:
        r = 1.10
    return r / RHYTHM_NORM


def log_fact(n: int) -> float:
    n = min(n, 20)
    with _LFC_LOCK:
        while len(_LFC) <= n:
            _LFC.append(math.log(len(_LFC)) + _LFC[-1])
        return _LFC[n]


def poisson_pmf(lam: float, k: int) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(k * math.log(lam) - lam - log_fact(k))


def calc_xg_obs(sot: float, mis: float, da: float, min_played: float = 90.0) -> float:
    """A3: DA contribution weighted by rate per minute (high DA rate = more informative)."""
    total = sot + mis
    da_eff = min(da, 35) + max(0, da - 35) * 0.15
    # Rate-aware DA quality: 8 DA/90' is considered neutral
    da_rate = da / max(min_played, 1.0)
    da_quality = clamp(da_rate / (8.0 / 90.0), 0.5, 1.5)
    da_eff_weighted = da_eff * da_quality
    if total == 0:
        return da_eff_weighted * 0.003
    conv_rate = clamp(sot / total, 0, 1)
    return sot * (0.10 + conv_rate * 0.04) + mis * (0.02 + conv_rate * 0.02) + da_eff_weighted * 0.003


def pressure_factor(cor: float, sot: float, mis: float, min_played: float) -> float:
    """A4: Corner pressure factor, conditional on shot quality.

    When shots are low but corners are high, corners are the main source of
    danger (set pieces) → more aggressive boost.  When shots are already high,
    xg_rate already captures the threat → lighter boost to avoid double-counting.
    """
    cor_rate = cor / max(1.0, min_played / 10.0)
    ts = sot + mis
    ts_rate = ts / max(1.0, min_played) * 90.0  # shots per 90'
    if ts_rate < 6.0:
        # Low shots: corners are primary danger signal
        coeff = 0.022
        cap = 1.15
    elif ts_rate > 14.0:
        # High shots: xg_rate already captures threat
        coeff = 0.008
        cap = 1.06
    else:
        coeff = 0.015
        cap = 1.12
    return clamp(1.0 + cor_rate * coeff, 1.0, cap)


def dc_correction_adaptive(h: int, a: int, l_h: float, l_a: float, min_: float) -> float:
    dc_scale = clamp(1.0 / (1.0 + max(0, l_h * l_a - 1.0) * 0.35), 0.3, 1.0)
    # F3: floor raised from 0.05 to 0.15 — prevents DC correlation from becoming
    # negligibly small late in the game, keeping draw probability realistic.
    time_decay = max(0.15, 1.0 - (min_ or 0) / 90)
    rho = DC_RHO * dc_scale * time_decay
    if h == 0 and a == 0:
        return 1 - l_h * l_a * rho
    if h == 0 and a == 1:
        return 1 + l_h * rho
    if h == 1 and a == 0:
        return 1 + l_a * rho
    if h == 1 and a == 1:
        return 1 - rho
    return 1.0


def build_result_matrix(l_h: float, l_a: float, min_: float) -> list[dict]:
    max_g = max(9, math.ceil(max(l_h, l_a)) + 5)
    matrix: list[dict] = []
    total = 0.0
    for h in range(max_g + 1):
        for a in range(max_g + 1):
            dc = max(0.005, dc_correction_adaptive(h, a, l_h, l_a, min_))
            p = poisson_pmf(l_h, h) * poisson_pmf(l_a, a) * dc
            matrix.append({"h": h, "a": a, "p": p})
            total += p
    if total > 0:
        for cell in matrix:
            cell["p"] /= total
    return matrix


def aggregate_from_matrix(matrix: list[dict], hg_now: int, ag_now: int) -> dict:
    p1 = px = p2 = p_over = p_over25 = p_btts = p_dnb_h = p_dnb_a = p_over15 = 0.0
    for cell in matrix:
        h, a, p = cell["h"], cell["a"], cell["p"]
        fh = hg_now + h
        fa = ag_now + a
        if fh > fa:
            p1 += p
            p_dnb_h += p
        elif fh < fa:
            p2 += p
            p_dnb_a += p
        else:
            px += p
        if (h + a) >= 1:
            p_over += p
        if (fh + fa) > 2.5:
            p_over25 += p
        if (fh + fa) > 1.5:
            p_over15 += p
        if fh > 0 and fa > 0:
            p_btts += p
    dnb = p1 + p2
    return {
        "p1": p1, "pX": px, "p2": p2,
        "pOver": p_over, "pOver25": p_over25, "pOver15": p_over15,
        "pBTTS": p_btts,
        "p1X": p1 + px, "pX2": p2 + px, "p12": p1 + p2,
        "pDNB_H": (p_dnb_h / dnb) if dnb > 0 else 0.0,
        "pDNB_A": (p_dnb_a / dnb) if dnb > 0 else 0.0,
    }


def poisson_sample(lam: float) -> int:
    if lam <= 0:
        return 0
    L = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        k += 1
        p *= random.random()
        if p <= L:
            return k - 1


def simulate_extra_time(l_h: float, l_a: float) -> dict:
    SIMS = 5000
    et_h1 = l_h * 0.78 * (15 / 90)
    et_a1 = l_a * 0.78 * (15 / 90)
    et_h2 = l_h * 0.87 * (15 / 90)
    et_a2 = l_a * 0.87 * (15 / 90)
    res_h = res_a = res_ph = res_pa = 0
    for _ in range(SIMS):
        g_h = poisson_sample(et_h1) + poisson_sample(et_h2)
        g_a = poisson_sample(et_a1) + poisson_sample(et_a2)
        if g_h > g_a:
            res_h += 1
        elif g_a > g_h:
            res_a += 1
        else:
            if random.random() < 0.52:
                res_ph += 1
            else:
                res_pa += 1
    return {
        "etH": res_h / SIMS, "etA": res_a / SIMS,
        "penH": res_ph / SIMS, "penA": res_pa / SIMS,
    }


# ─────────────────────────────────────────────────────────────────
#  STEAM ANALYSIS
# ─────────────────────────────────────────────────────────────────

def analyze_steam(s_o: float, s_c: float, t_o: float, t_c: float) -> dict:
    d_spread = s_c - s_o
    d_total = t_c - t_o
    ad_s = abs(d_spread)
    ad_t = abs(d_total)
    signals: list[str] = []
    level = "none"
    lambda_mod = {"h": 1.0, "a": 1.0}

    has_spread = ad_s >= 0.5
    has_total = ad_t >= 0.25
    is_reverse = has_spread and d_total < -0.25 and d_spread < 0
    is_reverse_a = has_spread and d_total < -0.25 and d_spread > 0

    if has_spread:
        mod = 1.0 + clamp(ad_s * 0.04, 0.02, 0.08)
        if d_spread < 0:
            signals.append(f"Le quote favoriscono la casa: spread passato da {s_o} a {s_c}")
            lambda_mod["h"] *= mod
            lambda_mod["a"] *= (1 / mod)
        else:
            signals.append(f"Le quote favoriscono la trasferta: spread passato da {s_o} a {s_c}")
            lambda_mod["a"] *= mod
            lambda_mod["h"] *= (1 / mod)
        level = "strong" if ad_s >= 1.0 else "weak"

    if has_total:
        mod = 1.0 + clamp(ad_t * 0.06, 0.01, 0.07)
        if d_total > 0:
            signals.append(f"Atteso più gol: total salito da {t_o} a {t_c}")
            lambda_mod["h"] *= mod
            lambda_mod["a"] *= mod
        else:
            signals.append(f"Atteso meno gol: total sceso da {t_o} a {t_c}")
            lambda_mod["h"] /= mod
            lambda_mod["a"] /= mod
        if not has_spread and level == "none":
            level = "weak"
        if ad_t >= 0.5 and level == "weak" and not is_reverse and not is_reverse_a:
            level = "strong"

    if is_reverse or is_reverse_a:
        level = "reverse"
        if is_reverse:
            signals.append("⚠️ STEAM INVERSO: Casa favorita ma total scende")
        if is_reverse_a:
            signals.append("⚠️ STEAM INVERSO: Trasf favorita ma total scende")

    if has_spread and has_total and level == "weak" and not is_reverse and not is_reverse_a:
        if (d_spread < 0 and d_total > 0) or (d_spread > 0 and d_total > 0):
            level = "strong"

    if has_spread and has_total and level == "strong":
        if d_spread < 0 and d_total > 0:
            signals.append("🔥 CONSENSO: Casa favorita + più gol attesi")
            lambda_mod["h"] *= 1.03
        elif d_spread > 0 and d_total > 0:
            signals.append("🔥 CONSENSO: Trasf favorita + più gol attesi")
            lambda_mod["a"] *= 1.03

    return {
        "level": level,
        "signals": signals,
        "lambdaMod": lambda_mod,
        "dSpread": d_spread,
        "dTotal": d_total,
        "adS": ad_s,
        "adT": ad_t,
    }


# ─────────────────────────────────────────────────────────────────
#  MAIN ENGINE
# ─────────────────────────────────────────────────────────────────

def scan(payload: dict) -> dict:
    """
    Pure Oracle Engine computation.

    Expected payload keys (all optional, fall back to 0 / sensible default):
      min, rec, hg, ag, lastGoal,
      sotH, misH, corH, daH,
      sotA, misA, corA, daA,
      rcH, rcA,
      tC (float, default 2.5), sC (float, default 0),
      tO (float, default tC), sO (float, default sC),
      isKnockout (bool, default false),
      possH (int 0-100, default 0 → 50),
      possA (int 0-100, default 0 → 50),
      prevScans (list of previous scan raw dicts, optional, for trend/stabQ)

    Returns a dict mirroring the JS runOracleEngine() return value.
    """

    # ── Read inputs ─────────────────────────────────────────────
    def g(key: str, default: float = 0, as_float: bool = False) -> float:
        v = payload.get(key, default)
        try:
            return float(v) if as_float else int(float(v))
        except (TypeError, ValueError):
            return float(default)

    minute   = g("min")
    rec      = clamp(g("rec"), 0, 15)
    hg       = g("hg")
    ag       = g("ag")
    last_goal = g("lastGoal")
    sot_h = g("sotH"); mis_h = g("misH"); cor_h = g("corH"); da_h = g("daH")
    sot_a = g("sotA"); mis_a = g("misA"); cor_a = g("corA"); da_a = g("daA")
    rc_h  = g("rcH");  rc_a  = g("rcA")
    t_c   = float(payload.get("tC", 2.5) or 2.5)
    s_c   = float(payload.get("sC") or 0)
    t_o   = float(payload.get("tO") or t_c)
    s_o   = float(payload.get("sO") or s_c)
    is_ko = bool(payload.get("isKnockout", False))

    poss_h_raw = g("possH"); poss_a_raw = g("possA")
    poss_h = clamp(poss_h_raw if poss_h_raw > 0 else 50, 20, 80) / 100
    poss_a = clamp(
        poss_a_raw if poss_a_raw > 0 else (100 - poss_h_raw if poss_h_raw > 0 else 50),
        20, 80
    ) / 100

    prev_scans: list[dict] = list(payload.get("prevScans") or [])

    alerts: list[dict] = []
    safe_min = max(1.0, float(minute))
    match_name = payload.get("matchName", "")

    # ── Steam ────────────────────────────────────────────────────
    steam = analyze_steam(s_o, s_c, t_o, t_c)
    if steam["level"] == "strong":
        msg = steam["signals"][0] if steam["signals"] else ""
        alerts.append({"type": "steam", "msg": f"⚡ Movimento quote: {msg}"})
    elif steam["level"] == "reverse":
        inv = next((s for s in steam["signals"] if "INVERSO" in s), steam["signals"][0] if steam["signals"] else "")
        alerts.append({"type": "steam", "msg": f"🔄 Movimento contrario: {inv}"})
    elif steam["level"] == "weak" and steam["signals"]:
        alerts.append({"type": "info", "msg": f"📊 Quote in movimento: {steam['signals'][0]}"})
    for s in steam["signals"][1:]:
        if "INVERSO" in s or "CONSENSO" in s:
            t = "steam" if steam["level"] == "reverse" else "tactical"
            alerts.append({"type": t, "msg": s})

    # ── Prior pre-match ──────────────────────────────────────────
    t_c_adj = t_c / 1.025
    prior_h = max(0.15, (t_c_adj - s_c) / 2) / 90
    prior_a = max(0.15, (t_c_adj + s_c) / 2) / 90

    # ── xG Rate live ─────────────────────────────────────────────
    ts_h = sot_h + mis_h; ts_a = sot_a + mis_a
    # F8: DA weight increased to 0.20 (more informative signal)
    sample_w_h = clamp(((sot_h + mis_h) + da_h * 0.20) / 12, 0, 1)
    sample_w_a = clamp(((sot_a + mis_a) + da_a * 0.20) / 12, 0, 1)
    # A3: calc_xg_obs is now rate-aware (DA weighted by rate/minute)
    xg_rate_h = min((calc_xg_obs(sot_h, mis_h, da_h, safe_min) / safe_min) * sample_w_h + prior_h * (1 - sample_w_h), 3.5 / 90)
    xg_rate_a = min((calc_xg_obs(sot_a, mis_a, da_a, safe_min) / safe_min) * sample_w_a + prior_a * (1 - sample_w_a), 3.5 / 90)
    xg_rate_h *= pressure_factor(cor_h, sot_h, mis_h, safe_min)
    xg_rate_a *= pressure_factor(cor_a, sot_a, mis_a, safe_min)

    # ── Possesso modifier ─────────────────────────────────────────
    conv_rate_h = clamp(sot_h / ts_h, 0.20, 0.80) if ts_h > 0 else 0.50
    conv_rate_a = clamp(sot_a / ts_a, 0.20, 0.80) if ts_a > 0 else 0.50
    poss_mod_h_raw = clamp(1.0 + (conv_rate_h - poss_h) * 0.35, 0.85, 1.15)
    poss_mod_a_raw = clamp(1.0 + (conv_rate_a - poss_a) * 0.35, 0.85, 1.15)
    poss_mod_h = poss_mod_h_raw
    poss_mod_a = poss_mod_a_raw
    if poss_mod_h_raw < 0.92 and poss_h > 0.58:
        alerts.append({"type": "tactical", "msg": f"La casa ha il {round(poss_h*100)}% del pallone ma non tira — possesso senza pericolosità."})
    if poss_mod_a_raw < 0.92 and poss_a > 0.58:
        alerts.append({"type": "tactical", "msg": f"La trasferta ha il {round(poss_a*100)}% del pallone ma non tira — possesso senza pericolosità."})
    if poss_mod_h_raw > 1.06 and poss_h < 0.42:
        alerts.append({"type": "chaos", "msg": "La casa tira molto pur avendo poco possesso — attacco efficace in contropiede."})
    if poss_mod_a_raw > 1.06 and poss_a < 0.42:
        alerts.append({"type": "chaos", "msg": "La trasferta tira molto pur avendo poco possesso — attacco efficace in contropiede."})

    # ── Momentum / Trend from history ────────────────────────────
    pace_h = pace_a = m_samples = 0.0
    trend_boost_h = trend_boost_a = 1.0
    if prev_scans:
        filtered = [h for h in prev_scans if h.get("min", 0) < minute and h.get("min", 0) > 0]
        filtered = filtered[:3]
        exp_weights = [1.0, 0.5, 0.25]
        for i, h in enumerate(filtered):
            ls = h
            d_min = minute - ls.get("min", 0)
            w = exp_weights[i] if i < len(exp_weights) else 0.125
            if 1 < d_min < 20:
                pace_h += clamp(((da_h - float(ls.get("daH", 0))) / d_min) * 90, 0, 60) * w
                pace_a += clamp(((da_a - float(ls.get("daA", 0))) / d_min) * 90, 0, 60) * w
                m_samples += w
        if m_samples > 0:
            pace_h /= m_samples
            pace_a /= m_samples

        if len(filtered) >= 3:
            # B2: consistent fallback — only use xgRateH/xgRateA (blend with prior);
            # do NOT mix in xgObsH/min which is raw-observed and has a different scale
            rates_h = [
                {"m": h.get("min", 0), "r": float(h.get("xgRateH", 0) or 0)}
                for h in filtered
                if float(h.get("xgRateH", 0) or 0) > 0 and h.get("min", 0) > 0
            ]
            rates_a = [
                {"m": h.get("min", 0), "r": float(h.get("xgRateA", 0) or 0)}
                for h in filtered
                if float(h.get("xgRateA", 0) or 0) > 0 and h.get("min", 0) > 0
            ]

            def lin_slope(pts: list[dict]) -> float:
                n = len(pts)
                if n < 2:
                    return 0.0
                sx = sum(p["m"] for p in pts)
                sy = sum(p["r"] for p in pts)
                sxy = sum(p["m"] * p["r"] for p in pts)
                sxx = sum(p["m"] * p["m"] for p in pts)
                d = n * sxx - sx * sx
                return ((n * sxy - sx * sy) / d) if d != 0 else 0.0

            slope_h = lin_slope(rates_h)
            slope_a = lin_slope(rates_a)
            trend_boost_h = clamp(1.0 + slope_h / (MEAN_GOALS_PER_MIN / 45) * 0.10, 0.90, 1.10)
            trend_boost_a = clamp(1.0 + slope_a / (MEAN_GOALS_PER_MIN / 45) * 0.10, 0.90, 1.10)
            if trend_boost_h > 1.06:
                alerts.append({"type": "tactical", "msg": "La casa sta crescendo nel tempo — ritmo offensivo in aumento."})
            if trend_boost_h < 0.94:
                alerts.append({"type": "tactical", "msg": "La casa sta calando nel tempo — ritmo offensivo in diminuzione."})
            if trend_boost_a > 1.06:
                alerts.append({"type": "tactical", "msg": "La trasferta sta crescendo nel tempo — ritmo offensivo in aumento."})
            if trend_boost_a < 0.94:
                alerts.append({"type": "tactical", "msg": "La trasferta sta calando nel tempo — ritmo offensivo in diminuzione."})

    if pace_h > 40:
        alerts.append({"type": "tactical", "msg": "🔥 La casa mantiene un buon ritmo offensivo — continua a creare occasioni"})
    if pace_a > 40:
        alerts.append({"type": "tactical", "msg": "🔥 La trasferta mantiene un buon ritmo offensivo — continua a creare occasioni"})

    # ── Lambda pre-match ──────────────────────────────────────────
    h_xg_pre = max(0.15, (t_c_adj - s_c) / 2) * steam["lambdaMod"]["h"]
    a_xg_pre = max(0.15, (t_c_adj + s_c) / 2) * steam["lambdaMod"]["a"]

    # A2: sigmoid_weight data-quality aware — scale towards sample_w when data is sparse
    # w_effective = sigmoid * clamp(sw/0.15,0,1) + sw * (1 - clamp(sw/0.15,0,1))
    sw_scale_h = clamp(sample_w_h / 0.15, 0.0, 1.0)
    sw_scale_a = clamp(sample_w_a / 0.15, 0.0, 1.0)
    sig_h = sigmoid_weight(minute)
    sig_a = sigmoid_weight(minute)
    w_h = sig_h * sw_scale_h + sample_w_h * (1.0 - sw_scale_h)
    w_a = sig_a * sw_scale_a + sample_w_a * (1.0 - sw_scale_a)
    lambda_h = clamp(h_xg_pre * (1 - w_h) + xg_rate_h * 90 * w_h, 0.10, 3.5)
    lambda_a = clamp(a_xg_pre * (1 - w_a) + xg_rate_a * 90 * w_a, 0.10, 3.5)
    lambda_h = clamp(lambda_h * poss_mod_h, 0.10, 3.5)
    lambda_a = clamp(lambda_a * poss_mod_a, 0.10, 3.5)
    lambda_h = clamp(lambda_h * trend_boost_h, 0.10, 3.5)
    lambda_a = clamp(lambda_a * trend_boost_a, 0.10, 3.5)

    base_lambda_h = lambda_h
    base_lambda_a = lambda_a

    # ── Palle inattive ────────────────────────────────────────────
    cor_rate_h = cor_h / max(2.0, safe_min / 10)
    cor_rate_a = cor_a / max(2.0, safe_min / 10)
    if minute > 30 and (ts_h == 0 and cor_h > 0 or cor_rate_h > 2.0):
        palle_red_h = clamp(0.72 + (1 - 0.72) * max(0, 1 - cor_rate_h / 3), 0.72, 0.92)
        lambda_h *= palle_red_h
        alerts.append({"type": "tactical", "msg": "La casa è pericolosa quasi solo su calci piazzati — poco gioco da azione."})
    if minute > 30 and (ts_a == 0 and cor_a > 0 or cor_rate_a > 2.0):
        palle_red_a = clamp(0.72 + (1 - 0.72) * max(0, 1 - cor_rate_a / 3), 0.72, 0.92)
        lambda_a *= palle_red_a
        alerts.append({"type": "tactical", "msg": "La trasferta è pericolosa quasi solo su calci piazzati — poco gioco da azione."})

    xq_h = xg_rate_h * safe_min
    xq_a = xg_rate_a * safe_min

    da_r_h = da_h / ts_h if ts_h > 0 else 3.0
    da_r_a = da_a / ts_a if ts_a > 0 else 3.0
    pos_q_h = clamp(1.0 - (da_r_h - 2.0) * 0.03, 0.88, 1.08)
    pos_q_a = clamp(1.0 - (da_r_a - 2.0) * 0.03, 0.88, 1.08)
    xg_obs_pure_h = calc_xg_obs(sot_h, mis_h, da_h)
    xg_obs_pure_a = calc_xg_obs(sot_a, mis_a, da_a)

    # ── Karma portiere ────────────────────────────────────────────
    sot_h_per90 = sot_h / safe_min * 90
    sot_a_per90 = sot_a / safe_min * 90
    max_time_karma = 90 + rec
    rem_min_karma = max(1.0, max_time_karma - minute)

    def karma_factor(base: float) -> float:
        return clamp(base + (1 - base) * (1 - rem_min_karma / 90) * 0.30, base, 1.0)

    if sot_h > 0 and hg <= 1 and minute >= 20:
        atten_h = 0.88 if hg == 1 else 1.0
        if sot_h_per90 >= 20:
            lambda_h *= karma_factor(0.60) * atten_h
            alerts.append({"type": "tactical", "msg": "Il portiere della trasferta sta parando tutto — la casa tira molto ma non segna."})
        elif sot_h_per90 >= 13:
            lambda_h *= karma_factor(0.70) * atten_h
            alerts.append({"type": "tactical", "msg": "Il portiere della trasferta sta tenendo bene — la casa tira ma fatica a segnare."})
        elif sot_h_per90 >= 9 and sot_h >= 5:
            lambda_h *= karma_factor(0.80) * atten_h
            alerts.append({"type": "tactical", "msg": "Il portiere della trasferta è attento — buon numero di tiri subiti senza concedere."})
        elif hg == 0:
            conv_h = sot_h / ts_h if ts_h > 0 else 0
            if ts_h >= 10 and conv_h < 0.40:
                lambda_h *= 1.20
                alerts.append({"type": "chaos", "msg": "La casa tira molto ma non segna — potrebbe essere sfortuna o il portiere avversario in grande giornata."})
    elif hg == 0 and minute >= 20:
        conv_h = sot_h / ts_h if ts_h > 0 else 0
        if ts_h >= 10 and conv_h < 0.40:
            lambda_h *= 1.20
            alerts.append({"type": "chaos", "msg": "La casa tira molto ma non segna — potrebbe essere sfortuna o il portiere avversario in grande giornata."})

    if sot_a > 0 and ag <= 1 and minute >= 20:
        atten_a = 0.88 if ag == 1 else 1.0
        if sot_a_per90 >= 20:
            lambda_a *= karma_factor(0.60) * atten_a
            alerts.append({"type": "tactical", "msg": "Il portiere della casa sta parando tutto — la trasferta tira molto ma non segna."})
        elif sot_a_per90 >= 13:
            lambda_a *= karma_factor(0.70) * atten_a
            alerts.append({"type": "tactical", "msg": "Il portiere della casa sta tenendo bene — la trasferta tira ma fatica a segnare."})
        elif sot_a_per90 >= 9 and sot_a >= 5:
            lambda_a *= karma_factor(0.80) * atten_a
            alerts.append({"type": "tactical", "msg": "Il portiere della casa è attento — buon numero di tiri subiti senza concedere."})
        elif ag == 0:
            conv_a = sot_a / ts_a if ts_a > 0 else 0
            if ts_a >= 10 and conv_a < 0.40:
                lambda_a *= 1.20
                alerts.append({"type": "chaos", "msg": "La trasferta tira molto ma non segna — potrebbe essere sfortuna o il portiere avversario in grande giornata."})
    elif ag == 0 and minute >= 20:
        conv_a = sot_a / ts_a if ts_a > 0 else 0
        if ts_a >= 10 and conv_a < 0.40:
            lambda_a *= 1.20
            alerts.append({"type": "chaos", "msg": "La trasferta tira molto ma non segna — potrebbe essere sfortuna o il portiere avversario in grande giornata."})

    # ── Scoreline effects ─────────────────────────────────────────
    goal_diff = hg - ag
    ts_h_r = ts_h / max(1, safe_min) * 90
    ts_a_r = ts_a / max(1, safe_min) * 90
    da_h_r = da_h / max(1, safe_min) * 90
    da_a_r = da_a / max(1, safe_min) * 90
    if hg > ag and minute > 35 and ts_h_r < 7 and da_h_r < 10:
        alerts.append({"type": "tactical", "msg": f"La casa sta gestendo il vantaggio {hg}-{ag} — pochi tiri, difende il risultato."})
    if ag > hg and minute > 35 and ts_a_r < 7 and da_a_r < 10:
        alerts.append({"type": "tactical", "msg": f"La trasferta sta gestendo il vantaggio {ag}-{hg} — pochi tiri, difende il risultato."})

    is_garbage_time = (abs(goal_diff) >= 3) or (abs(goal_diff) >= 2 and minute > 80)

    if is_garbage_time:
        if goal_diff > 0:
            lambda_h *= 0.55; lambda_a *= 0.85
        elif goal_diff < 0:
            lambda_a *= 0.55; lambda_h *= 0.85
        else:
            lambda_h *= 0.65; lambda_a *= 0.65
        alerts.append({"type": "chaos", "msg": "La partita è in garbage time — la squadra in vantaggio non attacca più."})
    else:
        is_post_goal_shock = last_goal > 0 and last_goal < minute and (minute - last_goal) >= 1 and minute < 85
        if is_post_goal_shock:
            delta = minute - last_goal
            shock_mult = 1.0 + 0.32 * math.exp(-0.22 * (delta - 1))
            if goal_diff > 0:
                lambda_h *= max(1.0, shock_mult * 0.90); lambda_a *= shock_mult
            elif goal_diff < 0:
                lambda_a *= max(1.0, shock_mult * 0.90); lambda_h *= shock_mult
            else:
                lambda_h *= shock_mult * 0.92; lambda_a *= shock_mult * 0.92
            if delta <= 4:
                alerts.append({"type": "chaos", "msg": f"Gol appena segnato al {int(last_goal)}' — la squadra che ha subito il gol tende a spingere di più subito dopo."})
            elif delta <= 8:
                alerts.append({"type": "chaos", "msg": f"Gol recente al {int(last_goal)}' — ancora possibile reazione dalla squadra che lo ha subito."})
        else:
            if is_ko:
                ko_boost = min(1.15 + abs(goal_diff) * 0.12, 1.55)
                if goal_diff > 0:
                    lambda_a *= ko_boost
                    alerts.append({"type": "chaos", "msg": f"La trasferta è sotto di {goal_diff} gol — deve attaccare per forza, aspettati più ritmo."})
                elif goal_diff < 0:
                    lambda_h *= ko_boost
                    alerts.append({"type": "chaos", "msg": f"La casa è sotto di {abs(goal_diff)} gol — deve attaccare per forza, aspettati più ritmo."})
            else:
                # B3: chasing_mult scales inversely with goal_diff — large deficit = less realistic comeback
                chasing_mult = clamp(1.10 + 0.12 * (minute / 90) - 0.04 * (abs(goal_diff) - 1), 1.02, 1.22)
                if goal_diff > 0:
                    lambda_a *= chasing_mult
                elif goal_diff < 0:
                    lambda_h *= chasing_mult

        if is_ko and goal_diff == 0 and minute > 60:
            lambda_h *= 1.20; lambda_a *= 1.20
            alerts.append({"type": "chaos", "msg": "Partita KO: è parità — entrambe devono segnare, ritmo più alto."})

    # ── Cartellino rosso ──────────────────────────────────────────
    rd = clamp((90 - minute) / 90, 0.15, 1.0)
    if rc_h > 0:
        rc_boost_h = min(rc_h, 2) * 0.35
        if hg > ag:
            lambda_h *= (1 - min(rc_h, 2) * 0.40 * rd)
            lambda_a *= (1 - min(rc_h, 2) * 0.10 * rd)
            alerts.append({"type": "tactical", "msg": f"La casa ha {min(rc_h, 2)} uomo(i) in meno — si difende, difficile che segni."})
        else:
            lambda_h *= (1 - min(rc_h, 2) * 0.10 * rd)
            lambda_a = clamp(lambda_a * (1 + rc_boost_h * rd), 1.0, 1.65)
            alerts.append({"type": "chaos", "msg": f"La casa ha {min(rc_h, 2)} uomo(i) in meno — la trasferta ha più spazio per attaccare."})
    if rc_a > 0:
        rc_boost_a = min(rc_a, 2) * 0.35
        if ag > hg:
            lambda_a *= (1 - min(rc_a, 2) * 0.40 * rd)
            lambda_h *= (1 - min(rc_a, 2) * 0.10 * rd)
            alerts.append({"type": "tactical", "msg": f"La trasferta ha {min(rc_a, 2)} uomo(i) in meno — si difende, difficile che segni."})
        else:
            lambda_a *= (1 - min(rc_a, 2) * 0.10 * rd)
            lambda_h = clamp(lambda_h * (1 + rc_boost_a * rd), 1.0, 1.65)
            alerts.append({"type": "chaos", "msg": f"La trasferta ha {min(rc_a, 2)} uomo(i) in meno — la casa ha più spazio per attaccare."})

    # ── Soft cap ──────────────────────────────────────────────────
    if base_lambda_h > 0 and lambda_h / base_lambda_h > 1.55:
        lambda_h = base_lambda_h * (1.55 + (lambda_h / base_lambda_h - 1.55) * 0.3)
    if base_lambda_a > 0 and lambda_a / base_lambda_a > 1.55:
        lambda_a = base_lambda_a * (1.55 + (lambda_a / base_lambda_a - 1.55) * 0.3)
    lambda_h = clamp(lambda_h, 0.05, 3.5)
    lambda_a = clamp(lambda_a, 0.05, 3.5)
    lambda_h = clamp(lambda_h * pos_q_h, 0.05, 3.5)
    lambda_a = clamp(lambda_a * pos_q_a, 0.05, 3.5)
    # B1: Final cumulative multiplier guard — prevents cascade of modifiers from
    # exploding (>×MULT_GUARD_MAX) or collapsing (<×MULT_GUARD_MIN) relative to
    # the post-blend base lambda.
    if base_lambda_h > 0:
        mult_total_h = lambda_h / base_lambda_h
        if mult_total_h > MULT_GUARD_MAX:
            lambda_h = base_lambda_h * MULT_GUARD_MAX
        elif mult_total_h < MULT_GUARD_MIN:
            lambda_h = base_lambda_h * MULT_GUARD_MIN
    if base_lambda_a > 0:
        mult_total_a = lambda_a / base_lambda_a
        if mult_total_a > MULT_GUARD_MAX:
            lambda_a = base_lambda_a * MULT_GUARD_MAX
        elif mult_total_a < MULT_GUARD_MIN:
            lambda_a = base_lambda_a * MULT_GUARD_MIN
    lambda_h = clamp(lambda_h, 0.05, 3.5)
    lambda_a = clamp(lambda_a, 0.05, 3.5)

    # ── Proiezione gol rimanenti ──────────────────────────────────
    max_time = 90 + rec
    rem_min = max(1, max_time - minute)
    proj_h = proj_a = 0.0
    for m_off in range(int(rem_min)):
        r = rhythm_curve(minute + m_off)
        proj_h += (lambda_h / 90) * r
        proj_a += (lambda_a / 90) * r
    proj_h = clamp(proj_h, 0, 6.0)
    proj_a = clamp(proj_a, 0, 6.0)

    naked_xg_h = round(hg + proj_h, 2)
    naked_xg_a = round(ag + proj_a, 2)

    # ── Matrix ────────────────────────────────────────────────────
    matrix = build_result_matrix(proj_h, proj_a, minute)
    agg = aggregate_from_matrix(matrix, hg, ag)

    p00_dc = next((c["p"] for c in matrix if c["h"] == 0 and c["a"] == 0), 0.0)
    p_goal = 1 - p00_dc
    total_proj = proj_h + proj_a

    next_h_cond = proj_h / total_proj if total_proj > 0 else 0.0
    next_a_cond = proj_a / total_proj if total_proj > 0 else 0.0
    next_h_abs = next_h_cond * p_goal
    next_a_abs = next_a_cond * p_goal

    # A6: dyn_over is the next commercial half-point above the current score total.
    # Goals are integers, so ceil(hg+ag) = hg+ag, giving hg+ag+0.5 always.
    # This ensures dyn_over ∈ {0.5, 1.5, 2.5, 3.5, 4.5, ...} and P(O_Dyn)
    # is always the probability of at least 1 more goal (a tradable market).
    dyn_over = float(hg + ag) + 0.5
    p_btts = agg["pBTTS"]

    # ── Extra time ───────────────────────────────────────────────
    et_probs = None
    if is_ko and goal_diff == 0 and minute >= 88:
        et_probs = simulate_extra_time(lambda_h, lambda_a)
        alerts.append({"type": "chaos", "msg": "Tempi supplementari simulati — calcolo su 120 minuti totali."})

    # ── Analisi primo tempo ───────────────────────────────────────
    ht_probs = None
    if minute <= 45:
        # C3: dynamic blend — live weight scales with minute (0.10 at kick-off → 0.50 at min≥35)
        ht_live_w = clamp(minute / 35.0, 0.10, 0.50)
        ht_pre_w = 1.0 - ht_live_w
        l_ht_h = clamp(h_xg_pre * ht_pre_w + xg_rate_h * 90 * ht_live_w, 0.10, 3.5)
        l_ht_a = clamp(a_xg_pre * ht_pre_w + xg_rate_a * 90 * ht_live_w, 0.10, 3.5)
        rem_ht = max(1, 45 + rec - minute)
        ht_proj_h = ht_proj_a = 0.0
        for m_off in range(int(rem_ht)):
            r = rhythm_curve(minute + m_off)
            ht_proj_h += (l_ht_h / 90) * r
            ht_proj_a += (l_ht_a / 90) * r
        ht_proj_h = clamp(ht_proj_h, 0, 3.5)
        ht_proj_a = clamp(ht_proj_a, 0, 3.5)
        ht_mat = build_result_matrix(ht_proj_h, ht_proj_a, minute)
        ht_agg = aggregate_from_matrix(ht_mat, hg, ag)
        ht_over = sum(c["p"] for c in ht_mat if (c["h"] + c["a"]) >= 1)
        ht_probs = {"1": ht_agg["p1"], "X": ht_agg["pX"], "2": ht_agg["p2"], "O": ht_over}

    # ── VIX ───────────────────────────────────────────────────────
    vix_raw = clamp((xg_rate_h + xg_rate_a) / 2 / MEAN_GOALS_PER_MIN * 50, 10, 95)
    vix_shot_s_proxy = clamp((ts_h + ts_a) / 26, 0, 1)
    vix_da_s_proxy = clamp((da_h + da_a) / 60, 0, 1)
    vix_data_proxy = vix_shot_s_proxy * 0.80 + vix_da_s_proxy * 0.20
    # C4: vix_conf_scale — cap at 1.20 (not 1.5), upward-only adjustment (never
    # reduces below 1.0) to avoid artificially deflating VIX in real-intensity matches.
    vix_conf_scale = clamp(max(vix_data_proxy, clamp(minute / 75, 0, 1) * 0.5) / 0.6, 1.0, 1.20)
    vix = clamp(vix_raw * vix_conf_scale, 10, 95)
    vix_h = clamp(xg_rate_h / MEAN_GOALS_PER_MIN * 50, 10, 95)
    vix_a = clamp(xg_rate_a / MEAN_GOALS_PER_MIN * 50, 10, 95)
    vix_asym = abs(vix_h - vix_a)

    # ── Metriche analitiche ───────────────────────────────────────
    pace_def_h = clamp((xg_rate_h - prior_h) / max(prior_h, 0.001), -2, 2) if safe_min > 15 else 0.0
    pace_def_a = clamp((xg_rate_a - prior_a) / max(prior_a, 0.001), -2, 2) if safe_min > 15 else 0.0
    pace_def_tot = pace_def_h + pace_def_a
    pace_def_label = "OVER_PACE" if pace_def_tot > 0.3 else ("UNDER_PACE" if pace_def_tot < -0.3 else "ON_PACE")

    l_h_pre_full = max(0.15, (t_c - s_c) / 2)
    l_a_pre_full = max(0.15, (t_c + s_c) / 2)
    elapsed = clamp(minute / 90, 0.05, 1)
    p_score_prematch = poisson_pmf(l_h_pre_full * elapsed, int(hg)) * poisson_pmf(l_a_pre_full * elapsed, int(ag))
    # C1: surprise_idx vs modal result — compare current score to the most probable
    # pre-match scoreline (mode of the Poisson distribution), not to 0-0.
    modal_h = max(0, math.floor(l_h_pre_full * elapsed))
    modal_a = max(0, math.floor(l_a_pre_full * elapsed))
    p_modal = poisson_pmf(l_h_pre_full * elapsed, modal_h) * poisson_pmf(l_a_pre_full * elapsed, modal_a)
    surprise_raw = (
        clamp(-math.log2(p_score_prematch / p_modal), 0, 10)
        if p_score_prematch > 0 and p_modal > 0.001 else 0.0
    )
    surprise_damp = clamp(minute / 25, 0, 1)
    surprise_idx = surprise_raw * surprise_damp
    surprise_label = "ATTESO" if surprise_idx < 1 else ("INATTESO" if surprise_idx < 3 else "SHOCK")

    xg_gap_h = round(xg_obs_pure_h, 2) - hg
    xg_gap_a = round(xg_obs_pure_a, 2) - ag
    xg_gap_label = "SFORTUNA" if (xg_gap_h > 0.8 or xg_gap_a > 0.8) else ("FORTUNA" if (xg_gap_h < -0.5 or xg_gap_a < -0.5) else "ALLINEATO")

    model_total = lambda_h + lambda_a
    tl_eff = clamp(model_total - t_c, -3, 3)
    tl_eff_label = "MODELLO_OVER" if tl_eff > 0.35 else ("MODELLO_UNDER" if tl_eff < -0.35 else "ALLINEATO")

    if safe_min > 20:
        steam_d_total = float(steam["dTotal"] or 0)
        if steam_d_total > 0.4 and pace_def_tot < -0.4:
            alerts.append({"type": "info", "msg": "Attenzione: il mercato si aspetta più gol ma la partita è lenta — situazione contraddittoria."})
        if steam_d_total < -0.4 and pace_def_tot > 0.4:
            alerts.append({"type": "info", "msg": "Attenzione: il mercato si aspetta meno gol ma la partita è intensa — il bookie sa qualcosa?"})
        if abs(tl_eff) > 0.8 and steam["level"] == "none":
            alerts.append({"type": "info", "msg": "Il modello e il bookie non concordano sul totale gol — possibile valore su Over o Under."})

    # ── Confidence ────────────────────────────────────────────────
    shot_s = clamp((ts_h + ts_a) / 26, 0, 1)
    da_s = clamp((da_h + da_a) / 60, 0, 1)
    data_s = shot_s * 0.80 + da_s * 0.20
    time_s = clamp(minute / 75, 0, 1)

    stab_q = 0.50
    # A1: stab_q improved — use up to 5 scans and 4 markets; apply rolling variance
    # to measure systematic instability; attenuate penalty when a goal was scored
    # between scans (justified variance) to avoid false instability penalties.
    _STAB_MARKETS = ("1", "X", "2", "Over25", "O_Dyn")
    if len(prev_scans) >= 2:
        scan_probs = []
        for sc in prev_scans[:5]:  # up to 5 most-recent scans
            if isinstance(sc, dict):
                p = sc.get("probs")
                if isinstance(p, dict):
                    scan_probs.append(p)
        if len(scan_probs) >= 2:
            # Detect goal spike between scans (score change justifies variance)
            def _is_goal_spike(s_a: dict, s_b: dict) -> bool:
                sc_a = s_a.get("score", "0-0"); sc_b = s_b.get("score", "0-0")
                try:
                    home_a, away_a = (int(x) for x in sc_a.split("-"))
                    home_b, away_b = (int(x) for x in sc_b.split("-"))
                    return (home_a + away_a) != (home_b + away_b)
                except Exception:
                    return False
            # Collect per-market variances across consecutive scan pairs
            market_variances: list[float] = []
            for mk in _STAB_MARKETS:
                vals = [float(p.get(mk, 0) or 0) for p in scan_probs]
                if len(vals) < 2:
                    continue
                mean_v = sum(vals) / len(vals)
                var_v = sum((v - mean_v) ** 2 for v in vals) / len(vals)
                market_variances.append(var_v)
            if market_variances:
                avg_var = sum(market_variances) / len(market_variances)
                # Penalise only systematic instability (variance > 0.002 threshold)
                instability = clamp((avg_var - 0.002) / 0.03, 0.0, 1.0)
                # Attenuate penalty if a goal was scored between any consecutive scans
                goal_spike = any(
                    _is_goal_spike(prev_scans[i], prev_scans[i + 1])
                    for i in range(min(len(prev_scans) - 1, 4))
                    if isinstance(prev_scans[i], dict) and isinstance(prev_scans[i + 1], dict)
                )
                if goal_spike:
                    instability *= 0.50  # halve penalty when variance is goal-driven
                stab_q = clamp(1.0 - instability * 0.70, 0.20, 0.90)

    has_data = data_s > 0.05
    confidence = min(
        (data_s * 0.50 + stab_q * 0.20 + time_s * 0.30) * 100 if has_data else time_s * 25,
        88.0,
    )

    # ── Anti-overconfidence shrink ────────────────────────────────
    dq = compute_data_quality(data_s, time_s, stab_q, confidence)
    ss = 0.35 * (1 - dq)
    p1, px, p2 = renormalize_1x2(
        shrink_to_base(safe_num(agg["p1"], 1 / 3), 1 / 3, ss),
        shrink_to_base(safe_num(agg["pX"], 1 / 3), 1 / 3, ss),
        shrink_to_base(safe_num(agg["p2"], 1 / 3), 1 / 3, ss),
    )
    # A5: market-specific neutral bases for shrink.
    # Over 1.5 ≈ 0.72, Over 2.5 ≈ 0.52, BTTS ≈ 0.51 (not 0.50 flat for all).
    # Using 0.50 for Over 1.5 was a systematic anti-over bias when data is sparse.
    p_over25 = shrink_to_base(safe_num(agg["pOver25"], 0.52), 0.52, ss)
    p_over15 = shrink_to_base(safe_num(agg["pOver15"], 0.72), 0.72, ss)
    p_over   = shrink_to_base(safe_num(agg["pOver"],   0.52), 0.52, ss)
    p_btts   = shrink_to_base(safe_num(p_btts,         0.51), 0.51, ss)
    n_hc = shrink_to_base(safe_num(next_h_cond, 0.5), 0.5, ss)
    n_ac = shrink_to_base(safe_num(next_a_cond, 0.5), 0.5, ss)
    dc_1x = clamp01(p1 + px)
    dc_x2 = clamp01(px + p2)
    dc_12  = clamp01(p1 + p2)
    sum12  = p1 + p2
    dnb_h  = clamp01(p1 / sum12) if sum12 > 0 else 0.5
    dnb_a  = clamp01(p2 / sum12) if sum12 > 0 else 0.5

    # ── Confidence intervals (C2: Wilson CI, asymmetric, correct at extremes) ───
    unc = max(0.01, (1 - confidence / 100) * 0.45)
    # n_eff: effective sample size derived from confidence (e.g., conf=50 → n_eff=25)
    n_eff = max(5.0, confidence * 0.5)

    def ci(p: float) -> dict:
        """Wilson score interval (95%) — handles p near 0 or 1 correctly."""
        _z = 1.96
        _denom = 1.0 + _z * _z / n_eff
        _center = (p + _z * _z / (2.0 * n_eff)) / _denom
        _half = (_z / _denom) * math.sqrt(
            max(0.0, p * (1.0 - p) / n_eff + _z * _z / (4.0 * n_eff * n_eff))
        )
        return {
            "lo": round(max(0.01, _center - _half), 3),
            "hi": round(min(0.99, _center + _half), 3),
        }

    # ── Projected final ───────────────────────────────────────────
    rem_m = max(1, 90 + rec - minute)
    p_fin_h = p_fin_a = 0.0
    norm = (15 * 0.80 + 15 * 0.90 + 35 * 0.95 + 15 * 1.10 + 10 * 1.30) / 90
    for m_off in range(int(rem_m)):
        mm = minute + m_off
        if mm < 15:    r = 0.80
        elif mm < 30:  r = 0.90
        elif mm < 65:  r = 0.95
        elif mm < 80:  r = 1.10
        elif mm < 90:  r = 1.30
        else:          r = 1.10
        r /= norm
        p_fin_h += lambda_h / 90 * r
        p_fin_a += lambda_a / 90 * r
    gt_factor = 0.90 if is_garbage_time else 1.0
    p_fin_h *= gt_factor; p_fin_a *= gt_factor
    projected_final = {
        "h": round(hg + p_fin_h, 1),
        "a": round(ag + p_fin_a, 1),
        "pH": round(p_fin_h, 2),
        "pA": round(p_fin_a, 2),
    }

    # ── D3: Model edge vs book fair odds ─────────────────────────
    # Optional: if payload contains bookOdds dict and vig, compute model_edge per market.
    # Non-breaking: omitted from output when bookOdds not provided.
    book_odds_raw = payload.get("bookOdds") or {}
    vig_pct = float(payload.get("vig", 0.05) or 0.05)
    _market_probs = {
        "1": p1, "X": px, "2": p2,
        "Over25": p_over25, "Over15": p_over15, "BTTS": p_btts,
        "O_Dyn": p_over,
    }
    model_edge: dict = {}
    if book_odds_raw:
        for mkt, q_raw in book_odds_raw.items():
            try:
                q = float(q_raw)
                if q > 1.0 and mkt in _market_probs:
                    # Fair probability removes vig from raw implied
                    p_book_fair = (1.0 / q) / (1.0 + vig_pct)
                    p_model = _market_probs[mkt]
                    model_edge[mkt] = round(p_model - p_book_fair, 4)
            except (TypeError, ValueError):
                pass

    # ── Result ────────────────────────────────────────────────────
    return {
        "probs": {
            "1": p1, "X": px, "2": p2,
            "DNB_H": dnb_h, "DNB_A": dnb_a,
            "Over15": p_over15, "Under25": 1 - p_over25,
            "DC_1X": dc_1x, "DC_X2": dc_x2, "DC_12": dc_12,
            "Next_H": next_h_abs, "Next_A": next_a_abs,
            "Next_H_c": n_hc, "Next_A_c": n_ac,
            "O_Dyn": p_over, "Over25": p_over25, "BTTS": p_btts,
        },
        "matrix": matrix,
        "htProbs": ht_probs,
        "etProbs": et_probs,
        "dynOver": dyn_over,
        "vix": round(vix, 1),
        "confidence": round(confidence, 1),
        "alerts": alerts,
        "min": int(minute),
        "confIntervals": {"p1": ci(p1), "pX": ci(px), "p2": ci(p2), "pO": ci(p_over)},
        "score": f"{int(hg)}-{int(ag)}",
        "currentHg": int(hg),
        "currentAg": int(ag),
        "isGarbageTime": is_garbage_time,
        "projectedFinal": projected_final,
        "steam": {
            "level": steam["level"],
            "dSpread": round(steam["dSpread"], 2),
            "dTotal": round(steam["dTotal"], 2),
            "signals": steam["signals"],
        },
        "vixDetail": {
            "H": str(round(vix_h)),
            "A": str(round(vix_a)),
            "asym": str(round(vix_asym)),
        },
        "xG": {"h": str(naked_xg_h), "a": str(naked_xg_a)},
        "metrics": {
            "paceDefH": round(pace_def_h, 2),
            "paceDefA": round(pace_def_a, 2),
            "paceDefLabel": pace_def_label,
            "surpriseIdx": round(surprise_idx, 2),
            "surpriseLabel": surprise_label,
            "xgGapH": round(xg_gap_h, 2),
            "xgGapA": round(xg_gap_a, 2),
            "xgGapLabel": xg_gap_label,
            "tlEff": round(tl_eff, 2),
            "tlEffLabel": tl_eff_label,
            "modelTotal": round(model_total, 2),
            "lH_live": round(lambda_h, 3),
            "lA_live": round(lambda_a, 3),
            "projH": round(proj_h, 3),
            "projA": round(proj_a, 3),
            "modelEdge": model_edge if model_edge else None,
        },
        "raw": {
            "min": int(minute), "hg": int(hg), "ag": int(ag),
            "sotH": sot_h, "sotA": sot_a, "misH": mis_h, "misA": mis_a,
            "corH": cor_h, "corA": cor_a, "daH": da_h, "daA": da_a,
            "possH": poss_h, "possA": poss_a, "lastGoal": last_goal,
            "xgObsH": round(xg_obs_pure_h, 3), "xgObsA": round(xg_obs_pure_a, 3),
            "xgRateH": round(xg_rate_h, 6), "xgRateA": round(xg_rate_a, 6),
            "xgQualH": round(xg_obs_pure_h, 2), "xgQualA": round(xg_obs_pure_a, 2),
            "halfPeriod": "PT" if minute <= 45 else "ST",
        },
    }
