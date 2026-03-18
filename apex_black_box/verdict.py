"""
Apex Black Box – Oracle Verdict Engine (Python port of generateAdvice() in V40.html)

Computes exchange-oriented betting advice from a scan result dict.
Primary entry point: generate_verdict(data, bookie_odds=None) -> dict

The returned dict mirrors the structure produced by the JS generateAdvice()
function so that renderAdvice() in V40.html can consume it directly when
Python is the active engine.

Kill switch: set ENHANCED_VERDICT = False to revert to baseline scoring
behaviour (no exchange-specific market enhancements or next-goal penalties).
"""

from __future__ import annotations

import math
from typing import Any

from .engine import clamp, clamp01

# ── Kill switch ──────────────────────────────────────────────────────────────
# Set to False to revert to baseline verdict behaviour (disables exchange
# enhancements: NoMoreGoals gate, Next Goal late-game penalty, Under/Over 3.5
# context scoring).  Default: True (enhancements active, as requested).
ENHANCED_VERDICT: bool = True


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe(v: Any, fallback: float = 0.0) -> float:
    try:
        r = float(v)
        return r if math.isfinite(r) else fallback
    except (TypeError, ValueError):
        return fallback


def _calc_fair_odd(p: float) -> str:
    """Return the fair odd as a formatted string (like JS calcFairOdd())."""
    if p <= 0:
        return "∞"
    return f"{max(1.01, 1.0 / p):.2f}"


# ─────────────────────────────────────────────────────────────────────────────
#  Main verdict function
# ─────────────────────────────────────────────────────────────────────────────

def generate_verdict(
    data: dict,
    bookie_odds: dict | None = None,
    kelly_frac: float = 0.25,
) -> dict:
    """
    Python port of the JS generateAdvice() function (V40.html).

    Returns a dict with the same structure as the JS function so that the
    existing renderAdvice() JavaScript can consume it transparently when
    data.verdict is present.

    Parameters
    ----------
    data        : scan result dict produced by engine.scan()
    bookie_odds : optional {market_id: odd_float} for edge calculation
    kelly_frac  : fractional Kelly coefficient (default 0.25)
    """
    bk: dict = bookie_odds or data.get("bookieOdds") or {}
    conf: float = _safe(data.get("confidence"), 0.0)
    min_: float = _safe(data.get("min"), 0.0)
    hg: int = int(_safe(data.get("currentHg"), 0))
    ag: int = int(_safe(data.get("currentAg"), 0))
    goals: int = hg + ag

    probs: dict = data.get("probs") or {}
    raw: dict = data.get("raw") or {}

    # ── Confidence thresholds (mirrors JS) ────────────────────────────────────
    conf_r = round(conf)
    low_conf = conf_r < 40
    mid_conf = 40 <= conf_r < 55
    late_game = min_ > 85

    vix_: float = _safe(data.get("vix"), 50.0)
    vix_adj = clamp((vix_ - 50) * 0.001, -0.030, 0.030)
    is_gt = bool(data.get("isGarbageTime", False))
    gt_adj = -0.03 if is_gt else 0.0  # 2h: lower threshold in garbage time
    green_thresh = clamp(0.62 - (conf / 88) * 0.08 + vix_adj + gt_adj, 0.49, 0.66)
    watch_thresh = max(0.48, green_thresh - 0.06)
    spec_thresh = max(0.40, watch_thresh - 0.10)

    # ── Context values ────────────────────────────────────────────────────────
    diff_ = abs(hg - ag)
    rem_min = max(1.0, 90.0 + _safe(raw.get("rec"), 0.0) - min_)
    proj_h = _safe((data.get("projectedFinal") or {}).get("pH"), 0.0)
    proj_a = _safe((data.get("projectedFinal") or {}).get("pA"), 0.0)
    proj_tot = proj_h + proj_a
    rc_h = int(_safe(raw.get("rcH"), 0))
    rc_a = int(_safe(raw.get("rcA"), 0))
    last_goal = int(_safe(raw.get("lastGoal"), 0))

    btts_ = _safe(probs.get("BTTS"), 0.0)
    p_over_dyn = _safe(probs.get("O_Dyn"), 0.0)

    # ── Market helpers ────────────────────────────────────────────────────────
    over15_useful = goals >= 1
    over15_prob = _safe(probs.get("Over15"), 0.0)
    under25_prob = _safe(probs.get("Under25"), 0.0)

    dc1x = _safe(probs.get("DC_1X"), 0.0) or clamp01(
        _safe(probs.get("1"), 0.0) + _safe(probs.get("X"), 0.0)
    )
    dcx2 = _safe(probs.get("DC_X2"), 0.0) or clamp01(
        _safe(probs.get("2"), 0.0) + _safe(probs.get("X"), 0.0)
    )

    p_under35 = clamp01(1.0 - _safe(probs.get("Over35"), 0.0))
    p_over35 = _safe(probs.get("Over35"), 0.0)
    p_no_more = _safe(probs.get("NoMoreGoals"), 0.0)

    ph_c = _safe(probs.get("Next_H_c"), 0.0)
    pa_c = _safe(probs.get("Next_A_c"), 0.0)
    goal_diff_adv = abs(hg - ag)

    # ── Build market list (mirrors JS markets array construction) ─────────────
    markets: list[dict] = [
        {"id": "1",    "name": "1 — Vittoria Casa",  "p": _safe(probs.get("1"),    0.0), "bkKey": "1"},
        {"id": "X",    "name": "X — Pareggio",        "p": _safe(probs.get("X"),    0.0), "bkKey": "X"},
        {"id": "2",    "name": "2 — Vittoria Trasf",  "p": _safe(probs.get("2"),    0.0), "bkKey": "2"},
        {"id": "over", "name": f"Over {data.get('dynOver', 2.5)}",
                                                       "p": _safe(probs.get("O_Dyn"),0.0), "bkKey": "O_Dyn"},
        {"id": "btts", "name": "BTTS Sì",             "p": btts_,                         "bkKey": "BTTS"},
    ]

    if over15_useful:
        markets += [
            {"id": "over15",  "name": "Over 1.5",  "p": over15_prob,  "bkKey": "O15"},
            {"id": "under25", "name": "Under 2.5", "p": under25_prob, "bkKey": "U25"},
        ]
    else:
        markets += [
            {"id": "dc1X", "name": "1X (Dbl Chance)", "p": dc1x, "bkKey": None},
            {"id": "dcX2", "name": "X2 (Dbl Chance)", "p": dcx2, "bkKey": None},
        ]

    # Exchange markets: Under/Over 3.5 always available
    markets.append({"id": "under35", "name": "Under 3.5", "p": p_under35, "bkKey": None})
    markets.append({"id": "over35",  "name": "Over 3.5",  "p": p_over35,  "bkKey": None})

    # No More Goals: operational from 55', conservative conditions checked in scoring block
    if min_ >= 55 and p_no_more > 0.005:
        markets.append({
            "id": "nomore",
            "name": "🛡️ Nessun altro gol (0 gol rimanenti)",
            "p": p_no_more,
            "bkKey": None,
        })

    # Next Goal: only when match is still open (goal diff ≤ 1)
    if goal_diff_adv <= 1 and ph_c > 0.005 and pa_c > 0.005:
        if ph_c >= pa_c:
            markets.append({"id": "nextH", "name": "Prox Gol Casa (cond.)", "p": ph_c, "bkKey": "Next_H_c"})
        else:
            markets.append({"id": "nextA", "name": "Prox Gol Trasf (cond.)", "p": pa_c, "bkKey": "Next_A_c"})

    # Chasing scenario: goal diff ≥ 2 and home team is behind
    if goal_diff_adv >= 2 and hg < ag and ph_c > 0.01:
        markets.append({"id": "nextH_chase", "name": "Prox Gol Casa (rimonta)", "p": ph_c, "bkKey": "Next_H_c"})

    # Half-time over
    ht_probs: dict = data.get("htProbs") or {}
    ht_over = _safe(ht_probs.get("O"), 0.0)
    if ht_over > 0:
        markets.append({"id": "overHT", "name": "Over 0.5 HT", "p": ht_over, "bkKey": None})

    # 4a: AH/Combo markets suggested by steam advice
    steam_d = data.get("steam") or {}
    steam_advice_list = steam_d.get("advice") or []
    _ah_steam_idx = 0
    for adv in steam_advice_list:
        mkt_name = adv.get("market", "")
        if "AH" in mkt_name or "Combo" in mkt_name:
            if "Casa" in mkt_name:
                implied_p = _safe(probs.get("DNB_H"), 0.0)
            elif "Trasferta" in mkt_name:
                implied_p = _safe(probs.get("DNB_A"), 0.0)
            else:
                implied_p = max(_safe(probs.get("1"), 0.0), _safe(probs.get("2"), 0.0))
            if implied_p > 0.30:
                markets.append({
                    "id": f"ah_steam_{_ah_steam_idx}",
                    "name": f"🧭 Pre-Match: {mkt_name}",
                    "p": implied_p,
                    "bkKey": None,
                    "_steam_advice": adv,
                })
                _ah_steam_idx += 1

    # ── Context-intelligent scoring: 13 rules (mirrors JS V40 scoring block) ──
    for m in markets:
        score = 5
        remove = False
        reason = ""
        m_id: str = m["id"]
        m_p: float = m["p"]

        # Rule 1: extremes (very high or very low probability)
        if m_p > 0.90 or m_p < 0.10:
            score -= 3

        # Rule 2: 1X2 with high lead and late match
        if m_id in ("1", "X", "2"):
            if diff_ >= 2 and min_ > 70:
                score -= 4
                reason = "partita quasi chiusa"
            if diff_ >= 3:
                remove = True
                reason = "partita chiusa 3+ gol"
            if m_id == "X" and diff_ >= 2 and min_ > 65:
                remove = True
                reason = "pareggio impossibile"

        # Rule 3: Over/Under coherence with projection
        if m_id == "over":
            if proj_tot < 0.20 and rem_min < 20:
                score -= 3
                reason = "pochi gol attesi"
            if is_gt and p_over_dyn < 0.10:
                score -= 4
            if diff_ == 0 and (hg + ag) <= 1:
                score += 2
            # 4d: penalise Over when steam signal is inverse (market expects fewer goals)
            if steam_d.get("level") == "reverse":
                score -= 2
                reason = "steam inverso: Over controindicato"
        if m_id == "under25":
            if (hg + ag >= 2) and min_ > 70:
                score -= 3
            if vix_ < 30 and (hg + ag) == 0:
                score += 2

        # Rule 4: BTTS
        if m_id == "btts":
            if hg > 0 and ag > 0:
                remove = True
                reason = "BTTS già avvenuto"
            if min_ > 80 and (hg == 0 or ag == 0) and btts_ < 0.10:
                score -= 4
            if hg == 0 and ag == 0 and vix_ > 45 and min_ > 20:
                score += 2

        # Rule 5: Over 1.5
        if m_id == "over15":
            if (hg + ag) >= 2:
                remove = True
                reason = "Over 1.5 già avvenuto"
            if (hg + ag) == 1 and min_ > 60 and m_p > 0.80:
                score += 1

        # Rule 6: Double Chance
        if m_id in ("dc1X", "dcX2", "dc12"):
            if diff_ >= 2:
                score -= 2
            if m_id == "dc12" and hg == 0 and ag == 0 and vix_ > 50:
                score += 1

        # Rule 7: Next Goal — exchange-oriented penalties
        if m_id in ("nextH", "nextH_chase"):
            if rc_h >= 1:
                score -= 2
            if hg < ag and last_goal > 0 and (min_ - last_goal) < 10:
                score += 2
            if vix_ < 25:
                score -= 1
            # EXCHANGE: late-game heavy penalty when home team is winning
            if ENHANCED_VERDICT and min_ >= 60 and hg > ag:
                score -= 4
                reason = "penalità late-game: Casa in vantaggio"
        if m_id == "nextA":
            if rc_a >= 1:
                score -= 2
            if ag < hg and last_goal > 0 and (min_ - last_goal) < 10:
                score += 2
            if vix_ < 25:
                score -= 1
            # EXCHANGE: late-game heavy penalty when away team is winning
            if ENHANCED_VERDICT and min_ >= 60 and ag > hg:
                score -= 4
                reason = "penalità late-game: Trasferta in vantaggio"

        # Rule 8: end of game (< 15' remaining)
        if rem_min <= 15:
            if m_p < 0.30 and m_id not in ("nextH", "nextA", "nextH_chase"):
                score -= 2
            if diff_ >= 1 and m_id in ("1", "2") and m_p > 0.70:
                score += 2

        # Rule 9: blocked match (low VIX)
        if vix_ < 25:
            if m_id in ("over", "btts"):
                score -= 2
            if m_id == "under25":
                score += 1

        # Rule 10: high intensity (high VIX)
        if vix_ > 70:
            if m_id in ("over", "btts", "nextH", "nextA"):
                score += 1
            if m_id == "under25":
                score -= 2

        # Rule 11: red card
        if rc_h >= 1 or rc_a >= 1:
            stronger = "1" if rc_a >= 1 else "2"
            if m_id == stronger:
                score += 2
            if m_id == "dc1X" and rc_a >= 1:
                score += 1
            if m_id == "dcX2" and rc_h >= 1:
                score += 1

        # Rule 12: Under/Over 3.5 (exchange enhanced scoring)
        if ENHANCED_VERDICT:
            if m_id == "under35":
                if (hg + ag) >= 4:
                    remove = True
                    reason = "Under 3.5 già superato"
                elif (hg + ag) == 3 and min_ > 60:
                    score += 1
                if vix_ > 55 and (hg + ag) >= 2:
                    score -= 2
                if vix_ < 35 and (hg + ag) <= 1:
                    score += 2
                if min_ >= 60 and (hg + ag) <= 2:
                    score += 1
            if m_id == "over35":
                if rem_min <= 20 and m_p < 0.25:
                    score -= 3
                if vix_ > 55 and (hg + ag) >= 2:
                    score += 2
                if min_ >= 65 and (hg + ag) <= 1:
                    score -= 2
        else:
            # Baseline: just clip irrelevant extremes
            if m_id == "under35" and (hg + ag) >= 4:
                remove = True
                reason = "Under 3.5 già superato"

        # Rule 13: No More Goals — conservative operational conditions
        if m_id == "nomore":
            if ENHANCED_VERDICT:
                sot_h = int(_safe(raw.get("sotH"), 0))
                sot_a = int(_safe(raw.get("sotA"), 0))
                mis_h = int(_safe(raw.get("misH"), 0))
                mis_a = int(_safe(raw.get("misA"), 0))
                da_h = int(_safe(raw.get("daH"), 0))
                da_a = int(_safe(raw.get("daA"), 0))
                sot_tot = sot_h + sot_a
                da_tot = da_h + da_a
                total_shots = sot_h + mis_h + sot_a + mis_a
                no_goal_recently = last_goal == 0 or (min_ - last_goal) >= 10
                can_operate = (
                    vix_ <= 38
                    and sot_tot <= 5
                    and da_tot <= 46
                    and (rc_h + rc_a == 0)
                    and no_goal_recently
                )
                if not can_operate:
                    remove = True
                    reason = "condizioni non conservative per No More Goals"
                else:
                    score += 2
                    if min_ >= 70 and m_p >= 0.60:
                        score += 2
                    if total_shots > 14:
                        score -= 2
            else:
                # Without enhancements, simply remove the market
                remove = True
                reason = "enhanced verdict disabilitato"

        m["_score"] = score
        m["_remove"] = remove
        m["_reason"] = reason

    # Remove marked markets
    markets = [m for m in markets if not m.get("_remove")]

    # Hard traps (p ≈ 100%)
    hard_traps = [m for m in markets if m["p"] >= 0.9995]

    # BTTS / Over correlation suppression
    p_btts_v = _safe(probs.get("BTTS"), 0.0)
    p_odyn_v = _safe(probs.get("O_Dyn"), 0.0)
    btts_corr_suppressed = abs(p_btts_v - p_odyn_v) < 0.05 and p_btts_v < p_odyn_v
    over_corr_suppressed = abs(p_btts_v - p_odyn_v) < 0.05 and p_odyn_v < p_btts_v

    dnb_h_is_trap = _safe(probs.get("DNB_H"), 0.0) > 0.80
    dnb_a_is_trap = _safe(probs.get("DNB_A"), 0.0) > 0.80

    # Markets that are excluded from being labelled as "traps"
    exclude_from_trap = {"under25", "over15", "under35", "nomore"}

    # Build the analyzed list: add fair odds, bookie odds, edge, Kelly
    analyzed: list[dict] = []
    for m in markets:
        m_p = m["p"]
        if m_p <= 0.005 or m_p >= 0.9995:
            continue
        fair_odd = _calc_fair_odd(m_p)
        b_key = m.get("bkKey")
        b_odd_raw = bk.get(b_key) if b_key else None
        b_odd: float | None = None
        if b_odd_raw:
            try:
                v = float(b_odd_raw)
                b_odd = v if v > 1.0 else None
            except (TypeError, ValueError):
                pass
        edge: float | None = ((m_p * b_odd - 1) * 100) if b_odd is not None else None
        kelly_full: float | None = (
            max(0.0, ((b_odd - 1) * m_p - (1 - m_p)) / (b_odd - 1))
            if b_odd is not None and b_odd > 1
            else None
        )
        kelly: float | None = (
            kelly_full * kelly_frac * clamp(conf, 30.0, 100.0) / 100.0
            if kelly_full is not None
            else None
        )
        analyzed.append({
            **m,
            "fairOdd": fair_odd,
            "bOdd": b_odd,
            "edge": edge,
            "kelly": kelly,
        })

    # Sort by context score (desc), then edge as tiebreaker, then probability (desc)
    analyzed.sort(key=lambda m: (
        -(m.get("_score") or 5),
        -(m.get("edge") or 0) * 0.5,  # 2g: edge as secondary tiebreaker
        -m["p"],
    ))

    def _is_eligible(m: dict) -> bool:
        if m["p"] > 0.80 and m["id"] not in exclude_from_trap:
            return False
        if m["p"] > 0.9995:
            return False
        if m["id"] == "1" and dnb_h_is_trap:
            return False
        if m["id"] == "2" and dnb_a_is_trap:
            return False
        if m["id"] == "btts" and btts_corr_suppressed:
            return False
        if m["id"] == "over" and over_corr_suppressed:
            return False
        return True

    traps = [
        m for m in analyzed
        if m["p"] > 0.80
        and m["id"] not in exclude_from_trap
        and (m.get("edge") is None or m["edge"] < 5)
    ]

    green = [
        m for m in analyzed
        if _is_eligible(m) and (
            (m.get("edge") is not None and m["edge"] >= 3.0)
            or (m.get("edge") is None and m["p"] >= green_thresh and m["p"] <= 0.80)
        )
    ]
    watch = [
        m for m in analyzed
        if _is_eligible(m) and m not in green and (
            (m.get("edge") is not None and m["edge"] > 0 and m["edge"] < 3.0)
            or (m.get("edge") is None and watch_thresh <= m["p"] < green_thresh)
        )
    ]
    spec = [
        m for m in analyzed
        if _is_eligible(m)
        and m not in green
        and m not in watch
        and m.get("edge") is None
        and spec_thresh <= m["p"] < watch_thresh
    ]
    negative = [
        m for m in analyzed
        if m.get("edge") is not None and m["edge"] < 0 and m not in traps
    ]

    valid_candidates = sorted(green + watch, key=lambda m: -m["p"])
    top_pick = valid_candidates[0] if valid_candidates else None

    # FIX 4: Diversify alt_pick — prefer a pick from a different market family than top_pick.
    # Market families to avoid correlated picks (goal/result/AH are largely independent).
    _GOAL_FAMILY = {"over", "over15", "over35", "under25", "under35", "btts"}
    _RESULT_FAMILY = {"1", "X", "2", "dc1X", "dcX2"}
    _AH_FAMILY = {"nextH", "nextA", "nextH_chase"}

    def _market_family(mid: str) -> str:
        if mid in _GOAL_FAMILY:
            return "goal"
        if mid in _RESULT_FAMILY:
            return "result"
        if mid in _AH_FAMILY or mid.startswith("ah_steam_"):
            return "ah"
        return "other"

    top_family = _market_family(top_pick["id"]) if top_pick else None

    alt_pick = None
    if len(valid_candidates) > 1:
        # Try to find a pick from a different family
        for c in valid_candidates[1:]:
            if _market_family(c["id"]) != top_family:
                alt_pick = c
                break
        # If all remaining picks are in the same family, fall back to index [1]
        if alt_pick is None:
            alt_pick = valid_candidates[1]

    alt_family = _market_family(alt_pick["id"]) if alt_pick else None

    # FIX 4: spec_pick should be from a different family than both top and alt
    spec_sorted = sorted(spec, key=lambda m: -m["p"])
    spec_pick = None
    for s in spec_sorted:
        sf = _market_family(s["id"])
        if sf != top_family and sf != alt_family:
            spec_pick = s
            break
    if spec_pick is None and spec_sorted:
        spec_pick = spec_sorted[0]

    return {
        "topPick": top_pick,
        "altPick": alt_pick,
        "specPick": spec_pick,
        "traps": traps,
        "green": green,
        "watch": watch,
        "spec": spec,
        "negative": negative,
        "hardTraps": hard_traps,
        "all": analyzed,
        "LOW_CONF": low_conf,
        "MID_CONF": mid_conf,
        "LATE_GAME": late_game,
        "GREEN_THRESH": green_thresh,
        "bttsCorrSuppressed": btts_corr_suppressed,
        "overCorrSuppressed": over_corr_suppressed,
    }
