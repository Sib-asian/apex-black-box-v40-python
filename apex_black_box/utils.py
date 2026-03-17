"""
Apex Black Box — utility functions.

Pure, stateless helpers for probability formatting, statistical intervals,
edge calculation, and Kelly criterion.  No I/O or side effects.
"""

from __future__ import annotations
import math
from typing import Optional


# ─────────────────────────────────────────────────────────────────
#  Formatting helpers
# ─────────────────────────────────────────────────────────────────

def format_odd(odd: float, decimals: int = 2) -> str:
    """Format a decimal odd to a fixed number of decimal places."""
    try:
        return f"{float(odd):.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def format_probability(probability: float, as_pct: bool = False) -> str:
    """Format a probability as a decimal or percentage string."""
    try:
        p = float(probability)
        if as_pct:
            return f"{p * 100:.1f}%"
        return f"{p:.3f}"
    except (TypeError, ValueError):
        return "N/A"


def prob_to_fair_odd(p: float) -> float:
    """Convert a probability to its fair (vig-free) decimal odd."""
    if p <= 0.0:
        return float("inf")
    return round(1.0 / p, 3)


def parse_result(result: str) -> tuple[int, int]:
    """Parse a 'H-A' score string into (home_goals, away_goals).

    Returns (0, 0) if the string cannot be parsed.
    """
    try:
        h, a = result.strip().split("-")
        return int(h), int(a)
    except Exception:
        return 0, 0


# ─────────────────────────────────────────────────────────────────
#  Statistical intervals
# ─────────────────────────────────────────────────────────────────

def wilson_ci(p: float, n_eff: float, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval.

    More accurate than the normal approximation, especially near p=0 or p=1.

    Args:
        p:     Observed probability in [0, 1].
        n_eff: Effective sample size (≥ 1).
        z:     z-score for the desired confidence level (default 1.96 → 95%).

    Returns:
        (lower_bound, upper_bound) clamped to [0.01, 0.99].
    """
    p = max(0.0, min(1.0, p))
    n_eff = max(1.0, n_eff)
    denom = 1.0 + z * z / n_eff
    center = (p + z * z / (2.0 * n_eff)) / denom
    half = (z / denom) * math.sqrt(
        max(0.0, p * (1.0 - p) / n_eff + z * z / (4.0 * n_eff * n_eff))
    )
    return (max(0.01, center - half), min(0.99, center + half))


def confidence_interval(p: float, n_eff: float = 30.0) -> dict:
    """Return a {'lo': ..., 'hi': ...} Wilson CI dict, rounded to 3 d.p."""
    lo, hi = wilson_ci(p, n_eff)
    return {"lo": round(lo, 3), "hi": round(hi, 3)}


def poisson_ci(count: float, confidence_level: float = 0.95) -> tuple[float, float]:
    """Approximate Poisson confidence interval for a count observation.

    Uses the normal approximation (valid for count ≥ 5).
    For small counts the interval is wider than exact methods but sufficient
    for display purposes.

    Args:
        count:            Observed count (≥ 0).
        confidence_level: Desired coverage (default 0.95).

    Returns:
        (lower_bound, upper_bound) both ≥ 0.
    """
    count = max(0.0, count)
    z = _z_for_confidence(confidence_level)
    half = z * math.sqrt(max(count, 1.0))
    return (max(0.0, count - half), count + half)


def _z_for_confidence(level: float) -> float:
    """Return the one-sided z-score for common confidence levels."""
    _table = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    return _table.get(level, 1.960)


# ─────────────────────────────────────────────────────────────────
#  Edge & Kelly
# ─────────────────────────────────────────────────────────────────

def calculate_edge(p_model: float, odd: float, vig: float = 0.05) -> float:
    """Calculate the model edge vs a bookmaker odd.

    Args:
        p_model: Model probability in [0, 1].
        odd:     Bookmaker decimal odd (> 1.0).
        vig:     Estimated bookmaker margin (default 5%).

    Returns:
        edge = p_model − p_book_fair.  Positive means value bet.
    """
    try:
        q = float(odd)
        if q <= 1.0:
            return 0.0
        p_book_fair = (1.0 / q) / (1.0 + vig)
        return round(float(p_model) - p_book_fair, 4)
    except (TypeError, ValueError):
        return 0.0


def kelly_fraction(
    p: float,
    odd: float,
    fraction: float = 0.25,
    max_kelly: float = 0.10,
) -> float:
    """Full Kelly criterion, optionally fractionalised.

    Args:
        p:         Model win probability.
        odd:       Decimal odd (> 1.0).
        fraction:  Kelly fraction (default 0.25 = quarter Kelly).
        max_kelly: Cap on the returned fraction (default 10% of bankroll).

    Returns:
        Recommended bankroll fraction in [0, max_kelly].  Returns 0 for no-bet.
    """
    try:
        b = float(odd) - 1.0  # net decimal profit per unit
        q = 1.0 - float(p)
        if b <= 0 or float(p) <= 0:
            return 0.0
        full_kelly = (b * float(p) - q) / b
        if full_kelly <= 0:
            return 0.0
        return min(full_kelly * fraction, max_kelly)
    except (TypeError, ValueError):
        return 0.0

