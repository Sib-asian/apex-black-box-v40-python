"""
Apex Black Box V40 - Utility Functions
========================================
General-purpose helpers used throughout the Apex Black Box V40 package.
"""

from __future__ import annotations

import math
import re
from typing import List, Tuple

import scipy.stats as _stats  # type: ignore[import-untyped]


def format_odd(odd: float) -> str:
    """Format a decimal odds value to 2 decimal places.

    Parameters
    ----------
    odd: Decimal odds value.

    Returns
    -------
    str: e.g. "2.35".
    """
    return f"{odd:.2f}"


def format_probability(prob: float) -> str:
    """Format a probability (0-1) as a percentage string.

    Parameters
    ----------
    prob: Probability in [0, 1].

    Returns
    -------
    str: e.g. "47.30%".
    """
    return f"{prob * 100:.2f}%"


def calculate_edge(prob: float, odd: float) -> float:
    """Compute the bettor's edge.

    edge = model_probability - bookmaker_implied_probability
         = prob - (1 / odd)

    Parameters
    ----------
    prob: Model probability for the outcome.
    odd:  Decimal odds offered by the bookmaker.

    Returns
    -------
    float: Edge (positive = value bet, negative = no value).
    """
    if odd <= 0.0:
        raise ValueError(f"odd must be positive, got {odd}")
    return prob - (1.0 / odd)


def confidence_interval(
    data: List[float],
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Compute a two-sided Student-t confidence interval for a sample mean.

    Parameters
    ----------
    data:       List of sample values (must have >= 2 elements).
    confidence: Desired confidence level (e.g. 0.95 for 95%).

    Returns
    -------
    Tuple[float, float]: (lower_bound, upper_bound).

    Raises
    ------
    ValueError: If data has fewer than 2 elements.
    """
    if len(data) < 2:
        raise ValueError("Need at least 2 data points for a confidence interval.")

    n = len(data)
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / (n - 1)
    se = math.sqrt(variance / n)

    alpha = 1.0 - confidence
    t_crit = _stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)

    margin = t_crit * se
    return (mean - margin, mean + margin)


def poisson_ci(
    count: float,
    confidence_level: float = 0.95,
) -> Tuple[float, float]:
    """Exact Poisson confidence interval using chi-squared quantiles.

    Parameters
    ----------
    count:            Observed count (non-negative).
    confidence_level: Desired confidence level (e.g. 0.95 for 95%).

    Returns
    -------
    Tuple[float, float]: (lower_bound, upper_bound) for the true Poisson rate.

    Notes
    -----
    Uses the Garwood interval:
      lower = chi2.ppf(alpha/2, 2*count) / 2
      upper = chi2.ppf(1 - alpha/2, 2*(count+1)) / 2
    """
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")

    alpha = 1.0 - confidence_level

    if count == 0:
        lower = 0.0
    else:
        lower = _stats.chi2.ppf(alpha / 2.0, df=2 * count) / 2.0

    upper = _stats.chi2.ppf(1.0 - alpha / 2.0, df=2 * (count + 1)) / 2.0

    return (lower, upper)


def parse_result(result: str) -> Tuple[int, int]:
    """Parse a score string such as "2-1" or "0 - 3" into a (home, away) tuple.

    Parameters
    ----------
    result: Score string (e.g. "1-0", "2 - 3", "0:1").

    Returns
    -------
    Tuple[int, int]: (home_goals, away_goals).

    Raises
    ------
    ValueError: If the string cannot be parsed.
    """
    match = re.match(r"^\s*(\d+)\s*[-:]\s*(\d+)\s*$", result.strip())
    if not match:
        raise ValueError(
            f"Cannot parse result string: {result!r}. "
            "Expected format '1-0' or '2:1'."
        )
    home = int(match.group(1))
    away = int(match.group(2))
    return home, away


def decimal_to_american(odd: float) -> str:
    """Convert decimal odds to American (moneyline) odds format.

    Parameters
    ----------
    odd: Decimal odds (must be >= 1.0).

    Returns
    -------
    str: American odds string, e.g. "+150" or "-200".
    """
    if odd < 1.0:
        raise ValueError(f"Decimal odds must be >= 1.0, got {odd}")

    if odd >= 2.0:
        american = (odd - 1.0) * 100.0
        return f"+{int(round(american))}"
    else:
        american = -100.0 / (odd - 1.0)
        return str(int(round(american)))


def implied_probability(odd: float) -> float:
    """Compute raw implied probability from decimal odds.

    No overround (vig) removal is applied.  Use this for individual markets.

    Parameters
    ----------
    odd: Decimal odds (must be > 0).

    Returns
    -------
    float: 1 / odd.
    """
    if odd <= 0.0:
        raise ValueError(f"odd must be positive, got {odd}")
    return 1.0 / odd


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value to the inclusive range [min_val, max_val].

    Parameters
    ----------
    value:   Input value.
    min_val: Lower bound.
    max_val: Upper bound.

    Returns
    -------
    float: value clamped to [min_val, max_val].
    """
    return max(min_val, min(max_val, value))
