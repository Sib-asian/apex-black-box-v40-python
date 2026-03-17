"""
Apex Black Box V40 - Oracle Verdict Generator
===============================================
Classifies model probabilities against bookmaker odds, generating
structured pick verdicts with edge, EV, and confidence grading.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from apex_black_box.core import OracleOutput


# Classification thresholds
_GREEN_EDGE: float = 0.03    # >= 3% edge required for GREEN
_GREEN_CONF: float = 60.0    # >= 60 confidence required for GREEN
_WATCH_EDGE: float = 0.00    # >= 0% edge for WATCH
_WATCH_CONF: float = 45.0    # >= 45 confidence for WATCH
_SPEC_EDGE: float = 0.00     # >= 0% edge but low confidence


class OracleVerdictGenerator:
    """Convert oracle output into human-readable pick verdicts.

    Each verdict assigns a pick a classification:
    - **GREEN**: Strong value. Edge ≥ 3% and confidence ≥ 60%.
    - **WATCH**: Marginal value. Edge ≥ 0% and confidence ≥ 45%.
    - **SPEC**: Speculative. Edge ≥ 0% but confidence < 45%.
    - **SKIP**: Negative edge. No value detected.

    Parameters
    ----------
    output: Full OracleOutput from an OracleEngineV40.run() call.
    """

    # Market labels for display
    _MARKET_LABELS: Dict[str, str] = {
        "1": "Home Win (1)",
        "X": "Draw (X)",
        "2": "Away Win (2)",
        "Over25": "Over 2.5 Goals",
        "Under25": "Under 2.5 Goals",
        "BTTS": "Both Teams To Score",
        "DNB_H": "Draw No Bet – Home",
        "DNB_A": "Draw No Bet – Away",
    }

    def __init__(self, output: "OracleOutput") -> None:
        self.output = output

    # ------------------------------------------------------------------
    # Core classification
    # ------------------------------------------------------------------

    def classify_pick(
        self,
        market: str,
        prob: float,
        bookie_odds: float,
    ) -> Dict[str, Any]:
        """Classify a single market pick.

        Parameters
        ----------
        market:      Market key (e.g. '1', 'X', '2', 'Over25', …).
        prob:        Model probability for this outcome.
        bookie_odds: Decimal odds offered by bookmaker.

        Returns
        -------
        Dict[str, Any]:
            - ``market``         Market key.
            - ``label``          Human-readable market name.
            - ``prob``           Model probability (0-1).
            - ``bookie_odds``    Decimal odds.
            - ``implied_prob``   Bookmaker implied probability.
            - ``edge``           Edge = prob - implied_prob.
            - ``ev``             Expected value per unit staked.
            - ``confidence``     Oracle confidence (0-100).
            - ``classification`` One of: GREEN, WATCH, SPEC, SKIP.
        """
        if bookie_odds <= 1.0 or prob <= 0.0 or prob >= 1.0:
            return self._build_pick(market, prob, bookie_odds, 0.0, 0.0, "SKIP")

        implied_prob = 1.0 / bookie_odds
        edge = prob - implied_prob
        ev = prob * (bookie_odds - 1.0) - (1.0 - prob)
        confidence = self.output.confidence

        classification = self._grade(edge, confidence)

        return self._build_pick(market, prob, bookie_odds, edge, ev, classification,
                                implied_prob=implied_prob)

    def _grade(self, edge: float, confidence: float) -> str:
        """Assign classification label based on edge and confidence."""
        if edge >= _GREEN_EDGE and confidence >= _GREEN_CONF:
            return "GREEN"
        if edge >= _WATCH_EDGE and confidence >= _WATCH_CONF:
            return "WATCH"
        if edge >= _SPEC_EDGE:
            return "SPEC"
        return "SKIP"

    def _build_pick(
        self,
        market: str,
        prob: float,
        bookie_odds: float,
        edge: float,
        ev: float,
        classification: str,
        implied_prob: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Build the standardised pick dictionary."""
        if implied_prob is None:
            implied_prob = 1.0 / bookie_odds if bookie_odds > 1.0 else 1.0

        return {
            "market": market,
            "label": self._MARKET_LABELS.get(market, market),
            "prob": round(prob, 4),
            "bookie_odds": round(bookie_odds, 3),
            "implied_prob": round(implied_prob, 4),
            "edge": round(edge, 4),
            "ev": round(ev, 4),
            "confidence": round(self.output.confidence, 2),
            "vix": round(self.output.vix, 2),
            "classification": classification,
        }

    # ------------------------------------------------------------------
    # Batch verdict generation
    # ------------------------------------------------------------------

    def generate_verdicts(
        self,
        bookie_odds: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Generate classified verdicts for every market in ``bookie_odds``.

        Only markets that appear in both the oracle output probabilities and
        the supplied ``bookie_odds`` dict are evaluated.  Verdicts are sorted
        by edge descending (best first).

        Parameters
        ----------
        bookie_odds: Mapping of market key → decimal odds.

        Returns
        -------
        List[Dict[str, Any]]: Sorted list of pick verdicts.
        """
        verdicts: List[Dict[str, Any]] = []

        for market, odds in bookie_odds.items():
            prob = self.output.probs.get(market)
            if prob is None:
                continue
            verdict = self.classify_pick(market, prob, odds)
            verdicts.append(verdict)

        # Sort: GREEN first, then WATCH, then SPEC, then SKIP; within tier by edge desc
        tier_order = {"GREEN": 0, "WATCH": 1, "SPEC": 2, "SKIP": 3}
        verdicts.sort(
            key=lambda v: (tier_order[v["classification"]], -v["edge"])
        )

        return verdicts

    def get_best_pick(
        self,
        bookie_odds: Dict[str, float],
    ) -> Optional[Dict[str, Any]]:
        """Return the highest-value pick available.

        Priority order:
        1. The GREEN pick with the highest edge.
        2. If no GREEN picks, the WATCH pick with the highest edge.
        3. Returns ``None`` if no GREEN or WATCH picks exist.

        Parameters
        ----------
        bookie_odds: Mapping of market key → decimal odds.

        Returns
        -------
        Optional[Dict[str, Any]]: Best pick or None.
        """
        verdicts = self.generate_verdicts(bookie_odds)

        for pick in verdicts:
            if pick["classification"] == "GREEN":
                return pick

        for pick in verdicts:
            if pick["classification"] == "WATCH":
                return pick

        return None

    # ------------------------------------------------------------------
    # Convenience summary
    # ------------------------------------------------------------------

    def summary(self, bookie_odds: Dict[str, float]) -> Dict[str, Any]:
        """Return a full analysis summary including all verdicts and the best pick.

        Parameters
        ----------
        bookie_odds: Bookmaker odds for each market to evaluate.

        Returns
        -------
        Dict[str, Any]:
            - ``verdicts``   Sorted list of all classified picks.
            - ``best_pick``  Top GREEN or WATCH pick (or None).
            - ``alerts``     Oracle alerts list.
            - ``confidence`` Oracle confidence.
            - ``vix``        Oracle VIX.
            - ``lambdas``    Oracle rate parameters.
            - ``counts``     Number of picks per classification.
        """
        verdicts = self.generate_verdicts(bookie_odds)
        best = self.get_best_pick(bookie_odds)

        counts = {"GREEN": 0, "WATCH": 0, "SPEC": 0, "SKIP": 0}
        for v in verdicts:
            counts[v["classification"]] += 1

        return {
            "verdicts": verdicts,
            "best_pick": best,
            "alerts": self.output.alerts,
            "confidence": self.output.confidence,
            "vix": self.output.vix,
            "lambdas": self.output.lambdas,
            "counts": counts,
        }
