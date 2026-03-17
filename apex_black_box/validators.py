"""
Apex Black Box V40 - Input Validators
======================================
Plain-Python validation for all primary data classes.
Returns (is_valid, messages) tuples so callers can log or surface errors
without raising exceptions.
"""

from __future__ import annotations

from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from apex_black_box.core import MatchScore, MatchStats, PreMatchData


class InputValidator:
    """Static validation methods for Apex Black Box V40 input data.

    All methods return a ``(is_valid, messages)`` tuple where:
    - ``is_valid`` is ``True`` when no errors were found.
    - ``messages`` contains all error and warning strings collected.
    """

    # Reasonable market-line bounds
    _TOTAL_MIN: float = 0.5
    _TOTAL_MAX: float = 12.0
    _SPREAD_ABS_MAX: float = 10.0

    # ------------------------------------------------------------------
    # Individual validators
    # ------------------------------------------------------------------

    @staticmethod
    def validate_score(score: "MatchScore") -> Tuple[bool, List[str]]:
        """Validate a :class:`MatchScore` instance.

        Checks
        ------
        - Minute in [0, 120].
        - Recovery time >= 0.
        - Goals >= 0 for both teams.
        - Red cards in [0, 2] for each team.
        - last_goal, if set, is <= current minute.

        Parameters
        ----------
        score: The match score to validate.

        Returns
        -------
        Tuple[bool, List[str]]: (is_valid, error_messages).
        """
        errors: List[str] = []

        if not (0 <= score.min <= 120):
            errors.append(
                f"Minute {score.min} is out of range [0, 120]."
            )

        if score.rec < 0:
            errors.append(
                f"Recovery time {score.rec} cannot be negative."
            )

        if score.hg < 0:
            errors.append(f"Home goals {score.hg} cannot be negative.")

        if score.ag < 0:
            errors.append(f"Away goals {score.ag} cannot be negative.")

        if not (0 <= score.red_h <= 2):
            errors.append(
                f"Home red cards {score.red_h} must be in [0, 2]."
            )

        if not (0 <= score.red_a <= 2):
            errors.append(
                f"Away red cards {score.red_a} must be in [0, 2]."
            )

        if score.last_goal != -1:
            if score.last_goal < 0:
                errors.append(
                    f"last_goal {score.last_goal} must be -1 (none) or a valid minute >= 0."
                )
            elif score.last_goal > score.min:
                errors.append(
                    f"last_goal ({score.last_goal}) cannot be after current minute ({score.min})."
                )

        return len(errors) == 0, errors

    @staticmethod
    def validate_stats(stats: "MatchStats") -> Tuple[bool, List[str]]:
        """Validate a :class:`MatchStats` instance.

        Checks
        ------
        - All integer stats (sot, mis, cor, da) are non-negative.
        - Possession values in [0, 100].
        - Possession sum is approximately 100 (±5 tolerance).

        Parameters
        ----------
        stats: The match stats to validate.

        Returns
        -------
        Tuple[bool, List[str]]: (is_valid, error_messages).
        """
        errors: List[str] = []

        int_fields = {
            "sot_h": stats.sot_h,
            "mis_h": stats.mis_h,
            "cor_h": stats.cor_h,
            "da_h": stats.da_h,
            "sot_a": stats.sot_a,
            "mis_a": stats.mis_a,
            "cor_a": stats.cor_a,
            "da_a": stats.da_a,
        }

        for name, val in int_fields.items():
            if val < 0:
                errors.append(f"Stat '{name}' ({val}) cannot be negative.")

        if not (0.0 <= stats.poss_h <= 100.0):
            errors.append(
                f"poss_h ({stats.poss_h}) must be in [0, 100]."
            )

        if not (0.0 <= stats.poss_a <= 100.0):
            errors.append(
                f"poss_a ({stats.poss_a}) must be in [0, 100]."
            )

        poss_sum = stats.poss_h + stats.poss_a
        if abs(poss_sum - 100.0) > 5.0:
            errors.append(
                f"Possession values sum to {poss_sum:.1f}%, expected ~100% (±5)."
            )

        return len(errors) == 0, errors

    @staticmethod
    def validate_pre_match(pre: "PreMatchData") -> Tuple[bool, List[str]]:
        """Validate a :class:`PreMatchData` instance.

        Checks
        ------
        - Total lines in [0.5, 12.0] (reasonable goals O/U range).
        - Spread absolute value in [0, 10].
        - Prior values are positive.
        - No suspiciously large line movements (>3 goals or >3 spread units).

        Parameters
        ----------
        pre: Pre-match data to validate.

        Returns
        -------
        Tuple[bool, List[str]]: (is_valid, error_messages).
        """
        errors: List[str] = []
        tmin = InputValidator._TOTAL_MIN
        tmax = InputValidator._TOTAL_MAX
        smax = InputValidator._SPREAD_ABS_MAX

        if not (tmin <= pre.total_open <= tmax):
            errors.append(
                f"total_open ({pre.total_open}) out of range [{tmin}, {tmax}]."
            )

        if not (tmin <= pre.total_curr <= tmax):
            errors.append(
                f"total_curr ({pre.total_curr}) out of range [{tmin}, {tmax}]."
            )

        if abs(pre.spread_open) > smax:
            errors.append(
                f"spread_open ({pre.spread_open}) exceeds absolute limit ±{smax}."
            )

        if abs(pre.spread_curr) > smax:
            errors.append(
                f"spread_curr ({pre.spread_curr}) exceeds absolute limit ±{smax}."
            )

        if pre.prior_h <= 0.0:
            errors.append(f"prior_h ({pre.prior_h}) must be positive.")

        if pre.prior_a <= 0.0:
            errors.append(f"prior_a ({pre.prior_a}) must be positive.")

        if pre.prior_draw < 0.0:
            errors.append(f"prior_draw ({pre.prior_draw}) cannot be negative.")

        # Warn on large movements (informational, not blocking)
        total_move = abs(pre.total_curr - pre.total_open)
        if total_move > 3.0:
            errors.append(
                f"Large total line movement detected: {pre.total_open} → {pre.total_curr} "
                f"({total_move:.2f} goals). Consider verifying data."
            )

        spread_move = abs(pre.spread_curr - pre.spread_open)
        if spread_move > 3.0:
            errors.append(
                f"Large spread movement detected: {pre.spread_open} → {pre.spread_curr} "
                f"({spread_move:.2f} units). Consider verifying data."
            )

        return len(errors) == 0, errors

    @classmethod
    def validate_all(
        cls,
        pre: "PreMatchData",
        score: "MatchScore",
        stats: "MatchStats",
    ) -> Tuple[bool, List[str]]:
        """Run all three validators and aggregate results.

        Parameters
        ----------
        pre:   Pre-match data.
        score: Current match score.
        stats: Current in-play statistics.

        Returns
        -------
        Tuple[bool, List[str]]:
            (all_valid, combined_error_list) where all_valid is True only
            when every individual validator passes.
        """
        all_messages: List[str] = []

        ok_score, msgs_score = cls.validate_score(score)
        ok_stats, msgs_stats = cls.validate_stats(stats)
        ok_pre, msgs_pre = cls.validate_pre_match(pre)

        all_messages.extend(msgs_score)
        all_messages.extend(msgs_stats)
        all_messages.extend(msgs_pre)

        is_valid = ok_score and ok_stats and ok_pre
        return is_valid, all_messages
