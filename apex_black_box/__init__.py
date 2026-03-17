"""
Apex Black Box V40
==================
In-play football probability oracle with staking, calibration, and
persistence utilities.

Version: 4.0.0
"""

from __future__ import annotations

__version__ = "4.0.0"
__author__ = "Apex Black Box"
__all__ = [
    # Core engine and data classes
    "OracleEngineV40",
    "MatchScore",
    "MatchStats",
    "PreMatchData",
    "OracleOutput",
    # Steam analyser
    "SteamAnalyzer",
    # Verdict generator
    "OracleVerdictGenerator",
    # Kelly staking
    "KellyCriterion",
    # Persistence
    "DatabaseManager",
    # Calibration
    "CalibrationDashboard",
    # Input validation
    "InputValidator",
]

from apex_black_box.core import (
    MatchScore,
    MatchStats,
    OracleEngineV40,
    OracleOutput,
    PreMatchData,
)
from apex_black_box.steam import SteamAnalyzer
from apex_black_box.oracle import OracleVerdictGenerator
from apex_black_box.kelly import KellyCriterion
from apex_black_box.database import DatabaseManager
from apex_black_box.calibration import CalibrationDashboard
from apex_black_box.validators import InputValidator
