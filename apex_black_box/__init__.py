from .steam import SteamAnalyzer
from .calibration import CalibrationDashboard


def initialize():
    # Lightweight initialization; no network calls or side effects
    return {
        "SteamAnalyzer": SteamAnalyzer,
        "CalibrationDashboard": CalibrationDashboard,
    }
