from .steam import SteamAnalyzer
from .calibration import CalibrationDashboard
from .verdict import generate_verdict, ENHANCED_VERDICT
from .engine import scan


def initialize():
    # Lightweight initialization; no network calls or side effects
    return {
        "SteamAnalyzer": SteamAnalyzer,
        "CalibrationDashboard": CalibrationDashboard,
        "generate_verdict": generate_verdict,
        "scan": scan,
    }
