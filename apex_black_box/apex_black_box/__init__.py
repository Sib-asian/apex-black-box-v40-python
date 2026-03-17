from .steam import SteamAnalyzer
from .calibration import CalibrationDashboard

def initialize():
    # init leggero, senza side effects
    return {
        "SteamAnalyzer": SteamAnalyzer,
        "CalibrationDashboard": CalibrationDashboard,
    }
