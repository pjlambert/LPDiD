"""
LP-DiD: Local Projections Difference-in-Differences for Python
"""

from .lpdid import LPDiD, LPDiDPois, LPDiDResults
from .utils import generate_panel_data

__version__ = "0.7.0"
__all__ = ["LPDiD", "LPDiDPois", "LPDiDResults", "generate_panel_data"]
