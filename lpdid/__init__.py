"""
lpdid: Local Projections Difference-in-Differences for Python

A Python implementation of the LP-DiD estimator from 
Dube, Girardi, Jordà, and Taylor (2023).
"""

from .lpdid import LPDiD, LPDiDResults

# Try to import version from package metadata
try:
    from importlib.metadata import version
    __version__ = version("lpdid")
except:
    __version__ = "0.1.0"

__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = ["LPDiD", "LPDiDResults"]