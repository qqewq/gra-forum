"""
GRA Core: Ядро системы и метрики пены Φ.
"""

from .gra_core import GRACore, DebatePlan
from .metrics import FoamMetrics, PhiCalculator

__all__ = ["GRACore", "DebatePlan", "FoamMetrics", "PhiCalculator"]
