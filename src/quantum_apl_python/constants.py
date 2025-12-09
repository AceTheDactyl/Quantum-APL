"""Shared constants for Quantum-APL Python components."""

from __future__ import annotations

import math

# THE CRITICAL POINT (THE LENS): z_c = √3 / 2
Z_CRITICAL: float = math.sqrt(3.0) / 2.0

# TRIAD thresholds (rising-edge hysteresis and t6 unlock)
TRIAD_HIGH: float = 0.85  # rising-edge threshold (>=) to count a completion
TRIAD_LOW: float = 0.82   # falling threshold (<=) to re-arm for next completion
TRIAD_T6: float = 0.83    # t6 gate after three completions (unlocked)

__all__ = ["Z_CRITICAL", "TRIAD_HIGH", "TRIAD_LOW", "TRIAD_T6"]
