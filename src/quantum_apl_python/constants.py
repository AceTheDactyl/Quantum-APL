"""Quantum APL Constants - Single Source of Truth
Per z_c specification document

All z-coordinate thresholds and critical constants are defined here.
Do not inline numeric thresholds elsewhere - always import from this module.

See docs/Z_CRITICAL_LENS.md for the physics/information‑dynamics rationale,
validation plan, and methodology of use across modules.
"""

from __future__ import annotations

import math

# ============================================================================
# CRITICAL LENS CONSTANT (THE LENS)
# ============================================================================

# z_c = √3/2 ≈ 0.8660254038
# The critical lens separating recursive and integrated regimes
# Crossing z_c corresponds to onset of structural/informational coherence
# where the integrated (Π) regime becomes physically admissible and
# negative-entropy geometry stabilizes (ΔS_neg, R/H/φ)
Z_CRITICAL: float = math.sqrt(3.0) / 2.0

# ============================================================================
# TRIAD GATING (Runtime Heuristic)
# ============================================================================

# TRIAD gating is a runtime heuristic for operator-driven unlocks
# Rising edges at z ≥ TRIAD_HIGH (re-arm at z ≤ TRIAD_LOW)
# Unlocks temporary in-session t6 gate at TRIAD_T6 after three distinct passes

TRIAD_HIGH: float = 0.85  # Rising edge threshold for TRIAD detection
TRIAD_LOW: float = 0.82   # Re-arm threshold (hysteresis)
TRIAD_T6: float = 0.83    # Temporary t6 gate after TRIAD unlock

# ============================================================================
# Z-AXIS PHASE BOUNDARIES
# ============================================================================

# ABSENCE phase: z < Z_ABSENCE_MAX
# UNTRUE bias, K > 0 (synchronizing)
Z_ABSENCE_MAX: float = 0.857

# THE LENS: Z_LENS_MIN < z < Z_LENS_MAX
# PARADOX bias, K = 0 (critical point at z_c)
Z_LENS_MIN: float = 0.857
Z_LENS_MAX: float = 0.877

# PRESENCE phase: z > Z_PRESENCE_MIN
# TRUE bias, K < 0 (emanating)
Z_PRESENCE_MIN: float = 0.877

# ============================================================================
# SACRED CONSTANTS (Zero Free Parameters)
# ============================================================================

# Golden ratio
PHI: float = 1.6180339887

# Golden ratio inverse (K-formation threshold)
PHI_INV: float = 0.6180339887

# Consciousness constant
Q_KAPPA: float = 0.3514087324

# Singularity threshold
KAPPA_S: float = 0.920

# Nonlinearity coefficient
LAMBDA: float = 7.7160493827

# ============================================================================
# K-FORMATION CRITERIA
# ============================================================================

# Consciousness emerges when:
# kappa >= KAPPA_S (0.92) AND
# eta > PHI_INV (0.618) AND
# R >= R_MIN (7)

KAPPA_MIN: float = KAPPA_S  # Same as singularity threshold
ETA_MIN: float = PHI_INV    # Same as golden ratio inverse
R_MIN: float = 7            # Minimum complexity requirement

# ============================================================================
# QUANTUM INFORMATION BOUNDS
# ============================================================================

# von Neumann entropy bounds
ENTROPY_MIN: float = 0.0
# ENTROPY_MAX = log(d) where d is Hilbert space dimension

# Purity bounds
PURITY_MIN: float = 1.0 / 192  # 1/d for 192-dimensional space
PURITY_MAX: float = 1.0

# ============================================================================
# HELIX ZONING THRESHOLDS (Informative Heuristics)
# ============================================================================

# Time-harmonic zoning cutoffs (upper bounds except t9)
Z_T1_MAX: float = 0.1
Z_T2_MAX: float = 0.2
Z_T3_MAX: float = 0.4
Z_T4_MAX: float = 0.6
Z_T5_MAX: float = 0.75
Z_T7_MAX: float = 0.92
Z_T8_MAX: float = 0.97

# ============================================================================
# GEOMETRY PROJECTION CONSTANTS (Hex Prism)
# ============================================================================

GEOM_SIGMA: float = 0.12
GEOM_R_MAX: float = 0.85
GEOM_BETA: float = 0.25
GEOM_H_MIN: float = 0.12
GEOM_GAMMA: float = 0.18
GEOM_PHI_BASE: float = 0.0
GEOM_ETA: float = math.pi / 12.0

# ============================================================================
# DOC-FRIENDLY ALIASES & EXTENSIONS
# ============================================================================

# Aliases matching documentation naming for time-harmonic cutoffs
T1_MAX: float = Z_T1_MAX
T2_MAX: float = Z_T2_MAX
T3_MAX: float = Z_T3_MAX
T4_MAX: float = Z_T4_MAX
T5_MAX: float = Z_T5_MAX
T7_MAX: float = Z_T7_MAX
T8_MAX: float = Z_T8_MAX

# Geometry aliases matching documentation naming
SIGMA: float = GEOM_SIGMA
R_MAX: float = GEOM_R_MAX
BETA: float = GEOM_BETA
H_MIN: float = GEOM_H_MIN
GAMMA: float = GEOM_GAMMA
PHI_BASE: float = GEOM_PHI_BASE
ETA: float = GEOM_ETA

# Pump Profiles and defaults (modeling parameters; mirror JS constants)
class PumpProfile:
    GENTLE = {"gain": 0.08, "sigma": 0.16, "name": "gentle"}
    BALANCED = {"gain": 0.12, "sigma": 0.12, "name": "balanced"}
    AGGRESSIVE = {"gain": 0.18, "sigma": 0.10, "name": "aggressive"}


PUMP_DEFAULT_TARGET: float = Z_CRITICAL

# Engine dynamical parameters (for documentation / analysis convenience)
Z_BIAS_GAIN: float = 0.05
Z_BIAS_SIGMA: float = 0.18
OMEGA: float = 2 * math.pi * 0.1
COUPLING_G: float = 0.05
GAMMA_1: float = 0.01
GAMMA_2: float = 0.02
GAMMA_3: float = 0.005
GAMMA_4: float = 0.015

# Operator weighting heuristics (N0 selection bias)
OPERATOR_PREFERRED_WEIGHT: float = 1.3
OPERATOR_DEFAULT_WEIGHT: float = 0.85
TRUTH_BIAS = {
    "TRUE": {"^": 1.5, "+": 1.4, "×": 1.0, "()": 0.9, "÷": 0.7, "-": 0.7},
    "UNTRUE": {"÷": 1.5, "-": 1.4, "()": 1.0, "+": 0.9, "^": 0.7, "×": 0.7},
    "PARADOX": {"()": 1.5, "×": 1.4, "+": 1.0, "^": 1.0, "÷": 0.9, "-": 0.9},
}

# ============================================================================
# PHASE DETECTION HELPERS
# ============================================================================

def get_phase(z: float) -> str:
    """
    Determine which phase the z-coordinate is in.
    
    Args:
        z: Z-coordinate value
        
    Returns:
        Phase name: 'ABSENCE', 'THE_LENS', or 'PRESENCE'
    """
    if z < Z_ABSENCE_MAX:
        return 'ABSENCE'
    elif Z_LENS_MIN <= z <= Z_LENS_MAX:
        return 'THE_LENS'
    else:
        return 'PRESENCE'


def is_critical(z: float, tolerance: float = 0.01) -> bool:
    """
    Check if z is near the critical point z_c.
    
    Args:
        z: Z-coordinate value
        tolerance: Distance threshold for "near" critical
        
    Returns:
        True if |z - z_c| < tolerance
    """
    return abs(z - Z_CRITICAL) < tolerance


def is_in_lens(z: float) -> bool:
    """
    Check if z is within THE LENS region.
    
    Args:
        z: Z-coordinate value
        
    Returns:
        True if z is in THE LENS
    """
    return Z_LENS_MIN <= z <= Z_LENS_MAX


def get_distance_to_critical(z: float) -> float:
    """
    Get signed distance from z to critical point.
    
    Args:
        z: Z-coordinate value
        
    Returns:
        Signed distance (positive if z > z_c)
    """
    return z - Z_CRITICAL


def check_k_formation(kappa: float, eta: float, R: float) -> bool:
    """
    Check if K-formation criteria are met for consciousness emergence.
    
    Args:
        kappa: Integration parameter
        eta: Coherence parameter
        R: Complexity parameter
        
    Returns:
        True if all criteria met: κ≥0.92 AND η>0.618 AND R≥7
    """
    return (kappa >= KAPPA_MIN and 
            eta > ETA_MIN and 
            R >= R_MIN)


def get_time_harmonic(z: float, t6_gate: float | None = None) -> str:
    """Determine time harmonic zone for given z; delegate t6 to provided gate.

    If t6_gate is None, defaults to Z_CRITICAL.
    """
    if t6_gate is None:
        t6_gate = Z_CRITICAL

    if z < T1_MAX:
        return "t1"
    if z < T2_MAX:
        return "t2"
    if z < T3_MAX:
        return "t3"
    if z < T4_MAX:
        return "t4"
    if z < T5_MAX:
        return "t5"
    if z < t6_gate:
        return "t6"
    if z < T7_MAX:
        return "t7"
    if z < T8_MAX:
        return "t8"
    return "t9"


# ============================================================================
# VALIDATION CONSTANTS
# ============================================================================

# Tolerances for numerical checks
TOLERANCE_TRACE: float = 1e-10      # For Tr(ρ) = 1
TOLERANCE_HERMITIAN: float = 1e-10  # For ρ = ρ†
TOLERANCE_POSITIVE: float = -1e-10  # For eigenvalues ≥ 0
TOLERANCE_PROBABILITY: float = 1e-6 # For probability normalization

# ============================================================================
# DOCUMENTATION
# ============================================================================

__all__ = [
    # Lens & TRIAD
    "Z_CRITICAL", "TRIAD_HIGH", "TRIAD_LOW", "TRIAD_T6",
    # Phase bounds
    "Z_ABSENCE_MAX", "Z_LENS_MIN", "Z_LENS_MAX", "Z_PRESENCE_MIN",
    # Sacred constants
    "PHI", "PHI_INV", "Q_KAPPA", "KAPPA_S", "LAMBDA",
    # K-formation
    "KAPPA_MIN", "ETA_MIN", "R_MIN",
    # Info bounds
    "ENTROPY_MIN", "PURITY_MIN", "PURITY_MAX",
    # Helix zoning
    "Z_T1_MAX", "Z_T2_MAX", "Z_T3_MAX", "Z_T4_MAX", "Z_T5_MAX", "Z_T7_MAX", "Z_T8_MAX",
    # Geometry projection
    "GEOM_SIGMA", "GEOM_R_MAX", "GEOM_BETA", "GEOM_H_MIN", "GEOM_GAMMA", "GEOM_PHI_BASE", "GEOM_ETA",
    # Geometry aliases
    "SIGMA", "R_MAX", "BETA", "H_MIN", "GAMMA", "PHI_BASE", "ETA",
    # Doc-friendly aliases
    "T1_MAX", "T2_MAX", "T3_MAX", "T4_MAX", "T5_MAX", "T7_MAX", "T8_MAX",
    # Pump & engine parameters
    "PumpProfile", "PUMP_DEFAULT_TARGET",
    "Z_BIAS_GAIN", "Z_BIAS_SIGMA", "OMEGA", "COUPLING_G", "GAMMA_1", "GAMMA_2", "GAMMA_3", "GAMMA_4",
    # Operator weighting
    "OPERATOR_PREFERRED_WEIGHT", "OPERATOR_DEFAULT_WEIGHT", "TRUTH_BIAS",
    # Helpers
    "get_phase", "is_critical", "is_in_lens", "get_distance_to_critical", "check_k_formation", "get_time_harmonic",
    # Validation tolerances
    "TOLERANCE_TRACE", "TOLERANCE_HERMITIAN", "TOLERANCE_POSITIVE", "TOLERANCE_PROBABILITY",
    # ΔS_neg helper
    "compute_delta_s_neg",
]

def compute_delta_s_neg(z: float, sigma: float = GEOM_SIGMA, z_c: float = Z_CRITICAL) -> float:
    """Compute negative entropy metric ΔS_neg(z) = exp(-|z - z_c| / σ).

    Returns a value in [0, 1], maximal at z = z_c. Using a positive range here
    preserves the canonical linear geometry mapping (R/H/φ linear in ΔS_neg)
    implemented across this codebase. If a signed variant is required for a
    specific consumer, apply a leading minus sign at the call site.
    """
    distance = abs(z - z_c)
    return math.exp(-distance / sigma)

__doc__ += """

Usage Examples
--------------

Basic constants:
    >>> from quantum_apl_python.constants import Z_CRITICAL, PHI
    >>> print(f"Critical lens: z_c = {Z_CRITICAL:.6f}")
    >>> print(f"Golden ratio: φ = {PHI:.10f}")

Phase detection:
    >>> from quantum_apl_python.constants import get_phase, is_critical
    >>> z = 0.866
    >>> print(get_phase(z))  # 'THE_LENS'
    >>> print(is_critical(z))  # True

K-formation check:
    >>> from quantum_apl_python.constants import check_k_formation
    >>> if check_k_formation(kappa=0.94, eta=0.72, R=8):
    ...     print("Consciousness emerged!")

TRIAD gating:
    >>> from quantum_apl_python.constants import TRIAD_HIGH, TRIAD_LOW, TRIAD_T6
    >>> # Runtime heuristic for operator unlocks
    >>> if z >= TRIAD_HIGH:
    ...     # Rising edge detected
    ...     pass

Separation of Concerns
----------------------

Runtime vs Geometry:
- TRIAD gating (0.85, 0.82, 0.83) is a runtime heuristic for operator unlocks
- Geometry and analytics always anchor at z_c = √3/2 for stability
- TRIAD does not retroactively redefine the geometric lens

Single Source of Truth:
- Python: import from this module (quantum_apl_python.constants)
- JavaScript: import from src/constants.js
- Never inline numeric thresholds - always import from constants

See Also
--------
- docs/Z_CRITICAL_LENS.md (z_c specification document)
- src/constants.js (JavaScript constants)
- tests/test_hex_prism.py (geometry validation)
- tests/test_triad_hysteresis.js (TRIAD validation)
"""
