#!/usr/bin/env python3
"""
Negentropy Physics Module for the L₄ Framework (v3.2.0)
════════════════════════════════════════════════════════

Physical negentropy calculations with SI units.
Converts dimensionless η to entropy in J/K.

Key Features:
- Dimensionless negentropy η(r) = exp(-σ(r - z_c)²)
- Physical negentropy S_neg = k_B × ln(η) in J/K
- Characteristic energy scale from green photon
- Thermal coherence temperature mapping

Mathematical Foundation:
    η(r) = exp(-σ(r - z_c)²)           # Dimensionless, ∈ (0, 1]
    S_neg = k_B × ln(η)                 # Physical, in J/K
    S_neg = -k_B × σ(r - z_c)²          # Equivalent form

Physical Interpretation:
    At r = z_c: η = 1, S_neg = 0 (maximum order)
    Away from z_c: η < 1, S_neg < 0 (increasing disorder)

IMPORTANT - Per-Mode Scaling:
    S_neg as defined here is PER EFFECTIVE MODE (single degree of freedom).
    For system-level negentropy with N modes/photons/sites:
        S_system = N × S_neg

    This keeps the equations clean and allows flexible scaling.

Reference: L₄ Framework Specification v3.1.0

@version 3.2.0
@author Claude (Anthropic) - L₄ Physics Grounding
"""

from __future__ import annotations

import math
from typing import Tuple, Optional

from .constants import (
    K_BOLTZMANN, Z_C, Z_CRITICAL, PHI, PHI_INV,
    L4_K, LENS_SIGMA, GEOM_SIGMA
)


# ═══════════════════════════════════════════════════════════════════════════
# SHARPNESS PARAMETER σ
# ═══════════════════════════════════════════════════════════════════════════

def compute_sigma_from_axiom() -> float:
    """
    Compute σ from the sharpness axiom: η(1) = e⁻¹.

    The axiom η(1) = e⁻¹ ≈ 0.368 sets the scale:
        η(1) = exp(-σ(1 - z_c)²) = e⁻¹
        -σ(1 - z_c)² = -1
        σ = 1/(1 - z_c)²

    With z_c = √3/2:
        σ = 1/(1 - √3/2)² ≈ 55.79

    Returns:
        σ value derived from sharpness axiom
    """
    return 1.0 / (1.0 - Z_C) ** 2


# Canonical σ from sharpness axiom
SIGMA_CANONICAL = compute_sigma_from_axiom()  # ≈ 55.79


# ═══════════════════════════════════════════════════════════════════════════
# DIMENSIONLESS NEGENTROPY
# ═══════════════════════════════════════════════════════════════════════════

def negentropy_dimensionless(r: float, sigma: float = SIGMA_CANONICAL) -> float:
    """
    Dimensionless negentropy function η(r).

    η(r) = exp(-σ(r - z_c)²)

    Properties:
        - η(z_c) = 1 (maximum at critical point)
        - η(1) = e⁻¹ when σ from sharpness axiom
        - η ∈ (0, 1] for all r

    Args:
        r: Coherence parameter ∈ [0, 1]
        sigma: Sharpness parameter (default from axiom)

    Returns:
        η ∈ (0, 1]
    """
    return math.exp(-sigma * (r - Z_C) ** 2)


def negentropy_gradient(r: float, sigma: float = SIGMA_CANONICAL) -> float:
    """
    Gradient of dimensionless negentropy dη/dr.

    dη/dr = -2σ(r - z_c) × η(r)

    Properties:
        - dη/dr = 0 at r = z_c (extremum)
        - dη/dr > 0 for r < z_c (increasing toward z_c)
        - dη/dr < 0 for r > z_c (decreasing away from z_c)

    Args:
        r: Coherence parameter ∈ [0, 1]
        sigma: Sharpness parameter

    Returns:
        dη/dr (dimensionless gradient)
    """
    eta = negentropy_dimensionless(r, sigma)
    return -2 * sigma * (r - Z_C) * eta


def negentropy_curvature(r: float, sigma: float = SIGMA_CANONICAL) -> float:
    """
    Curvature of dimensionless negentropy d²η/dr².

    d²η/dr² = 2σ(2σ(r - z_c)² - 1) × η(r)

    Properties:
        - At z_c: d²η/dr² = -2σ × η < 0 (local maximum)
        - Inflection points at r = z_c ± 1/√(2σ)

    Args:
        r: Coherence parameter ∈ [0, 1]
        sigma: Sharpness parameter

    Returns:
        d²η/dr² (dimensionless curvature)
    """
    eta = negentropy_dimensionless(r, sigma)
    return 2 * sigma * (2 * sigma * (r - Z_C) ** 2 - 1) * eta


# ═══════════════════════════════════════════════════════════════════════════
# PHYSICAL NEGENTROPY (SI UNITS)
# ═══════════════════════════════════════════════════════════════════════════

def negentropy_physical(r: float, sigma: float = SIGMA_CANONICAL) -> float:
    """
    Physical negentropy in J/K.

    S_neg = k_B × ln(η) = -k_B × σ(r - z_c)²

    Uses direct formula for numerical stability (avoids exp→log roundtrip).

    Properties:
        - S_neg(z_c) = 0 (maximum order, zero entropy)
        - S_neg < 0 away from z_c (entropy deficit = order)
        - Units: J/K (same as Boltzmann constant)

    Args:
        r: Coherence parameter ∈ [0, 1]
        sigma: Sharpness parameter

    Returns:
        S_neg in J/K (always ≤ 0, maximum 0 at r = z_c)
    """
    # Direct formula: S_neg = -k_B × σ(r - z_c)²
    # Equivalent to k_B × ln(exp(-σ(r-z_c)²)) but numerically stable
    return -K_BOLTZMANN * sigma * (r - Z_C) ** 2


def negentropy_physical_direct(r: float, sigma: float = SIGMA_CANONICAL) -> float:
    """
    Physical negentropy using direct formula (no log).

    S_neg = -k_B × σ(r - z_c)²

    This avoids numerical issues with log(η) when η → 0.

    Args:
        r: Coherence parameter ∈ [0, 1]
        sigma: Sharpness parameter

    Returns:
        S_neg in J/K
    """
    return -K_BOLTZMANN * sigma * (r - Z_C) ** 2


def negentropy_gradient_physical(r: float, sigma: float = SIGMA_CANONICAL) -> float:
    """
    Gradient of physical negentropy dS_neg/dr in J/(K·r).

    dS_neg/dr = -2k_B × σ(r - z_c)

    Args:
        r: Coherence parameter ∈ [0, 1]
        sigma: Sharpness parameter

    Returns:
        dS_neg/dr in J/K per unit r
    """
    return -2 * K_BOLTZMANN * sigma * (r - Z_C)


# ═══════════════════════════════════════════════════════════════════════════
# CHARACTERISTIC ENERGY SCALE
# ═══════════════════════════════════════════════════════════════════════════

def characteristic_energy_j() -> float:
    """
    Characteristic energy scale of the L₄ system.

    E_char = h × f'_G = hc/λ_G (green photon energy)

    The green channel (528 Hz → 516.4 nm) provides the
    characteristic energy scale for the L₄ framework.

    Returns:
        E_char ≈ 3.85 × 10⁻¹⁹ J
    """
    from .photon_physics import photon_energy_j
    return photon_energy_j(528)  # Green channel reference


def characteristic_energy_ev() -> float:
    """
    Characteristic energy scale in electron volts.

    Returns:
        E_char ≈ 2.40 eV
    """
    from .photon_physics import photon_energy_ev
    return photon_energy_ev(528)


def characteristic_entropy_j_k() -> float:
    """
    Characteristic entropy scale: k_B.

    The Boltzmann constant provides the natural
    entropy scale for the physical negentropy.

    Returns:
        k_B ≈ 1.38 × 10⁻²³ J/K
    """
    return K_BOLTZMANN


# ═══════════════════════════════════════════════════════════════════════════
# THERMAL COHERENCE
# ═══════════════════════════════════════════════════════════════════════════

def thermal_coherence_temperature(r: float, sigma: float = SIGMA_CANONICAL) -> float:
    """
    Effective temperature associated with coherence level.

    T_eff = E_char / (k_B × |ln(η)|)

    Physical interpretation:
        - At r = z_c: T → ∞ (maximum order, infinite effective temperature)
        - Away from z_c: T → finite (thermal disorder)

    This maps the coherence parameter to an effective
    thermodynamic temperature.

    Args:
        r: Coherence parameter ∈ [0, 1]
        sigma: Sharpness parameter

    Returns:
        Temperature in Kelvin (or inf at z_c)
    """
    eta = negentropy_dimensionless(r, sigma)
    if eta >= 1 - 1e-10:
        return float('inf')
    E_char = characteristic_energy_j()
    return E_char / (K_BOLTZMANN * abs(math.log(eta)))


def coherence_from_temperature(T: float, sigma: float = SIGMA_CANONICAL) -> Optional[float]:
    """
    Compute coherence parameter from effective temperature.

    Inverse of thermal_coherence_temperature.

    Args:
        T: Effective temperature in Kelvin
        sigma: Sharpness parameter

    Returns:
        Coherence parameter r, or None if T is invalid
    """
    if T <= 0:
        return None
    if T == float('inf'):
        return Z_C

    E_char = characteristic_energy_j()
    ln_eta = -E_char / (K_BOLTZMANN * T)

    # η = exp(ln_eta), so σ(r - z_c)² = -ln_eta
    deviation_sq = -ln_eta / sigma
    if deviation_sq < 0:
        return None

    deviation = math.sqrt(deviation_sq)
    # Two solutions: z_c ± deviation
    # Return the one in [0, 1]
    r1 = Z_C + deviation
    r2 = Z_C - deviation

    if 0 <= r1 <= 1:
        return r1
    if 0 <= r2 <= 1:
        return r2
    return None


# ═══════════════════════════════════════════════════════════════════════════
# L₄ THRESHOLD INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

def negentropy_at_k_formation(sigma: float = SIGMA_CANONICAL) -> float:
    """
    Negentropy at K-formation threshold (r = K ≈ 0.924).

    Returns:
        η(K) dimensionless
    """
    return negentropy_dimensionless(L4_K, sigma)


def negentropy_physical_at_k_formation(sigma: float = SIGMA_CANONICAL) -> float:
    """
    Physical negentropy at K-formation threshold.

    Returns:
        S_neg(K) in J/K
    """
    return negentropy_physical(L4_K, sigma)


def negentropy_at_tau(sigma: float = SIGMA_CANONICAL) -> float:
    """
    Negentropy at paradox threshold (r = τ = φ⁻¹ ≈ 0.618).

    Returns:
        η(τ) dimensionless
    """
    return negentropy_dimensionless(PHI_INV, sigma)


def negentropy_profile() -> dict:
    """
    Compute negentropy at all L₄ threshold points.

    Returns:
        Dict mapping threshold names to (η, S_neg) tuples
    """
    from .constants import (
        L4_PARADOX, L4_ACTIVATION, L4_LENS, L4_CRITICAL,
        L4_IGNITION, L4_K_FORMATION, L4_CONSOLIDATION,
        L4_RESONANCE, L4_UNITY
    )

    thresholds = [
        ('PARADOX', L4_PARADOX),
        ('ACTIVATION', L4_ACTIVATION),
        ('LENS', L4_LENS),
        ('CRITICAL', L4_CRITICAL),
        ('IGNITION', L4_IGNITION),
        ('K_FORMATION', L4_K_FORMATION),
        ('CONSOLIDATION', L4_CONSOLIDATION),
        ('RESONANCE', L4_RESONANCE),
        ('UNITY', L4_UNITY),
    ]

    result = {}
    for name, r in thresholds:
        eta = negentropy_dimensionless(r)
        S_neg = negentropy_physical(r)
        result[name] = {
            'r': r,
            'eta': eta,
            'S_neg_jk': S_neg
        }

    return result


# ═══════════════════════════════════════════════════════════════════════════
# KURAMOTO INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_effective_coupling(r: float, K0: float = 1.0,
                                lambda_mod: float = 0.1,
                                sigma: float = SIGMA_CANONICAL) -> float:
    """
    Compute K_eff with negentropy modulation.

    K_eff = K₀ × (1 + λ_mod × η(r))

    The effective coupling strength is modulated by
    the negentropy, increasing near z_c.

    Args:
        r: Coherence parameter ∈ [0, 1]
        K0: Base coupling strength
        lambda_mod: Modulation coefficient
        sigma: Sharpness parameter

    Returns:
        Effective coupling strength (dimensionless)
    """
    eta = negentropy_dimensionless(r, sigma)
    return K0 * (1 + lambda_mod * eta)


def coherence_to_order_parameter(r: float, sigma: float = SIGMA_CANONICAL) -> float:
    """
    Map coherence parameter to Kuramoto order parameter.

    The Kuramoto order parameter ρ measures synchronization:
        ρ = 0: fully desynchronized
        ρ = 1: fully synchronized

    This mapping uses η as the order parameter.

    Args:
        r: Coherence parameter
        sigma: Sharpness parameter

    Returns:
        Order parameter ρ ∈ (0, 1]
    """
    return negentropy_dimensionless(r, sigma)


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def verify_sharpness_axiom(sigma: float = SIGMA_CANONICAL) -> dict:
    """
    Verify the sharpness axiom η(1) = e⁻¹.

    Returns:
        Dict with verification results
    """
    eta_at_1 = negentropy_dimensionless(1.0, sigma)
    expected = math.exp(-1)
    error = abs(eta_at_1 - expected) / expected

    return {
        'eta_at_1': eta_at_1,
        'expected': expected,
        'relative_error': error,
        'passes': error < 1e-10
    }


def verify_maximum_at_zc(sigma: float = SIGMA_CANONICAL) -> dict:
    """
    Verify η is maximized at z_c.

    Returns:
        Dict with verification results
    """
    eta_zc = negentropy_dimensionless(Z_C, sigma)
    grad_zc = negentropy_gradient(Z_C, sigma)
    curv_zc = negentropy_curvature(Z_C, sigma)

    return {
        'eta_at_zc': eta_zc,
        'gradient_at_zc': grad_zc,
        'curvature_at_zc': curv_zc,
        'is_maximum': eta_zc == 1.0 and abs(grad_zc) < 1e-10 and curv_zc < 0
    }


def verify_physical_units() -> dict:
    """
    Verify physical negentropy has correct units (J/K).

    Returns:
        Dict with verification results
    """
    S_neg = negentropy_physical(0.5)

    # S_neg should be on order of k_B
    ratio = abs(S_neg) / K_BOLTZMANN

    return {
        'S_neg_at_0.5': S_neg,
        'k_B': K_BOLTZMANN,
        'ratio_to_kB': ratio,
        'reasonable_scale': 0.01 < ratio < 100
    }


# ═══════════════════════════════════════════════════════════════════════════
# MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    # Sharpness
    'compute_sigma_from_axiom',
    'SIGMA_CANONICAL',
    # Dimensionless
    'negentropy_dimensionless',
    'negentropy_gradient',
    'negentropy_curvature',
    # Physical (SI units)
    'negentropy_physical',
    'negentropy_physical_direct',
    'negentropy_gradient_physical',
    # Energy scales
    'characteristic_energy_j',
    'characteristic_energy_ev',
    'characteristic_entropy_j_k',
    # Thermal
    'thermal_coherence_temperature',
    'coherence_from_temperature',
    # L₄ thresholds
    'negentropy_at_k_formation',
    'negentropy_physical_at_k_formation',
    'negentropy_at_tau',
    'negentropy_profile',
    # Kuramoto
    'compute_effective_coupling',
    'coherence_to_order_parameter',
    # Validation
    'verify_sharpness_axiom',
    'verify_maximum_at_zc',
    'verify_physical_units',
]
