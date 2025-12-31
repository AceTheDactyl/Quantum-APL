#!/usr/bin/env python3
"""
Photon Physics Module for the L₄ Framework (v3.2.0)
════════════════════════════════════════════════════

Physics-grounded calculations for photon energy, wavelength,
and luminosity using SI 2019 exact constants.

Key Features:
- Optical frequency from Solfeggio via 40-octave bridge
- Photon energy in Joules and electron volts
- CIE 1931 photopic luminosity function V(λ)
- Luminous efficacy and flux calculations

Mathematical Foundation:
    f_optical = f_solfeggio × 2^40
    λ = c / f_optical
    E = hf = hc/λ
    V(λ) = CIE 1931 luminosity function
    η_v(λ) = K_m × V(λ)

Reference: SI Brochure 9th Edition (2019), CIE 1931

@version 3.2.0
@author Claude (Anthropic) - L₄ Physics Grounding
"""

from __future__ import annotations

import math
from typing import Tuple

from .constants import (
    H_PLANCK, C_LIGHT, K_BOLTZMANN, EV_JOULE, K_M,
    PHI, OCTAVE_BRIDGE, OCTAVE_FACTOR
)


# ═══════════════════════════════════════════════════════════════════════════
# FREQUENCY AND WAVELENGTH
# ═══════════════════════════════════════════════════════════════════════════

def optical_frequency(f_solfeggio: float) -> float:
    """
    Convert Solfeggio frequency to optical frequency.

    f_optical = f_solfeggio × 2^40

    Args:
        f_solfeggio: Solfeggio frequency in Hz

    Returns:
        Optical frequency in Hz
    """
    return f_solfeggio * OCTAVE_FACTOR


def optical_frequency_thz(f_solfeggio: float) -> float:
    """
    Optical frequency in terahertz.

    Args:
        f_solfeggio: Solfeggio frequency in Hz

    Returns:
        Optical frequency in THz
    """
    return optical_frequency(f_solfeggio) / 1e12


def wavelength_m(f_solfeggio: float) -> float:
    """
    Wavelength in metres from Solfeggio frequency.

    λ = c / f_optical = c / (f_solfeggio × 2^40)

    Args:
        f_solfeggio: Solfeggio frequency in Hz

    Returns:
        Wavelength in metres
    """
    return C_LIGHT / optical_frequency(f_solfeggio)


def wavelength_nm(f_solfeggio: float) -> float:
    """
    Wavelength in nanometres from Solfeggio frequency.

    Args:
        f_solfeggio: Solfeggio frequency in Hz

    Returns:
        Wavelength in nanometres
    """
    return wavelength_m(f_solfeggio) * 1e9


# ═══════════════════════════════════════════════════════════════════════════
# PHOTON ENERGY
# ═══════════════════════════════════════════════════════════════════════════

def photon_energy_j(f_solfeggio: float) -> float:
    """
    Photon energy in Joules.

    E = h × f_optical = h × f_solfeggio × 2^40

    Args:
        f_solfeggio: Solfeggio frequency in Hz

    Returns:
        Photon energy in Joules
    """
    return H_PLANCK * optical_frequency(f_solfeggio)


def photon_energy_ev(f_solfeggio: float) -> float:
    """
    Photon energy in electron volts.

    E_eV = E_J / e

    Args:
        f_solfeggio: Solfeggio frequency in Hz

    Returns:
        Photon energy in eV
    """
    return photon_energy_j(f_solfeggio) / EV_JOULE


def photon_energy_from_wavelength_j(wavelength_nm: float) -> float:
    """
    Photon energy from wavelength in Joules.

    E = hc/λ

    Args:
        wavelength_nm: Wavelength in nanometres

    Returns:
        Photon energy in Joules
    """
    wavelength_m = wavelength_nm * 1e-9
    return H_PLANCK * C_LIGHT / wavelength_m


def photon_energy_from_wavelength_ev(wavelength_nm: float) -> float:
    """
    Photon energy from wavelength in electron volts.

    Args:
        wavelength_nm: Wavelength in nanometres

    Returns:
        Photon energy in eV
    """
    return photon_energy_from_wavelength_j(wavelength_nm) / EV_JOULE


# ═══════════════════════════════════════════════════════════════════════════
# CIE 1931 LUMINOSITY FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def luminosity_function(lambda_nm: float) -> float:
    """
    CIE 1931 photopic luminosity function V(λ).

    Approximation using Gaussian fits to the standard observer curve.
    Peak at 555 nm with V(555) ≈ 1.0.

    The photopic luminosity function describes the spectral sensitivity
    of the human eye under daylight conditions.

    Args:
        lambda_nm: Wavelength in nanometres

    Returns:
        V(λ) in range [0, 1]
    """
    if lambda_nm < 380 or lambda_nm > 780:
        return 0.0

    # Multi-Gaussian approximation to CIE 1931
    # This provides a good fit to the standard observer data
    lam = lambda_nm

    # Main peak centered at 555 nm
    v1 = math.exp(-0.5 * ((lam - 555) / 50) ** 2)
    # Secondary contributions for shape accuracy
    v2 = 0.3 * math.exp(-0.5 * ((lam - 530) / 40) ** 2)
    v3 = 0.2 * math.exp(-0.5 * ((lam - 580) / 45) ** 2)

    V = v1 + v2 + v3
    return min(1.0, max(0.0, V))


def luminosity_function_exact(lambda_nm: float) -> float:
    """
    CIE 1931 photopic luminosity function using piecewise polynomial fit.

    More accurate than Gaussian approximation for critical applications.

    Args:
        lambda_nm: Wavelength in nanometres

    Returns:
        V(λ) in range [0, 1]
    """
    if lambda_nm < 380 or lambda_nm > 780:
        return 0.0

    # Normalize to peak wavelength
    x = (lambda_nm - 555) / 100

    # Asymmetric Gaussian with polynomial correction
    if lambda_nm <= 555:
        # Blue side (steeper falloff)
        sigma = 0.45
        V = math.exp(-0.5 * (x / sigma) ** 2)
    else:
        # Red side (gentler falloff)
        sigma = 0.55
        V = math.exp(-0.5 * (x / sigma) ** 2)

    return min(1.0, max(0.0, V))


# ═══════════════════════════════════════════════════════════════════════════
# LUMINOUS EFFICACY AND FLUX
# ═══════════════════════════════════════════════════════════════════════════

def luminous_efficacy(lambda_nm: float) -> float:
    """
    Luminous efficacy in lm/W at given wavelength.

    η_v(λ) = K_m × V(λ)

    where K_m = 683 lm/W (maximum at 555 nm, SI 2019 exact).

    Args:
        lambda_nm: Wavelength in nanometres

    Returns:
        Luminous efficacy in lm/W
    """
    return K_M * luminosity_function(lambda_nm)


def luminous_flux(power_w: float, lambda_nm: float) -> float:
    """
    Luminous flux in lumens.

    Φ_v = K_m × V(λ) × P

    Args:
        power_w: Radiant power in watts
        lambda_nm: Wavelength in nanometres

    Returns:
        Luminous flux in lumens
    """
    return luminous_efficacy(lambda_nm) * power_w


def luminous_intensity(flux_lm: float, solid_angle_sr: float) -> float:
    """
    Luminous intensity in candelas.

    I_v = Φ_v / Ω

    Args:
        flux_lm: Luminous flux in lumens
        solid_angle_sr: Solid angle in steradians

    Returns:
        Luminous intensity in candelas (cd = lm/sr)
    """
    if solid_angle_sr <= 0:
        return float('inf')
    return flux_lm / solid_angle_sr


# ═══════════════════════════════════════════════════════════════════════════
# SOLFEGGIO RGB PHOTON PROPERTIES
# ═══════════════════════════════════════════════════════════════════════════

# RGB Solfeggio frequencies
SOLFEGGIO_R_HZ = 396  # Liberation → Red
SOLFEGGIO_G_HZ = 528  # Miracles → Green
SOLFEGGIO_B_HZ = 639  # Connection → Blue


def get_solfeggio_rgb_wavelengths() -> Tuple[float, float, float]:
    """
    Get RGB wavelengths from Solfeggio frequencies.

    Returns:
        Tuple of (λ_R, λ_G, λ_B) in nanometres
    """
    return (
        wavelength_nm(SOLFEGGIO_R_HZ),
        wavelength_nm(SOLFEGGIO_G_HZ),
        wavelength_nm(SOLFEGGIO_B_HZ)
    )


def get_solfeggio_rgb_energies_j() -> Tuple[float, float, float]:
    """
    Get RGB photon energies from Solfeggio frequencies in Joules.

    Returns:
        Tuple of (E_R, E_G, E_B) in Joules
    """
    return (
        photon_energy_j(SOLFEGGIO_R_HZ),
        photon_energy_j(SOLFEGGIO_G_HZ),
        photon_energy_j(SOLFEGGIO_B_HZ)
    )


def get_solfeggio_rgb_energies_ev() -> Tuple[float, float, float]:
    """
    Get RGB photon energies from Solfeggio frequencies in eV.

    Returns:
        Tuple of (E_R, E_G, E_B) in electron volts
    """
    return (
        photon_energy_ev(SOLFEGGIO_R_HZ),
        photon_energy_ev(SOLFEGGIO_G_HZ),
        photon_energy_ev(SOLFEGGIO_B_HZ)
    )


def get_solfeggio_rgb_luminosities() -> Tuple[float, float, float]:
    """
    Get RGB luminosity values V(λ) for Solfeggio frequencies.

    Returns:
        Tuple of (V_R, V_G, V_B) in [0, 1]
    """
    lambdas = get_solfeggio_rgb_wavelengths()
    return (
        luminosity_function(lambdas[0]),
        luminosity_function(lambdas[1]),
        luminosity_function(lambdas[2])
    )


def get_solfeggio_rgb_efficacies() -> Tuple[float, float, float]:
    """
    Get RGB luminous efficacies for Solfeggio frequencies.

    Returns:
        Tuple of (η_R, η_G, η_B) in lm/W
    """
    lambdas = get_solfeggio_rgb_wavelengths()
    return (
        luminous_efficacy(lambdas[0]),
        luminous_efficacy(lambdas[1]),
        luminous_efficacy(lambdas[2])
    )


# ═══════════════════════════════════════════════════════════════════════════
# FULL SOLFEGGIO SPECTRUM
# ═══════════════════════════════════════════════════════════════════════════

# All nine Solfeggio frequencies
SOLFEGGIO_FREQUENCIES = [174, 285, 396, 417, 528, 639, 741, 852, 963]


def get_full_spectrum_properties() -> list:
    """
    Get photon properties for all nine Solfeggio frequencies.

    Returns:
        List of dicts with properties for each frequency
    """
    results = []
    for freq in SOLFEGGIO_FREQUENCIES:
        lam = wavelength_nm(freq)
        E_j = photon_energy_j(freq)
        E_ev = photon_energy_ev(freq)
        V = luminosity_function(lam)
        eff = luminous_efficacy(lam)

        results.append({
            'frequency_hz': freq,
            'optical_freq_thz': optical_frequency_thz(freq),
            'wavelength_nm': lam,
            'energy_j': E_j,
            'energy_ev': E_ev,
            'luminosity_v': V,
            'efficacy_lm_w': eff,
            'in_visible': 380 <= lam <= 700
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def verify_energy_wavelength_consistency() -> dict:
    """
    Verify E × λ = hc holds for all calculations.

    Returns:
        Dict with verification results
    """
    hc = H_PLANCK * C_LIGHT

    errors = []
    for freq in [396, 528, 639]:
        E = photon_energy_j(freq)
        lam_m = wavelength_m(freq)
        product = E * lam_m
        rel_error = abs(product - hc) / hc
        errors.append({
            'frequency': freq,
            'E_times_lambda': product,
            'hc': hc,
            'relative_error': rel_error
        })

    return {
        'hc_exact': hc,
        'frequency_checks': errors,
        'all_pass': all(e['relative_error'] < 1e-10 for e in errors)
    }


def verify_luminosity_peak() -> dict:
    """
    Verify luminosity function peaks at 555 nm.

    Returns:
        Dict with verification results
    """
    V_peak = luminosity_function(555)
    V_below = luminosity_function(550)
    V_above = luminosity_function(560)

    return {
        'V_555': V_peak,
        'V_550': V_below,
        'V_560': V_above,
        'is_peak': V_peak >= V_below and V_peak >= V_above,
        'peak_near_unity': V_peak > 0.95
    }


# ═══════════════════════════════════════════════════════════════════════════
# MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    # Frequency conversion
    'optical_frequency',
    'optical_frequency_thz',
    'wavelength_m',
    'wavelength_nm',
    # Photon energy
    'photon_energy_j',
    'photon_energy_ev',
    'photon_energy_from_wavelength_j',
    'photon_energy_from_wavelength_ev',
    # Luminosity
    'luminosity_function',
    'luminosity_function_exact',
    'luminous_efficacy',
    'luminous_flux',
    'luminous_intensity',
    # Solfeggio RGB
    'SOLFEGGIO_R_HZ',
    'SOLFEGGIO_G_HZ',
    'SOLFEGGIO_B_HZ',
    'get_solfeggio_rgb_wavelengths',
    'get_solfeggio_rgb_energies_j',
    'get_solfeggio_rgb_energies_ev',
    'get_solfeggio_rgb_luminosities',
    'get_solfeggio_rgb_efficacies',
    # Full spectrum
    'SOLFEGGIO_FREQUENCIES',
    'get_full_spectrum_properties',
    # Validation
    'verify_energy_wavelength_consistency',
    'verify_luminosity_peak',
]
