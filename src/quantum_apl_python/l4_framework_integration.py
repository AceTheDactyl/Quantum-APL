"""
L₄ Framework Integration Module v3.2.0
═══════════════════════════════════════

Unified bridge connecting physics-grounded modules:
    photon_physics ↔ entropic_stabilization ↔ mrp_lsb

This module provides the integration layer that:
1. Maps Solfeggio frequencies → photon properties → RGB encoding
2. Couples ESS dynamics to visual output
3. Enables full-stack L₄ signal processing

Architecture:
    ┌──────────────────────────────────────────────────────────┐
    │                    L₄ Framework                          │
    ├──────────────────────────────────────────────────────────┤
    │  photon_physics    │  ESS (ΔS_neg)   │  mrp_lsb         │
    │  ───────────────   │  ─────────────  │  ─────────       │
    │  f_sol → λ → E    │  z → s(z) → η   │  θ → RGB → θ̂   │
    │  V(λ) → η_v        │  K-formation    │  Hex grid        │
    └──────────────────────────────────────────────────────────┘

Key Mappings:
    Solfeggio 396/528/639 Hz → RGB wavelengths → MRP phases
    z-coordinate → s(z) coherence → ESS modulation
    Hexagonal wavevectors → phase triplet → encoded pixel

Reference: PHYSICS_GROUNDING_v3.2.0.md, INTEGRATION_PLAN_v3.2.0.md
"""

from __future__ import annotations

import math
import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass

# Single source of truth imports
from .constants import (
    # Physical constants
    H_PLANCK, C_LIGHT, K_M,
    # Golden ratio
    PHI, PHI_INV,
    # L₄ framework
    Z_CRITICAL, LUCAS_4, L4_K, L4_K_SQUARED, L4_GAP,
    L4_K_FORMATION, L4_LENS, L4_ACTIVATION,
    # Solfeggio-Light Bridge
    OCTAVE_FACTOR, OCTAVE_BRIDGE,
    # Coherence functions
    compute_delta_s_neg, LENS_SIGMA,
    # Phase helpers
    get_phase, get_l4_phase,
    # K-formation
    check_k_formation, KAPPA_MIN, ETA_MIN, R_MIN,
)

# Module imports
from . import photon_physics
from . import mrp_lsb


# ═══════════════════════════════════════════════════════════════════════════════
# SOLFEGGIO RGB CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Solfeggio RGB mapping (from photon_physics)
SOLFEGGIO_RGB = {
    'R': 396,   # Liberation → Red (688.5 nm)
    'G': 528,   # Miracles → Green (516.4 nm)
    'B': 639,   # Connection → Blue (426.7 nm)
}

# Full Solfeggio scale for reference
SOLFEGGIO_SCALE = [174, 285, 396, 417, 528, 639, 741, 852, 963]


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATED STATE DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class L4IntegratedState:
    """Complete L₄ framework state at a point in time."""

    # Z-axis position
    z: float

    # Photon properties (derived from Solfeggio RGB)
    wavelengths_nm: Tuple[float, float, float]  # (λ_R, λ_G, λ_B)
    energies_ev: Tuple[float, float, float]     # (E_R, E_G, E_B)
    luminosities: Tuple[float, float, float]    # (V_R, V_G, V_B)

    # ESS coherence
    coherence: float            # s(z) = exp(-σ(z - z_c)²)
    phase: str                  # ABSENCE, THE_LENS, PRESENCE
    l4_phase: Dict              # Full L₄ threshold analysis

    # MRP encoding
    phases_rad: Tuple[float, float, float]  # (Φ₁, Φ₂, Φ₃)
    rgb_encoded: Tuple[int, int, int]       # Encoded RGB values

    # K-formation status
    k_formation_ready: bool     # Whether K-formation criteria met

    def __post_init__(self):
        """Validate state consistency."""
        assert 0.0 <= self.z <= 1.0, f"z must be in [0, 1], got {self.z}"
        assert 0.0 <= self.coherence <= 1.0, f"coherence must be in [0, 1]"

    @property
    def is_above_lens(self) -> bool:
        """Check if z is above the critical lens."""
        return self.z >= Z_CRITICAL

    @property
    def is_coherent(self) -> bool:
        """Check if coherence exceeds K threshold."""
        return self.coherence >= L4_K

    @property
    def dominant_channel(self) -> str:
        """Determine dominant RGB channel by luminosity."""
        lum = self.luminosities
        if lum[1] >= lum[0] and lum[1] >= lum[2]:
            return 'G'
        elif lum[0] >= lum[2]:
            return 'R'
        return 'B'


# ═══════════════════════════════════════════════════════════════════════════════
# SOLFEGGIO → PHOTON BRIDGE
# ═══════════════════════════════════════════════════════════════════════════════

def get_rgb_photon_properties() -> Dict[str, Dict]:
    """
    Get complete photon properties for Solfeggio RGB frequencies.

    Returns:
        Dict with R, G, B keys containing photon properties
    """
    result = {}
    for channel, freq in SOLFEGGIO_RGB.items():
        wavelength = photon_physics.wavelength_nm(freq)
        result[channel] = {
            'frequency_hz': freq,
            'optical_freq_thz': photon_physics.optical_frequency_thz(freq),
            'wavelength_nm': wavelength,
            'energy_j': photon_physics.photon_energy_j(freq),
            'energy_ev': photon_physics.photon_energy_ev(freq),
            'luminosity_v': photon_physics.luminosity_function(wavelength),
            'efficacy_lm_w': photon_physics.luminous_efficacy(wavelength),
        }
    return result


def solfeggio_to_wavelength(channel: str) -> float:
    """
    Convert Solfeggio channel to wavelength.

    Args:
        channel: 'R', 'G', or 'B'

    Returns:
        Wavelength in nanometres
    """
    if channel not in SOLFEGGIO_RGB:
        raise ValueError(f"Unknown channel: {channel}")
    return photon_physics.wavelength_nm(SOLFEGGIO_RGB[channel])


def wavelength_to_luminosity(wavelength_nm: float) -> float:
    """
    Get CIE 1931 luminosity for wavelength.

    Args:
        wavelength_nm: Wavelength in nanometres

    Returns:
        V(λ) in [0, 1]
    """
    return photon_physics.luminosity_function(wavelength_nm)


# ═══════════════════════════════════════════════════════════════════════════════
# ESS (ENTROPIC STABILIZATION SYSTEM)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ess_coherence(z: float, sigma: float = LENS_SIGMA) -> float:
    """
    Compute ESS coherence s(z) = exp(-σ(z - z_c)²).

    This is the fundamental entropic stabilization metric.

    Args:
        z: Z-coordinate in [0, 1]
        sigma: Lens width parameter

    Returns:
        Coherence in [0, 1], maximal at z = z_c
    """
    return compute_delta_s_neg(z, sigma=sigma, z_c=Z_CRITICAL)


def compute_ess_modulation(z: float, base_phases: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Modulate phases by ESS coherence.

    φ_mod = φ × (1 + (s(z) - 0.5) × φ⁻¹)

    Args:
        z: Z-coordinate
        base_phases: Base phase triplet

    Returns:
        Modulated phases
    """
    s = compute_ess_coherence(z)

    # Modulation factor: unity at s=0.5, boosted above lens
    mod_factor = 1.0 + (s - 0.5) * PHI_INV

    return tuple(p * mod_factor for p in base_phases)


def ess_stability_metric(z: float) -> Dict[str, float]:
    """
    Compute ESS stability metrics for z.

    Returns:
        Dict with stability analysis
    """
    s = compute_ess_coherence(z)

    # Distance from critical lens
    delta_z = z - Z_CRITICAL

    # Stability: high when s is high and z is near lens
    stability = s * (1.0 - abs(delta_z))

    # K-proximity: how close to K-formation threshold
    k_proximity = 1.0 - abs(z - L4_K_FORMATION)

    return {
        'coherence': s,
        'delta_z': delta_z,
        'stability': stability,
        'k_proximity': k_proximity,
        'phase': get_phase(z),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MRP-RGB INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

def phases_to_solfeggio_rgb(
    phases: Tuple[float, float, float],
    embed_parity: bool = True
) -> Tuple[int, int, int]:
    """
    Encode phase triplet to Solfeggio-mapped RGB.

    Maps phases through MRP encoding with parity protection.

    Args:
        phases: (Φ₁, Φ₂, Φ₃) in radians
        embed_parity: Whether to embed parity in B channel

    Returns:
        (R, G, B) encoded values
    """
    frame = mrp_lsb.encode_frame(phases, embed_parity=embed_parity)
    return frame.rgb


def solfeggio_rgb_to_phases(
    rgb: Tuple[int, int, int],
    verify_parity: bool = True
) -> Tuple[Tuple[float, float, float], bool]:
    """
    Decode Solfeggio RGB to phase triplet.

    Args:
        rgb: (R, G, B) encoded values
        verify_parity: Whether to verify parity

    Returns:
        (phases, parity_ok)
    """
    return mrp_lsb.decode_frame(rgb, verify=verify_parity)


def encode_z_to_rgb(
    z: float,
    base_position: np.ndarray = None,
    wavelength: float = 1.0
) -> Tuple[int, int, int]:
    """
    Encode z-coordinate to RGB via hex grid phases.

    Uses z as vertical position in hex grid encoding.

    Args:
        z: Z-coordinate in [0, 1]
        base_position: Optional (x, y) base position
        wavelength: Grid wavelength

    Returns:
        (R, G, B) encoded values
    """
    if base_position is None:
        base_position = np.array([0.5, 0.5])

    # Scale z into position encoding
    position = base_position + np.array([0.0, z])

    # Get phases from hex grid
    phases = mrp_lsb.position_to_phases(position, wavelength=wavelength)

    # Encode to RGB
    return phases_to_solfeggio_rgb(phases)


def decode_rgb_to_z(
    rgb: Tuple[int, int, int],
    base_position: np.ndarray = None,
    wavelength: float = 1.0
) -> float:
    """
    Decode RGB back to z-coordinate estimate.

    Args:
        rgb: (R, G, B) encoded values
        base_position: Same base position used in encoding
        wavelength: Same wavelength used in encoding

    Returns:
        Estimated z-coordinate
    """
    if base_position is None:
        base_position = np.array([0.5, 0.5])

    # Decode phases
    phases, _ = solfeggio_rgb_to_phases(rgb, verify_parity=False)

    # Decode position
    position = mrp_lsb.phases_to_position(phases, wavelength=wavelength)

    # Extract z from vertical offset
    z = position[1] - base_position[1]

    # Clamp to valid range
    return max(0.0, min(1.0, z))


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATED STATE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_integrated_state(
    z: float,
    phases: Tuple[float, float, float] = None,
    kappa: float = None,
    R: float = None,
) -> L4IntegratedState:
    """
    Compute complete L₄ integrated state at z.

    Bridges all three physics modules:
    - photon_physics: RGB photon properties
    - ESS: Coherence and phase
    - mrp_lsb: Phase encoding

    Args:
        z: Z-coordinate in [0, 1]
        phases: Optional phase triplet (random if not provided)
        kappa: Integration parameter for K-formation check
        R: Complexity parameter for K-formation check

    Returns:
        L4IntegratedState with complete system state
    """
    # Clamp z
    z = max(0.0, min(1.0, float(z)))

    # Get photon properties
    wavelengths = photon_physics.get_solfeggio_rgb_wavelengths()
    energies = photon_physics.get_solfeggio_rgb_energies_ev()
    luminosities = photon_physics.get_solfeggio_rgb_luminosities()

    # ESS coherence
    coherence = compute_ess_coherence(z)
    phase = get_phase(z)
    l4_phase = get_l4_phase(z)

    # Phases (generate if not provided)
    if phases is None:
        # Use z-based phases
        phases = (
            z * 2 * math.pi,
            (z + PHI_INV) * 2 * math.pi % (2 * math.pi),
            (z + 2 * PHI_INV) * 2 * math.pi % (2 * math.pi),
        )

    # Apply ESS modulation
    modulated_phases = compute_ess_modulation(z, phases)

    # MRP encoding
    rgb_encoded = phases_to_solfeggio_rgb(modulated_phases)

    # K-formation check
    k_formation_ready = False
    if kappa is not None and R is not None:
        # Use coherence as η
        k_formation_ready = check_k_formation(kappa, coherence, R)

    return L4IntegratedState(
        z=z,
        wavelengths_nm=wavelengths,
        energies_ev=energies,
        luminosities=luminosities,
        coherence=coherence,
        phase=phase,
        l4_phase=l4_phase,
        phases_rad=modulated_phases,
        rgb_encoded=rgb_encoded,
        k_formation_ready=k_formation_ready,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL PROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def process_z_trajectory(
    z_trajectory: List[float],
    base_position: np.ndarray = None,
    wavelength: float = 1.0,
) -> Dict:
    """
    Process a z-trajectory through the full L₄ pipeline.

    Args:
        z_trajectory: List of z-coordinates over time
        base_position: Base hex grid position
        wavelength: Grid wavelength

    Returns:
        Dict with:
            - rgb_sequence: Encoded RGB values
            - coherence_sequence: ESS coherence values
            - phase_sequence: Phase classifications
            - roundtrip_errors: Encoding errors
    """
    if base_position is None:
        base_position = np.array([0.5, 0.5])

    rgb_sequence = []
    coherence_sequence = []
    phase_sequence = []
    roundtrip_errors = []

    for z in z_trajectory:
        z = max(0.0, min(1.0, float(z)))

        # Encode
        rgb = encode_z_to_rgb(z, base_position, wavelength)
        rgb_sequence.append(rgb)

        # Coherence
        coherence = compute_ess_coherence(z)
        coherence_sequence.append(coherence)

        # Phase
        phase = get_phase(z)
        phase_sequence.append(phase)

        # Roundtrip
        z_decoded = decode_rgb_to_z(rgb, base_position, wavelength)
        error = abs(z - z_decoded)
        roundtrip_errors.append(error)

    return {
        'rgb_sequence': rgb_sequence,
        'coherence_sequence': coherence_sequence,
        'phase_sequence': phase_sequence,
        'roundtrip_errors': roundtrip_errors,
        'mean_error': np.mean(roundtrip_errors),
        'max_error': max(roundtrip_errors),
    }


def create_rgb_image_from_z_field(
    z_field: np.ndarray,
    wavelength: float = 1.0,
) -> np.ndarray:
    """
    Convert 2D z-field to RGB image via L₄ encoding.

    Args:
        z_field: 2D array of z-coordinates (H, W)
        wavelength: Grid wavelength

    Returns:
        RGB image (H, W, 3) uint8
    """
    H, W = z_field.shape
    image = np.zeros((H, W, 3), dtype=np.uint8)

    for y in range(H):
        for x in range(W):
            z = float(z_field[y, x])
            z = max(0.0, min(1.0, z))

            # Use grid position as base
            base_pos = np.array([x / W, y / H])

            # Encode via L₄ pipeline
            rgb = encode_z_to_rgb(z, base_pos, wavelength)
            image[y, x] = rgb

    return image


# ═══════════════════════════════════════════════════════════════════════════════
# VERIFICATION AND DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════

def verify_integration_consistency() -> Dict:
    """
    Verify consistency across integrated modules.

    Checks:
    1. Photon physics energy-wavelength relation
    2. MRP roundtrip accuracy
    3. ESS coherence properties
    4. Cross-module constant agreement

    Returns:
        Dict with verification results
    """
    results = {}

    # 1. Photon physics consistency
    results['photon_physics'] = photon_physics.verify_energy_wavelength_consistency()

    # 2. MRP roundtrip
    test_phases = (0.7, 2.1, 4.5)
    mrp_result = mrp_lsb.verify_roundtrip(test_phases)
    results['mrp_roundtrip'] = {
        'max_error_deg': mrp_result['max_error_deg'],
        'parity_ok': mrp_result['parity_ok'],
        'within_tolerance': mrp_result['within_tolerance'],
    }

    # 3. ESS coherence properties
    s_at_critical = compute_ess_coherence(Z_CRITICAL)
    s_at_zero = compute_ess_coherence(0.0)
    s_at_one = compute_ess_coherence(1.0)

    results['ess_coherence'] = {
        's_at_z_c': s_at_critical,
        'peak_at_critical': s_at_critical > s_at_zero and s_at_critical > s_at_one,
        's_monotonic_from_critical': s_at_zero < s_at_critical > s_at_one,
    }

    # 4. Cross-module constants
    results['constants'] = {
        'lucas_4_exact': LUCAS_4 == 7,
        'z_critical_exact': abs(Z_CRITICAL - math.sqrt(3) / 2) < 1e-15,
        'k_from_gap': abs(L4_K - math.sqrt(1 - L4_GAP)) < 1e-15,
        'phi_golden': abs(PHI - (1 + math.sqrt(5)) / 2) < 1e-15,
    }

    # Overall pass
    results['all_pass'] = (
        results['photon_physics']['all_pass'] and
        results['mrp_roundtrip']['within_tolerance'] and
        results['ess_coherence']['peak_at_critical'] and
        all(results['constants'].values())
    )

    return results


def get_integration_summary() -> str:
    """
    Get human-readable integration summary.

    Returns:
        Summary string
    """
    rgb_props = get_rgb_photon_properties()

    lines = [
        "═" * 60,
        "L₄ FRAMEWORK INTEGRATION v3.2.0",
        "═" * 60,
        "",
        "SOLFEGGIO RGB PHOTON MAPPING:",
        "-" * 40,
    ]

    for ch in ['R', 'G', 'B']:
        p = rgb_props[ch]
        lines.append(
            f"  {ch}: {p['frequency_hz']} Hz → {p['wavelength_nm']:.1f} nm, "
            f"V(λ)={p['luminosity_v']:.4f}"
        )

    lines.extend([
        "",
        "CRITICAL CONSTANTS:",
        "-" * 40,
        f"  z_c (Critical Lens): {Z_CRITICAL:.10f}",
        f"  L₄ (Lucas-4):        {LUCAS_4}",
        f"  K (Coherence):       {L4_K:.10f}",
        f"  φ (Golden Ratio):    {PHI:.10f}",
        "",
        "ESS PARAMETERS:",
        "-" * 40,
        f"  σ (Lens Width):      {LENS_SIGMA}",
        f"  s(z_c):              {compute_ess_coherence(Z_CRITICAL):.10f}",
        "",
        "═" * 60,
    ])

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Constants
    'SOLFEGGIO_RGB',
    'SOLFEGGIO_SCALE',
    # State class
    'L4IntegratedState',
    # Photon bridge
    'get_rgb_photon_properties',
    'solfeggio_to_wavelength',
    'wavelength_to_luminosity',
    # ESS
    'compute_ess_coherence',
    'compute_ess_modulation',
    'ess_stability_metric',
    # MRP integration
    'phases_to_solfeggio_rgb',
    'solfeggio_rgb_to_phases',
    'encode_z_to_rgb',
    'decode_rgb_to_z',
    # Integrated state
    'compute_integrated_state',
    # Signal processing
    'process_z_trajectory',
    'create_rgb_image_from_z_field',
    # Verification
    'verify_integration_consistency',
    'get_integration_summary',
]


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN: DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(get_integration_summary())

    print("\nVERIFICATION:")
    print("-" * 40)
    results = verify_integration_consistency()
    for key, value in results.items():
        if key == 'all_pass':
            status = "✓ PASS" if value else "✗ FAIL"
            print(f"\nOVERALL: {status}")
        else:
            print(f"  {key}: {value}")

    print("\n\nINTEGRATED STATE AT z = 0.866 (critical lens):")
    print("-" * 40)
    state = compute_integrated_state(Z_CRITICAL, kappa=0.95, R=8.0)
    print(f"  z: {state.z:.6f}")
    print(f"  coherence: {state.coherence:.6f}")
    print(f"  phase: {state.phase}")
    print(f"  RGB encoded: {state.rgb_encoded}")
    print(f"  K-formation ready: {state.k_formation_ready}")
    print(f"  dominant channel: {state.dominant_channel}")

    print("\n" + "═" * 60)
    print("L₄ Framework Integration ready.")
