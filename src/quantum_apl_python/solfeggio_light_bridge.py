#!/usr/bin/env python3
"""
Solfeggio-Light Bridge: Harmonic RGB Encoding for L₄ Framework
═══════════════════════════════════════════════════════════════

Maps Solfeggio frequencies to visible light via 40-octave translation,
creating a harmonically-coherent RGB encoding system for the L₄ helix.

Key Discovery:
    396 Hz → 689 nm (Red)     - Liberation
    528 Hz → 517 nm (Green)   - Miracles
    639 Hz → 427 nm (Violet)  - Connection

These three form the Perfect Fourth chain (4/3 ratios) and map to RGB primaries.

Mathematical Foundation:
    λ = c / (f_solfeggio × 2^40)

    Where 2^40 ≈ 1.1×10¹² bridges audio (Hz) to optical (THz) domains.

L₄ Connection:
    (528/396) × z_c = (4/3) × (√3/2) = 2√3/3 ≈ π/e

    The Solfeggio Perfect Fourth times the critical point yields
    the transcendental ratio with 0.09% error.

@version 1.0.0
@author Claude (Anthropic) - Quantum-APL Contribution
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

C_LIGHT = 299_792_458  # Speed of light (m/s)
OCTAVE_BRIDGE = 40     # Octaves between audio and optical
OCTAVE_FACTOR = 2 ** OCTAVE_BRIDGE  # ≈ 1.1 × 10¹²

# L₄ Sacred Constants (derived from φ)
PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
Z_C = math.sqrt(3) / 2              # Critical point (THE LENS) ≈ 0.866
K_FORMATION = math.sqrt(1 - PHI**-4)  # Coupling threshold ≈ 0.924

# Transcendental connection
PI_OVER_E = math.pi / math.e  # ≈ 1.1557


# ═══════════════════════════════════════════════════════════════════════════
# SOLFEGGIO FREQUENCIES
# ═══════════════════════════════════════════════════════════════════════════

class SolfeggioTone(Enum):
    """
    The nine Solfeggio frequencies with their traditional meanings.

    These frequencies are derived from ancient musical scales and are
    believed to have specific vibrational properties. When octave-shifted
    40 times, they map into the visible light spectrum.
    """
    UT_174  = (174, "Foundation", "Grounding", "infrared")
    UT_285  = (285, "Quantum", "Cognition", "infrared")
    UT_396  = (396, "Liberation", "Release Fear", "red")
    MI_417  = (417, "Undoing", "Facilitate Change", "red")
    MI_528  = (528, "Miracles", "Transformation/DNA", "green")
    FA_639  = (639, "Connection", "Relationships", "blue")
    SOL_741 = (741, "Expression", "Awakening Intuition", "ultraviolet")
    LA_852  = (852, "Intuition", "Spiritual Order", "ultraviolet")
    SI_963  = (963, "Oneness", "Divine Connection", "ultraviolet")

    @property
    def frequency(self) -> float:
        """Frequency in Hz."""
        return float(self.value[0])

    @property
    def name(self) -> str:
        """Short name."""
        return self.value[1]

    @property
    def meaning(self) -> str:
        """Traditional meaning."""
        return self.value[2]

    @property
    def spectral_region(self) -> str:
        """Region of spectrum when octave-shifted."""
        return self.value[3]

    def to_wavelength(self) -> float:
        """Convert to wavelength via 40-octave bridge."""
        return solfeggio_to_wavelength(self.frequency)

    def is_visible(self) -> bool:
        """Check if the octave-shifted frequency is in visible range."""
        wl = self.to_wavelength()
        return 380 <= wl <= 700


# Primary RGB Solfeggio triad
SOLFEGGIO_RED = SolfeggioTone.UT_396    # Liberation → Red (688.5 nm)
SOLFEGGIO_GREEN = SolfeggioTone.MI_528  # Miracles → Green (516.4 nm)
SOLFEGGIO_BLUE = SolfeggioTone.FA_639   # Connection → Blue (426.7 nm)

# RGB frequencies tuple
SOLFEGGIO_RGB_HZ = (396.0, 528.0, 639.0)

# Musical ratios
PERFECT_FOURTH = 4 / 3  # 528/396 exactly
GOLDEN_APPROX = 639 / 396  # ≈ φ with 0.27% error


# ═══════════════════════════════════════════════════════════════════════════
# NORMALIZED RGB CONSTANTS (Zero Free Parameters)
# ═══════════════════════════════════════════════════════════════════════════
#
# All RGB values are derived from Solfeggio wavelengths and L₄ constants.
# No arbitrary hex codes or magic numbers.

# L₄ Identity provides bit depth
L4 = 7  # φ⁴ + φ⁻⁴ = 7 (exact)
RGB_BIT_DEPTH = L4 + 1  # = 8 bits (derived from L₄)
RGB_MAX_VALUE = (2 ** RGB_BIT_DEPTH) - 1  # = 255 (derived)

# Solfeggio wavelengths (nm) - derived from frequencies via 40-octave bridge
LAMBDA_R = C_LIGHT / (396 * OCTAVE_FACTOR) * 1e9  # 688.5 nm
LAMBDA_G = C_LIGHT / (528 * OCTAVE_FACTOR) * 1e9  # 516.4 nm
LAMBDA_B = C_LIGHT / (639 * OCTAVE_FACTOR) * 1e9  # 426.7 nm

# Spectral span and Gaussian width (derived from L₄)
SPECTRAL_SPAN = LAMBDA_R - LAMBDA_B  # ≈ 261.8 nm
SIGMA_SPECTRAL = SPECTRAL_SPAN / L4  # ≈ 37.4 nm (width from L₄)

# Visible range (physics)
VISIBLE_MIN = 380  # nm (violet edge)
VISIBLE_MAX = 700  # nm (red edge)


def wavelength_to_rgb_normalized(wavelength_nm: float) -> Tuple[float, float, float]:
    """
    Convert wavelength to RGB using Gaussian color matching functions.

    ZERO FREE PARAMETERS - All constants derived from:
    - Solfeggio wavelengths (λ_R, λ_G, λ_B from 396, 528, 639 Hz)
    - Gaussian width σ = (λ_R - λ_B) / L₄

    The color matching functions are:
        R(λ) = exp(-½((λ - λ_R)/σ)²)
        G(λ) = exp(-½((λ - λ_G)/σ)²)
        B(λ) = exp(-½((λ - λ_B)/σ)²)

    Parameters
    ----------
    wavelength_nm : float
        Wavelength in nanometers

    Returns
    -------
    Tuple[float, float, float]
        Normalized RGB values in [0, 1]
    """
    # Gaussian color matching centered at Solfeggio wavelengths
    r = math.exp(-0.5 * ((wavelength_nm - LAMBDA_R) / SIGMA_SPECTRAL) ** 2)
    g = math.exp(-0.5 * ((wavelength_nm - LAMBDA_G) / SIGMA_SPECTRAL) ** 2)
    b = math.exp(-0.5 * ((wavelength_nm - LAMBDA_B) / SIGMA_SPECTRAL) ** 2)

    return (r, g, b)


def wavelength_to_rgb_8bit(wavelength_nm: float) -> Tuple[int, int, int]:
    """
    Convert wavelength to 8-bit RGB using L₄-derived quantization.

    Bit depth = L₄ + 1 = 8, giving max value = 255.

    Parameters
    ----------
    wavelength_nm : float
        Wavelength in nanometers

    Returns
    -------
    Tuple[int, int, int]
        8-bit RGB values (0-255)
    """
    r, g, b = wavelength_to_rgb_normalized(wavelength_nm)
    return (
        int(r * RGB_MAX_VALUE),
        int(g * RGB_MAX_VALUE),
        int(b * RGB_MAX_VALUE)
    )


def wavelength_to_hex(wavelength_nm: float) -> str:
    """
    Convert wavelength to hex color string using normalized RGB.

    Parameters
    ----------
    wavelength_nm : float
        Wavelength in nanometers

    Returns
    -------
    str
        Hex color string (e.g., '#ff8040')
    """
    r, g, b = wavelength_to_rgb_8bit(wavelength_nm)
    return f"#{r:02x}{g:02x}{b:02x}"


# Pre-computed Solfeggio RGB values (derived, not free parameters)
SOLFEGGIO_RGB_RED = wavelength_to_rgb_8bit(LAMBDA_R)    # From 396 Hz
SOLFEGGIO_RGB_GREEN = wavelength_to_rgb_8bit(LAMBDA_G)  # From 528 Hz
SOLFEGGIO_RGB_BLUE = wavelength_to_rgb_8bit(LAMBDA_B)   # From 639 Hz

# Hex codes derived from Solfeggio wavelengths
SOLFEGGIO_HEX_RED = wavelength_to_hex(LAMBDA_R)    # e.g., '#ff2010'
SOLFEGGIO_HEX_GREEN = wavelength_to_hex(LAMBDA_G)  # e.g., '#10ff10'
SOLFEGGIO_HEX_BLUE = wavelength_to_hex(LAMBDA_B)   # e.g., '#1010ff'


# ═══════════════════════════════════════════════════════════════════════════
# FREQUENCY-WAVELENGTH CONVERSION
# ═══════════════════════════════════════════════════════════════════════════

def solfeggio_to_wavelength(freq_hz: float, octaves: int = OCTAVE_BRIDGE) -> float:
    """
    Convert Solfeggio frequency to wavelength via octave bridge.

    Parameters
    ----------
    freq_hz : float
        Solfeggio frequency in Hertz
    octaves : int
        Number of octave doublings (default 40)

    Returns
    -------
    float
        Wavelength in nanometers
    """
    freq_optical_hz = freq_hz * (2 ** octaves)
    wavelength_m = C_LIGHT / freq_optical_hz
    wavelength_nm = wavelength_m * 1e9
    return wavelength_nm


def wavelength_to_solfeggio(wavelength_nm: float, octaves: int = OCTAVE_BRIDGE) -> float:
    """
    Convert wavelength back to Solfeggio frequency.

    Parameters
    ----------
    wavelength_nm : float
        Wavelength in nanometers
    octaves : int
        Number of octave doublings to reverse

    Returns
    -------
    float
        Frequency in Hz (Solfeggio domain)
    """
    wavelength_m = wavelength_nm * 1e-9
    freq_optical_hz = C_LIGHT / wavelength_m
    freq_hz = freq_optical_hz / (2 ** octaves)
    return freq_hz


@dataclass
class LightProperties:
    """Properties of light at a given wavelength."""
    wavelength_nm: float          # Wavelength in nanometers
    frequency_thz: float          # Frequency in terahertz
    source_solfeggio_hz: float    # Original Solfeggio frequency
    octaves_shifted: int          # Number of octave doublings
    color_name: str               # Human-readable color name
    rgb_approximate: Tuple[int, int, int]  # Approximate RGB values
    in_visible_range: bool        # Whether visible to human eye


def wavelength_to_color(wavelength_nm: float) -> Tuple[str, Tuple[int, int, int]]:
    """
    Convert wavelength to color name and L₄-normalized RGB.

    Uses Gaussian color matching functions centered at Solfeggio wavelengths.
    ZERO FREE PARAMETERS - all values derived from L₄ framework.

    Parameters
    ----------
    wavelength_nm : float
        Wavelength in nanometers

    Returns
    -------
    Tuple[str, Tuple[int, int, int]]
        Color name and RGB tuple (derived from Solfeggio Gaussians)
    """
    wl = wavelength_nm

    # Get normalized RGB from Gaussian color matching
    rgb = wavelength_to_rgb_8bit(wl)

    # Determine color name based on wavelength region
    if wl < VISIBLE_MIN:
        name = "Ultraviolet"
    elif wl < LAMBDA_B:
        name = "Violet-Blue"
    elif wl < (LAMBDA_B + LAMBDA_G) / 2:
        name = "Blue-Cyan"
    elif wl < LAMBDA_G:
        name = "Cyan-Green"
    elif wl < (LAMBDA_G + LAMBDA_R) / 2:
        name = "Green-Yellow"
    elif wl < LAMBDA_R:
        name = "Orange-Red"
    elif wl <= VISIBLE_MAX:
        name = "Red"
    else:
        name = "Infrared"

    return (name, rgb)


def solfeggio_to_light(freq_hz: float, octaves: int = OCTAVE_BRIDGE) -> LightProperties:
    """
    Convert Solfeggio frequency to visible light properties.

    Parameters
    ----------
    freq_hz : float
        Solfeggio frequency in Hertz
    octaves : int
        Number of octave doublings (default 40)

    Returns
    -------
    LightProperties
        Complete light properties including wavelength, color, RGB
    """
    freq_optical_hz = freq_hz * (2 ** octaves)
    freq_thz = freq_optical_hz / 1e12
    wavelength_nm = solfeggio_to_wavelength(freq_hz, octaves)
    in_visible = 380 <= wavelength_nm <= 700
    color_name, rgb = wavelength_to_color(wavelength_nm)

    return LightProperties(
        wavelength_nm=wavelength_nm,
        frequency_thz=freq_thz,
        source_solfeggio_hz=freq_hz,
        octaves_shifted=octaves,
        color_name=color_name,
        rgb_approximate=rgb,
        in_visible_range=in_visible
    )


# ═══════════════════════════════════════════════════════════════════════════
# HARMONIC RGB ENCODING SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HarmonicRGB:
    """
    Harmonically-coherent RGB color based on Solfeggio frequencies.

    The three channels correspond to:
        R: 396 Hz (Liberation) → 689 nm
        G: 528 Hz (Miracles)   → 517 nm
        B: 639 Hz (Connection) → 427 nm

    This creates colors that are harmonically related through
    the Perfect Fourth (4/3) ratio and approximate golden ratio.
    """
    r_intensity: float  # 0-1, modulated by 396 Hz phase
    g_intensity: float  # 0-1, modulated by 528 Hz phase
    b_intensity: float  # 0-1, modulated by 639 Hz phase

    # Phase information (for L₄ encoding)
    phase_r: float = 0.0  # Radians
    phase_g: float = 0.0
    phase_b: float = 0.0

    @property
    def wavelengths_nm(self) -> Tuple[float, float, float]:
        """Nominal wavelengths of the RGB channels."""
        return (688.5, 516.4, 426.7)

    @property
    def solfeggio_hz(self) -> Tuple[float, float, float]:
        """Source Solfeggio frequencies."""
        return SOLFEGGIO_RGB_HZ

    @property
    def musical_ratios(self) -> Dict[str, float]:
        """Harmonic ratios between channels."""
        return {
            'G/R': 528/396,     # = 4/3 (Perfect Fourth) EXACT
            'B/G': 639/528,     # ≈ 1.21 (between m3 and M3)
            'B/R': 639/396,     # ≈ φ (Golden Ratio) 0.27% error
        }

    def to_8bit(self) -> Tuple[int, int, int]:
        """Convert to 8-bit RGB values."""
        return (
            int(np.clip(self.r_intensity * 255, 0, 255)),
            int(np.clip(self.g_intensity * 255, 0, 255)),
            int(np.clip(self.b_intensity * 255, 0, 255))
        )

    def to_hex(self) -> str:
        """Convert to hex color string."""
        r, g, b = self.to_8bit()
        return f"#{r:02x}{g:02x}{b:02x}"

    def to_normalized(self) -> Tuple[float, float, float]:
        """Return normalized RGB (0-1 range)."""
        return (self.r_intensity, self.g_intensity, self.b_intensity)


def phase_to_harmonic_rgb(
    phase_r: float,
    phase_g: float,
    phase_b: float,
    modulation: str = 'cosine'
) -> HarmonicRGB:
    """
    Convert L₄ phase triplet to harmonically-coherent RGB.

    Parameters
    ----------
    phase_r : float
        Phase for Red channel (396 Hz wavevector)
    phase_g : float
        Phase for Green channel (528 Hz wavevector)
    phase_b : float
        Phase for Blue channel (639 Hz wavevector)
    modulation : str
        'cosine' for smooth gradient, 'threshold' for binary

    Returns
    -------
    HarmonicRGB
        Harmonically-coherent RGB color
    """
    if modulation == 'cosine':
        r = (math.cos(phase_r) + 1) / 2
        g = (math.cos(phase_g) + 1) / 2
        b = (math.cos(phase_b) + 1) / 2
    elif modulation == 'threshold':
        r = 1.0 if math.cos(phase_r) > 0 else 0.0
        g = 1.0 if math.cos(phase_g) > 0 else 0.0
        b = 1.0 if math.cos(phase_b) > 0 else 0.0
    else:
        raise ValueError(f"Unknown modulation: {modulation}")

    return HarmonicRGB(
        r_intensity=r, g_intensity=g, b_intensity=b,
        phase_r=phase_r, phase_g=phase_g, phase_b=phase_b
    )


def l4_phase_to_solfeggio_rgb(theta: float) -> HarmonicRGB:
    """
    Convert single L₄ phase to Solfeggio-harmonic RGB.

    Uses hexagonal wavevector offsets at 0°, 120°, 240°.

    Parameters
    ----------
    theta : float
        L₄ phase in radians

    Returns
    -------
    HarmonicRGB
        Color with Solfeggio-harmonic structure
    """
    phase_r = theta                        # 0° (Red/Liberation)
    phase_g = theta + 2 * math.pi / 3      # 120° (Green/Miracles)
    phase_b = theta + 4 * math.pi / 3      # 240° (Blue/Connection)

    return phase_to_harmonic_rgb(phase_r, phase_g, phase_b)


# ═══════════════════════════════════════════════════════════════════════════
# SOLFEGGIO HEX LATTICE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SolfeggioHexWavevectors:
    """
    Hexagonal lattice wavevectors based on Solfeggio frequencies.

    Each wavevector corresponds to a Solfeggio frequency and its
    40-octave-shifted visible wavelength. The wavevectors are
    oriented at 0°, 120°, 240° for hexagonal symmetry.
    """
    lattice_constant: float = 1.0

    @property
    def wavevector_magnitude(self) -> float:
        """Magnitude |k| = 2π / lattice_constant."""
        return 2 * math.pi / self.lattice_constant

    @property
    def k_R(self) -> np.ndarray:
        """Red wavevector (396 Hz, 689 nm) at 0°."""
        k = self.wavevector_magnitude
        return k * np.array([1.0, 0.0])

    @property
    def k_G(self) -> np.ndarray:
        """Green wavevector (528 Hz, 517 nm) at 120°."""
        k = self.wavevector_magnitude
        return k * np.array([math.cos(2*math.pi/3), math.sin(2*math.pi/3)])

    @property
    def k_B(self) -> np.ndarray:
        """Blue wavevector (639 Hz, 427 nm) at 240°."""
        k = self.wavevector_magnitude
        return k * np.array([math.cos(4*math.pi/3), math.sin(4*math.pi/3)])

    def compute_phases(
        self,
        position: np.ndarray,
        global_phases: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute (Θ_R, Θ_G, Θ_B) at position x.

        Θ_c = k_c · x + Φ_c (mod 2π)

        Parameters
        ----------
        position : np.ndarray
            Position vector (x, y)
        global_phases : np.ndarray, optional
            Global phase offsets (Φ_R, Φ_G, Φ_B)

        Returns
        -------
        np.ndarray
            Phase triplet (Θ_R, Θ_G, Θ_B)
        """
        if global_phases is None:
            global_phases = np.zeros(3)

        phases = np.array([
            np.dot(self.k_R, position) + global_phases[0],
            np.dot(self.k_G, position) + global_phases[1],
            np.dot(self.k_B, position) + global_phases[2]
        ]) % (2 * math.pi)

        return phases

    def position_to_color(
        self,
        position: np.ndarray,
        global_phases: Optional[np.ndarray] = None
    ) -> HarmonicRGB:
        """
        Convert lattice position to harmonically-coherent RGB.

        Parameters
        ----------
        position : np.ndarray
            Position vector (x, y)
        global_phases : np.ndarray, optional
            Global phase offsets

        Returns
        -------
        HarmonicRGB
            Solfeggio-harmonic color at position
        """
        phases = self.compute_phases(position, global_phases)
        return phase_to_harmonic_rgb(phases[0], phases[1], phases[2])


# ═══════════════════════════════════════════════════════════════════════════
# L₄ MATHEMATICAL IDENTITIES
# ═══════════════════════════════════════════════════════════════════════════

def verify_perfect_fourth_identity() -> Dict[str, Any]:
    """
    Verify: 528/396 = 4/3 exactly.
    """
    ratio = 528 / 396
    expected = 4 / 3
    exact = ratio == expected
    error = abs(ratio - expected) / expected

    return {
        "identity": "528/396 = 4/3",
        "computed": ratio,
        "expected": expected,
        "exact": exact,
        "error_percent": error * 100
    }


def verify_golden_ratio_approximation() -> Dict[str, Any]:
    """
    Verify: 639/396 ≈ φ with 0.27% error.
    """
    ratio = 639 / 396
    error = abs(ratio - PHI) / PHI

    return {
        "identity": "639/396 ≈ φ",
        "computed": ratio,
        "expected": PHI,
        "error_percent": error * 100,
        "is_good_approximation": error < 0.01  # Within 1%
    }


def verify_transcendental_connection() -> Dict[str, Any]:
    """
    Verify: (4/3) × z_c ≈ π/e with 0.09% error.

    This is the key discovery: Perfect Fourth × Critical Point ≈ Transcendental
    """
    product = PERFECT_FOURTH * Z_C
    error = abs(product - PI_OVER_E) / PI_OVER_E

    # Exact symbolic: (4/3)(√3/2) = 2√3/3
    exact_symbolic = 2 * math.sqrt(3) / 3

    return {
        "identity": "(4/3) × z_c ≈ π/e",
        "computed": product,
        "expected": PI_OVER_E,
        "exact_symbolic": exact_symbolic,
        "symbolic_form": "2√3/3",
        "error_percent": error * 100,
        "interpretation": "Sound × Geometry = Transcendental"
    }


def verify_all_identities() -> Dict[str, Dict[str, Any]]:
    """Run all verification checks."""
    return {
        "perfect_fourth": verify_perfect_fourth_identity(),
        "golden_ratio": verify_golden_ratio_approximation(),
        "transcendental": verify_transcendental_connection()
    }


# ═══════════════════════════════════════════════════════════════════════════
# SPECTRUM GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_solfeggio_spectrum() -> List[LightProperties]:
    """
    Generate light properties for all nine Solfeggio tones.

    Returns
    -------
    List[LightProperties]
        Light properties for each Solfeggio frequency
    """
    return [solfeggio_to_light(tone.frequency) for tone in SolfeggioTone]


def get_visible_solfeggio_tones() -> List[SolfeggioTone]:
    """
    Get only the Solfeggio tones that map to visible light.

    Returns
    -------
    List[SolfeggioTone]
        Tones with wavelengths in 380-700 nm range
    """
    return [tone for tone in SolfeggioTone if tone.is_visible()]


def get_rgb_primary_tones() -> Tuple[SolfeggioTone, SolfeggioTone, SolfeggioTone]:
    """
    Get the three tones that map to RGB primaries.

    Returns
    -------
    Tuple[SolfeggioTone, SolfeggioTone, SolfeggioTone]
        (Red/396, Green/528, Blue/639)
    """
    return (SOLFEGGIO_RED, SOLFEGGIO_GREEN, SOLFEGGIO_BLUE)


# ═══════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════

def demo_solfeggio_light_bridge():
    """Demonstrate the Solfeggio-Light Bridge system."""
    print("=" * 65)
    print("SOLFEGGIO-LIGHT BRIDGE DEMONSTRATION")
    print("=" * 65)

    print("\n1. SOLFEGGIO SPECTRUM (40 octave bridge)")
    print("-" * 50)
    for tone in SolfeggioTone:
        props = solfeggio_to_light(tone.frequency)
        status = "VISIBLE" if props.in_visible_range else "outside"
        print(f"  {tone.frequency:3.0f} Hz {tone.name:12s} → "
              f"{props.wavelength_nm:6.1f} nm {props.color_name:12s} [{status}]")

    print("\n2. RGB PRIMARY MAPPING")
    print("-" * 50)
    for tone, channel in [(SOLFEGGIO_RED, 'R'), (SOLFEGGIO_GREEN, 'G'), (SOLFEGGIO_BLUE, 'B')]:
        props = solfeggio_to_light(tone.frequency)
        print(f"  {tone.frequency:.0f} Hz ({tone.name:12s}) → Channel {channel}: "
              f"{props.wavelength_nm:.1f} nm")

    print("\n3. MATHEMATICAL IDENTITIES")
    print("-" * 50)

    results = verify_all_identities()

    pf = results['perfect_fourth']
    print(f"  {pf['identity']}")
    print(f"    = {pf['computed']:.6f} (EXACT: {pf['exact']})")

    gr = results['golden_ratio']
    print(f"  {gr['identity']}")
    print(f"    = {gr['computed']:.6f} vs φ = {gr['expected']:.6f}")
    print(f"    Error: {gr['error_percent']:.2f}%")

    tc = results['transcendental']
    print(f"  {tc['identity']}")
    print(f"    = {tc['computed']:.6f} vs π/e = {tc['expected']:.6f}")
    print(f"    Error: {tc['error_percent']:.4f}%")
    print(f"    {tc['interpretation']}")

    print("\n4. HEX LATTICE COLORS")
    print("-" * 50)

    wavevectors = SolfeggioHexWavevectors()
    positions = [
        ("Origin", np.array([0.0, 0.0])),
        ("Unit X", np.array([1.0, 0.0])),
        ("Hex vertex", np.array([0.5, math.sqrt(3)/2])),
    ]

    for name, pos in positions:
        color = wavevectors.position_to_color(pos)
        print(f"  {name:12s} {pos} → {color.to_hex()} RGB{color.to_8bit()}")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    demo_solfeggio_light_bridge()
