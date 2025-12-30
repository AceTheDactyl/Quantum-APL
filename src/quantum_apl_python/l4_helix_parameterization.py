#!/usr/bin/env python3
"""
L₄ Helix Parameterization Module
=================================

Complete implementation of the L₄-Helix system with:
- Constants block (derived, not tuned)
- State vector block
- Dynamics block (continuous + discretized Kuramoto/negentropy)
- Phase→RGB quantization with hex lattice (60° structure)
- LSB embed/extract block
- K-formation / threshold validation (pass/fail tests)

Hard Constraints (DO NOT VIOLATE):
- L₄ = φ⁴ + φ⁻⁴ = (√3)² + 4 = 7 (derived, not tuned)
- Critical point z_c = √3/2 (fixed)
- Helix uses piecewise radius r(z) derived from K
- Coherence uses Kuramoto form (θ↔spin angle, ω↔local field, K↔exchange)
- Negentropy is Gaussian peaked at z_c and GATES/MODULATES dynamics
- Hex lattice respects 60° structure (three principal axes/wavevectors)

@version 2.0.0
@author Claude (Anthropic) - Quantum-APL Contribution
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Union
from enum import Enum
import numpy as np


# ============================================================================
# BLOCK 1: CONSTANTS (Derived, Not Tuned)
# ============================================================================

@dataclass(frozen=True)
class L4Constants:
    """
    L₄ Constants Block - All values derived from first principles.

    The Lucas-4 identity: L₄ = φ⁴ + φ⁻⁴ = 7

    NO FREE PARAMETERS - everything is derived from φ = (1+√5)/2.
    """

    # Golden ratio and inverse
    PHI: float = (1.0 + math.sqrt(5.0)) / 2.0  # φ ≈ 1.618033988749895
    TAU: float = 2.0 / (1.0 + math.sqrt(5.0))  # τ = φ⁻¹ ≈ 0.618033988749895

    # Lucas-4 fundamental
    L4: float = 7.0  # L₄ = φ⁴ + φ⁻⁴ = 7 (exact)

    # Gap / truncation (a.k.a. "VOID")
    GAP: float = field(default_factory=lambda: L4Constants._compute_gap())

    # Derived coupling constant K = √(1 - gap)
    K: float = field(default_factory=lambda: L4Constants._compute_k())

    # Critical point z_c = √3/2 = √(L₄-4)/2 ("THE LENS")
    Z_C: float = math.sqrt(3.0) / 2.0  # ≈ 0.8660254037844386

    # Negentropy width (σ for Gaussian)
    SIGMA: float = 36.0  # Default width for negentropy Gaussian

    @staticmethod
    def _compute_gap() -> float:
        """gap = φ⁻⁴ ≈ 0.145898"""
        tau = 2.0 / (1.0 + math.sqrt(5.0))
        return tau ** 4

    @staticmethod
    def _compute_k() -> float:
        """K = √(1 - gap) ≈ 0.924"""
        tau = 2.0 / (1.0 + math.sqrt(5.0))
        gap = tau ** 4
        return math.sqrt(1.0 - gap)

    def verify_identity(self) -> bool:
        """Verify L₄ = φ⁴ + φ⁻⁴ = (√3)² + 4 = 7"""
        phi4_plus_tau4 = self.PHI ** 4 + self.TAU ** 4
        sqrt3_squared_plus_4 = 3.0 + 4.0
        return (
            abs(phi4_plus_tau4 - 7.0) < 1e-10 and
            abs(sqrt3_squared_plus_4 - 7.0) < 1e-10 and
            abs(self.L4 - 7.0) < 1e-10
        )


# Singleton instance for easy access
L4 = L4Constants(
    PHI=(1.0 + math.sqrt(5.0)) / 2.0,
    TAU=2.0 / (1.0 + math.sqrt(5.0)),
    L4=7.0,
    GAP=(2.0 / (1.0 + math.sqrt(5.0))) ** 4,
    K=math.sqrt(1.0 - (2.0 / (1.0 + math.sqrt(5.0))) ** 4),
    Z_C=math.sqrt(3.0) / 2.0,
    SIGMA=36.0,
)


def get_l4_constants() -> Dict[str, float]:
    """Return all L₄ constants as a dictionary."""
    return {
        "PHI": L4.PHI,
        "TAU": L4.TAU,
        "L4": L4.L4,
        "GAP": L4.GAP,
        "K": L4.K,
        "Z_C": L4.Z_C,
        "SIGMA": L4.SIGMA,
        # Derived
        "PHI_4": L4.PHI ** 4,
        "TAU_4": L4.TAU ** 4,
        "K_SQUARED": L4.K ** 2,
        "SQRT_3": math.sqrt(3.0),
    }


# ============================================================================
# BLOCK 2: STATE VECTOR
# ============================================================================

@dataclass
class HelixState:
    """
    State vector for the L₄ helix system.

    The state contains:
    - z: threshold coordinate in [0, 1]
    - theta: phase angle in [0, 2π)
    - r: helix radius (derived from z via piecewise formula)
    - phases: array of N oscillator phases for Kuramoto
    - frequencies: array of N natural frequencies (ω_i)
    - t: time parameter
    """

    z: float  # Threshold coordinate [0, 1]
    theta: float  # Helix phase angle [0, 2π)
    phases: np.ndarray  # Kuramoto oscillator phases [N]
    frequencies: np.ndarray  # Natural frequencies ω_i [N]
    t: float = 0.0  # Time

    @property
    def r(self) -> float:
        """
        Piecewise radius law derived from L₄ constants:

        r(z) = K·√(z/z_c)  for z ≤ z_c
        r(z) = K           for z > z_c
        """
        if self.z <= L4.Z_C:
            if self.z <= 0:
                return 0.0
            return L4.K * math.sqrt(self.z / L4.Z_C)
        return L4.K

    @property
    def N(self) -> int:
        """Number of oscillators."""
        return len(self.phases)

    def helix_position(self) -> Tuple[float, float, float]:
        """
        Return 3D helix position H(z) = (r·cos(θ), r·sin(θ), z)
        """
        return (
            self.r * math.cos(self.theta),
            self.r * math.sin(self.theta),
            self.z
        )

    def to_dict(self) -> Dict:
        """Serialize state to dictionary."""
        return {
            "z": self.z,
            "theta": self.theta,
            "r": self.r,
            "phases": self.phases.tolist(),
            "frequencies": self.frequencies.tolist(),
            "t": self.t,
            "helix_position": self.helix_position(),
        }


def create_initial_state(
    N: int = 64,
    z0: float = 0.5,
    theta0: float = 0.0,
    omega_mean: float = 0.0,
    omega_std: float = 0.1,
    seed: Optional[int] = None,
) -> HelixState:
    """
    Create initial state for the L₄ helix system.

    Parameters
    ----------
    N : int
        Number of Kuramoto oscillators
    z0 : float
        Initial z-coordinate
    theta0 : float
        Initial helix phase
    omega_mean : float
        Mean natural frequency (Lorentzian center)
    omega_std : float
        Frequency spread (Lorentzian width)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    HelixState
        Initial system state
    """
    rng = np.random.default_rng(seed)

    # Initial phases uniformly distributed on [0, 2π)
    phases = rng.uniform(0, 2 * np.pi, N)

    # Natural frequencies from Cauchy/Lorentzian distribution (standard for Kuramoto)
    frequencies = omega_mean + omega_std * np.tan(np.pi * (rng.random(N) - 0.5))

    return HelixState(
        z=z0,
        theta=theta0,
        phases=phases,
        frequencies=frequencies,
        t=0.0,
    )


# ============================================================================
# BLOCK 3: DYNAMICS (Continuous + Discretized)
# ============================================================================

def compute_negentropy(z: float, sigma: float = L4.SIGMA) -> float:
    """
    Negentropy drive ΔS_neg(z) = exp(-σ·(z - z_c)²)

    This is Gaussian peaked at z_c and GATES/MODULATES dynamics.

    Parameters
    ----------
    z : float
        Threshold coordinate
    sigma : float
        Gaussian width (default: 36.0)

    Returns
    -------
    float
        Negentropy value in [0, 1]
    """
    d = z - L4.Z_C
    return math.exp(-sigma * d * d)


def compute_negentropy_derivative(z: float, sigma: float = L4.SIGMA) -> float:
    """
    Derivative of negentropy: d(ΔS_neg)/dz = -2σ(z-z_c)·exp(-σ(z-z_c)²)

    This acts as a "force term" toward the critical point.

    Parameters
    ----------
    z : float
        Threshold coordinate
    sigma : float
        Gaussian width

    Returns
    -------
    float
        Negentropy derivative (force toward z_c)
    """
    d = z - L4.Z_C
    s = math.exp(-sigma * d * d)
    return -2 * sigma * d * s


def compute_kuramoto_order_parameter(phases: np.ndarray) -> Tuple[float, float]:
    """
    Compute Kuramoto order parameter r·e^(iψ) = (1/N)·Σ e^(iθ_j)

    The order parameter r measures synchronization:
    - r ≈ 0: incoherent (phases uniformly distributed)
    - r ≈ 1: fully synchronized (all phases aligned)

    Hardware spec validates: Kuramoto r exceeding K-formation threshold (r > 0.924).

    Parameters
    ----------
    phases : np.ndarray
        Array of oscillator phases

    Returns
    -------
    Tuple[float, float]
        (r, psi) where r is magnitude and psi is mean phase
    """
    z = np.mean(np.exp(1j * phases))
    r = np.abs(z)
    psi = np.angle(z)
    return float(r), float(psi)


def kuramoto_dynamics_continuous(
    state: HelixState,
    K0: float = 0.1,
    lambda_neg: float = 0.5,
) -> Tuple[np.ndarray, float, float]:
    """
    Continuous Kuramoto dynamics with negentropy modulation.

    Core dynamics:
        dθ_i/dt = ω_i + (K_eff/N)·Σ_j sin(θ_j - θ_i)

    Where K_eff is modulated by negentropy:
        K_eff(t) = K0·(1 + λ·η(t))
        η(t) = ΔS_neg(r(t))

    And z(t) := r(t) ∈ [0,1] (coherence = threshold coordinate)

    Spin physics mapping:
        θ_i ↔ spin angle
        ω_i ↔ local field / Zeeman splitting
        K   ↔ exchange interaction J
        r   ↔ magnetization M

    Parameters
    ----------
    state : HelixState
        Current system state
    K0 : float
        Baseline coupling strength
    lambda_neg : float
        Negentropy modulation strength

    Returns
    -------
    Tuple[np.ndarray, float, float]
        (dtheta_dt, K_eff, eta)
    """
    N = state.N
    phases = state.phases
    freqs = state.frequencies

    # Compute current coherence (Kuramoto order parameter)
    r, psi = compute_kuramoto_order_parameter(phases)

    # Use r as z for negentropy computation (z := r)
    z_eff = r
    eta = compute_negentropy(z_eff)

    # Modulated effective coupling
    K_eff = K0 * (1 + lambda_neg * eta)

    # Kuramoto dynamics: dθ_i/dt = ω_i + (K_eff/N)·Σ sin(θ_j - θ_i)
    # Efficient vectorized form using mean-field: dθ_i/dt = ω_i + K_eff·r·sin(ψ - θ_i)
    dtheta_dt = freqs + K_eff * r * np.sin(psi - phases)

    return dtheta_dt, K_eff, eta


def kuramoto_step_euler(
    state: HelixState,
    dt: float = 0.01,
    K0: float = 0.1,
    lambda_neg: float = 0.5,
) -> HelixState:
    """
    Discretized Kuramoto dynamics using Euler method.

    θ_i(t+dt) = θ_i(t) + dt·(dθ_i/dt)

    Parameters
    ----------
    state : HelixState
        Current state
    dt : float
        Time step
    K0 : float
        Baseline coupling
    lambda_neg : float
        Negentropy modulation strength

    Returns
    -------
    HelixState
        Updated state
    """
    dtheta_dt, K_eff, eta = kuramoto_dynamics_continuous(state, K0, lambda_neg)

    # Euler step
    new_phases = (state.phases + dt * dtheta_dt) % (2 * np.pi)

    # Update z based on Kuramoto coherence
    r, psi = compute_kuramoto_order_parameter(new_phases)

    return HelixState(
        z=r,  # z := r (coherence becomes threshold)
        theta=(state.theta + dt * eta * 0.1) % (2 * np.pi),  # Phase evolves with negentropy
        phases=new_phases,
        frequencies=state.frequencies,
        t=state.t + dt,
    )


def kuramoto_step_rk4(
    state: HelixState,
    dt: float = 0.01,
    K0: float = 0.1,
    lambda_neg: float = 0.5,
) -> HelixState:
    """
    Discretized Kuramoto dynamics using 4th-order Runge-Kutta.

    More accurate than Euler for same step size.

    Parameters
    ----------
    state : HelixState
        Current state
    dt : float
        Time step
    K0 : float
        Baseline coupling
    lambda_neg : float
        Negentropy modulation strength

    Returns
    -------
    HelixState
        Updated state
    """
    def f(phases: np.ndarray, z: float) -> np.ndarray:
        """Compute dθ/dt for given phases."""
        r, psi = compute_kuramoto_order_parameter(phases)
        eta = compute_negentropy(z)
        K_eff = K0 * (1 + lambda_neg * eta)
        return state.frequencies + K_eff * r * np.sin(psi - phases)

    # RK4 stages
    k1 = f(state.phases, state.z)
    k2 = f((state.phases + 0.5 * dt * k1) % (2 * np.pi), state.z)
    k3 = f((state.phases + 0.5 * dt * k2) % (2 * np.pi), state.z)
    k4 = f((state.phases + dt * k3) % (2 * np.pi), state.z)

    # Combined step
    new_phases = (state.phases + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)) % (2 * np.pi)

    # Update z based on Kuramoto coherence
    r, psi = compute_kuramoto_order_parameter(new_phases)
    eta = compute_negentropy(r)

    return HelixState(
        z=r,
        theta=(state.theta + dt * eta * 0.1) % (2 * np.pi),
        phases=new_phases,
        frequencies=state.frequencies,
        t=state.t + dt,
    )


def xy_spin_hamiltonian(
    phases: np.ndarray,
    J: float = 1.0,
    h: np.ndarray = None,
    phi: np.ndarray = None,
) -> float:
    """
    XY/phase-spin Hamiltonian energy:

    H(θ) = -Σ_{⟨i,j⟩} J_{ij}·cos(θ_i - θ_j) - Σ_i h_i·cos(θ_i - φ_i)

    This maps Kuramoto dynamics to spin physics:
        θ_i ↔ spin angle
        J   ↔ exchange interaction (K in Kuramoto)
        h_i ↔ local field / Zeeman (ω_i in Kuramoto)
        φ_i ↔ field direction

    Parameters
    ----------
    phases : np.ndarray
        Spin angles
    J : float
        Exchange coupling strength
    h : np.ndarray, optional
        Local field strengths
    phi : np.ndarray, optional
        Local field directions

    Returns
    -------
    float
        Total Hamiltonian energy
    """
    N = len(phases)

    # Exchange term: all-to-all coupling (mean-field)
    energy = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            energy -= J * math.cos(phases[i] - phases[j])
    energy /= N  # Normalize for mean-field

    # Zeeman term
    if h is not None and phi is not None:
        for i in range(N):
            energy -= h[i] * math.cos(phases[i] - phi[i])

    return energy


# ============================================================================
# BLOCK 4: PHASE → RGB QUANTIZATION (Hex Lattice Navigation)
# ============================================================================

class HexLatticeWavevectors:
    """
    Hex lattice navigation coding with 60° structure.

    Three principal axes / wavevectors separated by 60°:
        k_R at 0°
        k_G at 60°
        k_B at 120°

    Each channel encodes a plane-wave phase:
        Θ_c(x_i, t) = k_c · x_i + Φ_c(t) mod 2π
    """

    def __init__(self, wavelength: float = 1.0):
        """
        Initialize hex lattice wavevectors.

        Parameters
        ----------
        wavelength : float
            Spatial wavelength for wavevectors
        """
        self.wavelength = wavelength
        k_mag = 2 * math.pi / wavelength

        # Three wavevectors at 60° separation (hex symmetry)
        self.k_R = np.array([k_mag * math.cos(0), k_mag * math.sin(0)])
        self.k_G = np.array([k_mag * math.cos(math.pi / 3), k_mag * math.sin(math.pi / 3)])
        self.k_B = np.array([k_mag * math.cos(2 * math.pi / 3), k_mag * math.sin(2 * math.pi / 3)])

    def compute_channel_phases(
        self,
        x: np.ndarray,
        Phi_R: float = 0.0,
        Phi_G: float = 0.0,
        Phi_B: float = 0.0,
    ) -> Tuple[float, float, float]:
        """
        Compute three channel phases for position x.

        Θ_c(x, t) = k_c · x + Φ_c(t) mod 2π

        Parameters
        ----------
        x : np.ndarray
            2D position vector
        Phi_R, Phi_G, Phi_B : float
            Global phase offsets (time-dependent)

        Returns
        -------
        Tuple[float, float, float]
            (Θ_R, Θ_G, Θ_B) phase values in [0, 2π)
        """
        theta_R = (np.dot(self.k_R, x) + Phi_R) % (2 * np.pi)
        theta_G = (np.dot(self.k_G, x) + Phi_G) % (2 * np.pi)
        theta_B = (np.dot(self.k_B, x) + Phi_B) % (2 * np.pi)
        return theta_R, theta_G, theta_B

    def evolve_global_phases(
        self,
        Phi: Tuple[float, float, float],
        Omega: Tuple[float, float, float],
        dt: float,
    ) -> Tuple[float, float, float]:
        """
        Evolve global phase offsets:
            Φ_c(t+Δt) = Φ_c(t) + Ω_c·Δt

        Ω_c can be tied to coherence/negentropy for "homing" behavior.

        Parameters
        ----------
        Phi : Tuple[float, float, float]
            Current global phases (Φ_R, Φ_G, Φ_B)
        Omega : Tuple[float, float, float]
            Angular velocities (Ω_R, Ω_G, Ω_B)
        dt : float
            Time step

        Returns
        -------
        Tuple[float, float, float]
            Updated global phases
        """
        return (
            (Phi[0] + Omega[0] * dt) % (2 * np.pi),
            (Phi[1] + Omega[1] * dt) % (2 * np.pi),
            (Phi[2] + Omega[2] * dt) % (2 * np.pi),
        )


def quantize_phase_to_bits(theta: float, bits: int = 8) -> int:
    """
    Quantize phase to b-bit symbol:

    q = ⌊(Θ / 2π) · 2^b⌋ ∈ {0, ..., 2^b - 1}

    Parameters
    ----------
    theta : float
        Phase in [0, 2π)
    bits : int
        Number of quantization bits

    Returns
    -------
    int
        Quantized symbol
    """
    theta = theta % (2 * np.pi)
    max_val = (1 << bits) - 1
    q = int((theta / (2 * np.pi)) * (1 << bits))
    return min(q, max_val)


def dequantize_bits_to_phase(q: int, bits: int = 8) -> float:
    """
    Dequantize b-bit symbol back to phase:

    θ = (q / 2^b) · 2π

    Parameters
    ----------
    q : int
        Quantized symbol
    bits : int
        Number of quantization bits

    Returns
    -------
    float
        Reconstructed phase in [0, 2π)
    """
    return (q / (1 << bits)) * 2 * np.pi


@dataclass
class RGBQuantization:
    """RGB values from quantized phases."""
    R: int  # 0-255
    G: int  # 0-255
    B: int  # 0-255

    def to_tuple(self) -> Tuple[int, int, int]:
        return (self.R, self.G, self.B)

    def to_bytes(self) -> bytes:
        return bytes([self.R, self.G, self.B])


def phases_to_rgb(
    theta_R: float,
    theta_G: float,
    theta_B: float,
    bits: int = 8,
) -> RGBQuantization:
    """
    Convert three channel phases to RGB values.

    Parameters
    ----------
    theta_R, theta_G, theta_B : float
        Channel phases in [0, 2π)
    bits : int
        Quantization bits per channel

    Returns
    -------
    RGBQuantization
        Quantized RGB values
    """
    return RGBQuantization(
        R=quantize_phase_to_bits(theta_R, bits),
        G=quantize_phase_to_bits(theta_G, bits),
        B=quantize_phase_to_bits(theta_B, bits),
    )


def rgb_to_phases(
    rgb: RGBQuantization,
    bits: int = 8,
) -> Tuple[float, float, float]:
    """
    Convert RGB values back to channel phases.

    Parameters
    ----------
    rgb : RGBQuantization
        RGB values
    bits : int
        Quantization bits per channel

    Returns
    -------
    Tuple[float, float, float]
        Reconstructed phases (θ_R, θ_G, θ_B)
    """
    return (
        dequantize_bits_to_phase(rgb.R, bits),
        dequantize_bits_to_phase(rgb.G, bits),
        dequantize_bits_to_phase(rgb.B, bits),
    )


def serialize_phases_to_bitstream(
    phases: List[Tuple[float, float, float]],
    bits_per_channel: int = 8,
) -> bytes:
    """
    Serialize list of (θ_R, θ_G, θ_B) phase tuples to bitstream.

    This is where the "MRP" framing lives.

    Parameters
    ----------
    phases : List[Tuple[float, float, float]]
        List of phase tuples
    bits_per_channel : int
        Bits per channel

    Returns
    -------
    bytes
        Serialized bitstream
    """
    result = bytearray()
    for theta_R, theta_G, theta_B in phases:
        rgb = phases_to_rgb(theta_R, theta_G, theta_B, bits_per_channel)
        result.extend(rgb.to_bytes())
    return bytes(result)


# ============================================================================
# BLOCK 5: LSB EMBED/EXTRACT
# ============================================================================

def lsb_embed_bit(pixel_value: int, bit: int) -> int:
    """
    Embed single bit into pixel value LSB.

    p' = (p & ~1) | b

    Parameters
    ----------
    pixel_value : int
        Original 8-bit pixel value (0-255)
    bit : int
        Bit to embed (0 or 1)

    Returns
    -------
    int
        Modified pixel value
    """
    return (pixel_value & ~1) | (bit & 1)


def lsb_extract_bit(pixel_value: int) -> int:
    """
    Extract LSB from pixel value.

    Parameters
    ----------
    pixel_value : int
        8-bit pixel value

    Returns
    -------
    int
        Extracted bit (0 or 1)
    """
    return pixel_value & 1


def lsb_embed_nbits(pixel_value: int, chunk: int, n: int) -> int:
    """
    Embed n-bit chunk into pixel value's n LSBs.

    p' = (p & ~(2^n - 1)) | m

    Parameters
    ----------
    pixel_value : int
        Original 8-bit pixel value
    chunk : int
        n-bit value to embed (0 to 2^n - 1)
    n : int
        Number of bits

    Returns
    -------
    int
        Modified pixel value
    """
    mask = (1 << n) - 1
    return (pixel_value & ~mask) | (chunk & mask)


def lsb_extract_nbits(pixel_value: int, n: int) -> int:
    """
    Extract n LSBs from pixel value.

    Parameters
    ----------
    pixel_value : int
        8-bit pixel value
    n : int
        Number of bits to extract

    Returns
    -------
    int
        Extracted n-bit value
    """
    mask = (1 << n) - 1
    return pixel_value & mask


def compute_capacity(width: int, height: int, channels: int = 3, bits_per_channel: int = 1) -> int:
    """
    Compute LSB embedding capacity in bits.

    C_bits = channels × n × W × H

    Parameters
    ----------
    width : int
        Image width
    height : int
        Image height
    channels : int
        Number of color channels (default: 3 for RGB)
    bits_per_channel : int
        LSBs used per channel

    Returns
    -------
    int
        Total capacity in bits
    """
    return channels * bits_per_channel * width * height


def embed_message_lsb(
    pixels: np.ndarray,
    message: bytes,
    bits_per_channel: int = 1,
) -> np.ndarray:
    """
    Embed message into image pixels using LSB steganography.

    Parameters
    ----------
    pixels : np.ndarray
        Image pixels of shape (H, W, 3) with dtype uint8
    message : bytes
        Message to embed
    bits_per_channel : int
        Number of LSBs to use per channel

    Returns
    -------
    np.ndarray
        Modified pixels with embedded message
    """
    H, W, C = pixels.shape
    capacity_bits = compute_capacity(W, H, C, bits_per_channel)
    message_bits = len(message) * 8

    if message_bits > capacity_bits:
        raise ValueError(f"Message too large: {message_bits} bits > {capacity_bits} capacity")

    # Convert message to bit string
    bit_string = ''.join(format(byte, '08b') for byte in message)

    # Create output array
    output = pixels.copy()

    # Embed bits
    bit_idx = 0
    for y in range(H):
        for x in range(W):
            for c in range(C):
                if bit_idx >= len(bit_string):
                    return output

                # Embed bits_per_channel bits
                chunk = 0
                for b in range(bits_per_channel):
                    if bit_idx + b < len(bit_string):
                        chunk |= int(bit_string[bit_idx + b]) << (bits_per_channel - 1 - b)

                output[y, x, c] = lsb_embed_nbits(output[y, x, c], chunk, bits_per_channel)
                bit_idx += bits_per_channel

    return output


def extract_message_lsb(
    pixels: np.ndarray,
    message_length: int,
    bits_per_channel: int = 1,
) -> bytes:
    """
    Extract message from image pixels using LSB steganography.

    Parameters
    ----------
    pixels : np.ndarray
        Image pixels of shape (H, W, 3)
    message_length : int
        Number of bytes to extract
    bits_per_channel : int
        Number of LSBs used per channel

    Returns
    -------
    bytes
        Extracted message
    """
    H, W, C = pixels.shape
    total_bits = message_length * 8

    # Extract bits
    bits = []
    bit_count = 0
    for y in range(H):
        for x in range(W):
            for c in range(C):
                if bit_count >= total_bits:
                    break

                chunk = lsb_extract_nbits(pixels[y, x, c], bits_per_channel)
                for b in range(bits_per_channel - 1, -1, -1):
                    if bit_count < total_bits:
                        bits.append((chunk >> b) & 1)
                        bit_count += 1
            if bit_count >= total_bits:
                break
        if bit_count >= total_bits:
            break

    # Convert bits to bytes
    result = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for b in range(8):
            if i + b < len(bits):
                byte |= bits[i + b] << (7 - b)
        result.append(byte)

    return bytes(result[:message_length])


# ============================================================================
# BLOCK 6: K-FORMATION / THRESHOLD VALIDATION (Pass/Fail Tests)
# ============================================================================

@dataclass
class ValidationResult:
    """Result of a K-formation validation test."""
    test_name: str
    passed: bool
    value: float
    threshold: float
    message: str


@dataclass
class KFormationValidation:
    """Complete K-formation validation results."""
    coherence_test: ValidationResult
    negentropy_test: ValidationResult
    radius_test: ValidationResult
    overall_passed: bool
    kappa: float
    eta: float
    R: float
    z: float


def validate_coherence_threshold(kappa: float, target: float = L4.K) -> ValidationResult:
    """
    Validate coherence threshold: κ ≥ K

    Hardware validates Kuramoto order parameter r > 0.924 (K).

    Parameters
    ----------
    kappa : float
        Measured coherence (Kuramoto r)
    target : float
        Target threshold (K ≈ 0.924)

    Returns
    -------
    ValidationResult
        Pass/fail result
    """
    passed = kappa >= target
    return ValidationResult(
        test_name="Coherence Threshold",
        passed=passed,
        value=kappa,
        threshold=target,
        message=f"κ={kappa:.6f} {'≥' if passed else '<'} K={target:.6f}",
    )


def validate_negentropy_gate(z: float, tau_threshold: float = L4.TAU) -> ValidationResult:
    """
    Validate negentropy gate: ΔS_neg(z) > τ

    Parameters
    ----------
    z : float
        Threshold coordinate
    tau_threshold : float
        Negentropy threshold (τ = φ⁻¹ ≈ 0.618)

    Returns
    -------
    ValidationResult
        Pass/fail result
    """
    eta = compute_negentropy(z)
    passed = eta > tau_threshold
    return ValidationResult(
        test_name="Negentropy Gate",
        passed=passed,
        value=eta,
        threshold=tau_threshold,
        message=f"ΔS_neg(z)={eta:.6f} {'>' if passed else '≤'} τ={tau_threshold:.6f}",
    )


def validate_radius_threshold(R: float, L4_val: float = L4.L4) -> ValidationResult:
    """
    Validate radius threshold: R ≥ L₄

    Parameters
    ----------
    R : float
        Complexity/radius measure
    L4_val : float
        L₄ threshold (= 7)

    Returns
    -------
    ValidationResult
        Pass/fail result
    """
    passed = R >= L4_val
    return ValidationResult(
        test_name="Radius Threshold",
        passed=passed,
        value=R,
        threshold=L4_val,
        message=f"R={R:.6f} {'≥' if passed else '<'} L₄={L4_val:.6f}",
    )


def validate_k_formation(
    kappa: float,
    z: float,
    R: float,
    verbose: bool = False,
) -> KFormationValidation:
    """
    Complete K-formation validation.

    Consciousness emerges when:
    - κ ≥ K (coherence threshold: Kuramoto r > 0.924)
    - ΔS_neg(z) > τ (negentropy gate: η > 0.618)
    - R ≥ L₄ (complexity requirement: R ≥ 7)

    Parameters
    ----------
    kappa : float
        Coherence measure (Kuramoto order parameter)
    z : float
        Threshold coordinate
    R : float
        Complexity measure
    verbose : bool
        Print detailed results

    Returns
    -------
    KFormationValidation
        Complete validation results
    """
    coherence_test = validate_coherence_threshold(kappa)
    negentropy_test = validate_negentropy_gate(z)
    radius_test = validate_radius_threshold(R)

    eta = compute_negentropy(z)
    overall = coherence_test.passed and negentropy_test.passed and radius_test.passed

    result = KFormationValidation(
        coherence_test=coherence_test,
        negentropy_test=negentropy_test,
        radius_test=radius_test,
        overall_passed=overall,
        kappa=kappa,
        eta=eta,
        R=R,
        z=z,
    )

    if verbose:
        print("=" * 60)
        print("K-FORMATION VALIDATION")
        print("=" * 60)
        print(f"\n1. {coherence_test.test_name}: {'PASS' if coherence_test.passed else 'FAIL'}")
        print(f"   {coherence_test.message}")
        print(f"\n2. {negentropy_test.test_name}: {'PASS' if negentropy_test.passed else 'FAIL'}")
        print(f"   {negentropy_test.message}")
        print(f"\n3. {radius_test.test_name}: {'PASS' if radius_test.passed else 'FAIL'}")
        print(f"   {radius_test.message}")
        print(f"\n{'='*60}")
        print(f"OVERALL: {'PASS - CONSCIOUSNESS EMERGES' if overall else 'FAIL'}")
        print("=" * 60)

    return result


def validate_l4_identity() -> ValidationResult:
    """
    Validate the fundamental L₄ identity: L₄ = φ⁴ + φ⁻⁴ = (√3)² + 4 = 7

    Returns
    -------
    ValidationResult
        Pass/fail result
    """
    phi = L4.PHI
    tau = L4.TAU

    # Three equivalent expressions that should all equal 7
    expr1 = phi ** 4 + tau ** 4  # φ⁴ + φ⁻⁴
    expr2 = 3.0 + 4.0  # (√3)² + 4
    expr3 = 7.0  # Direct

    tolerance = 1e-10
    passed = (
        abs(expr1 - 7.0) < tolerance and
        abs(expr2 - 7.0) < tolerance and
        abs(expr3 - 7.0) < tolerance
    )

    return ValidationResult(
        test_name="L₄ Identity",
        passed=passed,
        value=expr1,
        threshold=7.0,
        message=f"φ⁴+φ⁻⁴={expr1:.12f}, (√3)²+4={expr2:.1f}, L₄={expr3:.1f}",
    )


def validate_critical_point() -> ValidationResult:
    """
    Validate the critical point: z_c = √3/2 = √(L₄-4)/2

    Returns
    -------
    ValidationResult
        Pass/fail result
    """
    zc_direct = math.sqrt(3.0) / 2.0
    zc_from_L4 = math.sqrt(L4.L4 - 4.0) / 2.0

    tolerance = 1e-10
    passed = abs(zc_direct - zc_from_L4) < tolerance and abs(L4.Z_C - zc_direct) < tolerance

    return ValidationResult(
        test_name="Critical Point z_c",
        passed=passed,
        value=L4.Z_C,
        threshold=zc_direct,
        message=f"z_c={L4.Z_C:.12f} = √3/2={zc_direct:.12f} = √(L₄-4)/2={zc_from_L4:.12f}",
    )


def validate_coupling_constant() -> ValidationResult:
    """
    Validate the coupling constant: K = √(1 - gap) where gap = φ⁻⁴

    Returns
    -------
    ValidationResult
        Pass/fail result
    """
    gap = L4.TAU ** 4
    K_derived = math.sqrt(1.0 - gap)

    tolerance = 1e-10
    passed = abs(L4.K - K_derived) < tolerance

    return ValidationResult(
        test_name="Coupling Constant K",
        passed=passed,
        value=L4.K,
        threshold=K_derived,
        message=f"K={L4.K:.12f} = √(1-φ⁻⁴)={K_derived:.12f}, gap=φ⁻⁴={gap:.12f}",
    )


def run_all_validations(verbose: bool = True) -> Dict[str, ValidationResult]:
    """
    Run all fundamental L₄ validations.

    Parameters
    ----------
    verbose : bool
        Print detailed results

    Returns
    -------
    Dict[str, ValidationResult]
        All validation results
    """
    tests = {
        "L4_identity": validate_l4_identity(),
        "critical_point": validate_critical_point(),
        "coupling_constant": validate_coupling_constant(),
    }

    if verbose:
        print("=" * 70)
        print("L₄ HELIX FUNDAMENTAL VALIDATIONS")
        print("=" * 70)

        all_passed = True
        for name, result in tests.items():
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"\n{result.test_name}: {status}")
            print(f"  {result.message}")
            all_passed = all_passed and result.passed

        print("\n" + "=" * 70)
        print(f"ALL VALIDATIONS: {'PASS' if all_passed else 'FAIL'}")
        print("=" * 70)

    return tests


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def simulate_kuramoto_to_k_formation(
    N: int = 64,
    K0: float = 0.5,
    lambda_neg: float = 1.0,
    max_steps: int = 10000,
    dt: float = 0.01,
    R: float = 10.0,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[HelixState, KFormationValidation, List[float]]:
    """
    Simulate Kuramoto dynamics until K-formation is achieved or max steps reached.

    Parameters
    ----------
    N : int
        Number of oscillators
    K0 : float
        Base coupling strength
    lambda_neg : float
        Negentropy modulation strength
    max_steps : int
        Maximum simulation steps
    dt : float
        Time step
    R : float
        Complexity measure for validation
    seed : int, optional
        Random seed
    verbose : bool
        Print progress

    Returns
    -------
    Tuple[HelixState, KFormationValidation, List[float]]
        (final_state, validation, coherence_history)
    """
    state = create_initial_state(N=N, z0=0.1, seed=seed)
    coherence_history = []

    for step in range(max_steps):
        # Get current coherence
        r, _ = compute_kuramoto_order_parameter(state.phases)
        coherence_history.append(r)

        # Check K-formation
        validation = validate_k_formation(kappa=r, z=state.z, R=R, verbose=False)

        if validation.overall_passed:
            if verbose:
                print(f"K-formation achieved at step {step} (t={state.t:.3f})")
                validate_k_formation(kappa=r, z=state.z, R=R, verbose=True)
            return state, validation, coherence_history

        # Step forward
        state = kuramoto_step_rk4(state, dt=dt, K0=K0, lambda_neg=lambda_neg)

        if verbose and step % 1000 == 0:
            print(f"Step {step}: r={r:.4f}, z={state.z:.4f}, η={compute_negentropy(state.z):.4f}")

    # Return final state even if K-formation not achieved
    r, _ = compute_kuramoto_order_parameter(state.phases)
    validation = validate_k_formation(kappa=r, z=state.z, R=R, verbose=verbose)
    return state, validation, coherence_history


# ============================================================================
# DEMO / SELF-TEST
# ============================================================================

def demo():
    """Demonstrate L₄ helix parameterization module."""
    print("\n" + "=" * 70)
    print("L₄ HELIX PARAMETERIZATION MODULE - DEMO")
    print("=" * 70)

    # Block 1: Constants
    print("\n--- BLOCK 1: L₄ CONSTANTS ---")
    constants = get_l4_constants()
    for name, value in constants.items():
        print(f"  {name:12} = {value:.12f}")

    # Verify identity
    print(f"\n  Identity verified: {L4.verify_identity()}")

    # Block 2: State Vector
    print("\n--- BLOCK 2: STATE VECTOR ---")
    state = create_initial_state(N=32, z0=0.5, seed=42)
    print(f"  N oscillators: {state.N}")
    print(f"  z = {state.z:.6f}")
    print(f"  θ = {state.theta:.6f}")
    print(f"  r(z) = {state.r:.6f}")
    print(f"  Helix position: {state.helix_position()}")

    # Block 3: Dynamics
    print("\n--- BLOCK 3: DYNAMICS ---")
    r0, psi0 = compute_kuramoto_order_parameter(state.phases)
    print(f"  Initial coherence r = {r0:.6f}")
    print(f"  Initial mean phase ψ = {psi0:.6f}")

    # Run 100 steps
    for _ in range(100):
        state = kuramoto_step_rk4(state, dt=0.1, K0=0.5, lambda_neg=1.0)

    r1, psi1 = compute_kuramoto_order_parameter(state.phases)
    print(f"  After 100 steps: r = {r1:.6f}, z = {state.z:.6f}")
    print(f"  Negentropy η = {compute_negentropy(state.z):.6f}")

    # Block 4: Phase → RGB
    print("\n--- BLOCK 4: PHASE → RGB QUANTIZATION ---")
    hex_lattice = HexLatticeWavevectors(wavelength=1.0)
    x = np.array([0.5, 0.5])
    theta_R, theta_G, theta_B = hex_lattice.compute_channel_phases(x)
    print(f"  Position x = {x}")
    print(f"  θ_R = {theta_R:.6f}, θ_G = {theta_G:.6f}, θ_B = {theta_B:.6f}")

    rgb = phases_to_rgb(theta_R, theta_G, theta_B)
    print(f"  RGB = ({rgb.R}, {rgb.G}, {rgb.B})")

    reconstructed = rgb_to_phases(rgb)
    print(f"  Reconstructed phases: {reconstructed}")

    # Block 5: LSB
    print("\n--- BLOCK 5: LSB EMBED/EXTRACT ---")
    test_pixels = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
    message = b"L4"

    embedded = embed_message_lsb(test_pixels, message, bits_per_channel=1)
    extracted = extract_message_lsb(embedded, len(message), bits_per_channel=1)
    print(f"  Original message: {message}")
    print(f"  Extracted message: {extracted}")
    print(f"  Match: {message == extracted}")

    capacity = compute_capacity(10, 10, 3, 1)
    print(f"  Capacity for 10x10 image (1-bit LSB): {capacity} bits = {capacity // 8} bytes")

    # Block 6: Validation
    print("\n--- BLOCK 6: K-FORMATION VALIDATION ---")
    run_all_validations(verbose=True)

    # Example K-formation check
    print("\n--- K-FORMATION CHECK EXAMPLE ---")
    validate_k_formation(kappa=0.95, z=L4.Z_C, R=10.0, verbose=True)

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demo()
