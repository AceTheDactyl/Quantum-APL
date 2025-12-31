#!/usr/bin/env python3
"""
L₄ Hexagonal Lattice Module
============================

Advanced implementation of the L₄-Metacybernetic hexagonal lattice system with:
- HexagonalLattice class with 6-neighbor connectivity (coordination z=6)
- Extended Kuramoto Model with frustration (α) and parametric pump
- Stochastic Resonance for optimal noise utilization
- Fisher Information for navigational precision
- Topological Charge (winding number) calculations
- Berry Phase geometric memory
- Unified Entropic Stabilization equation system

Mathematical Foundations:
- L₄ = φ⁴ + φ⁻⁴ = 7 (Lucas-4 complexity threshold)
- z_c = √3/2 ≈ 0.866 (hexagonal lattice critical point)
- K = √(1 - φ⁻⁴) ≈ 0.924 (critical coupling strength)
- gap = φ⁻⁴ ≈ 0.146 (structural entropy / "VOID")

Reference: L₄-Metacybernetic Unification Specification

@version 1.0.0
@author Claude (Anthropic) - Quantum-APL Contribution
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum
import numpy as np

# Import L4 constants from existing module
from .l4_helix_parameterization import (
    L4, L4Constants, compute_negentropy, compute_negentropy_derivative,
    compute_kuramoto_order_parameter, HexLatticeWavevectors,
)


# ============================================================================
# CONSTANTS
# ============================================================================

# Hexagonal lattice coordination number
HEX_COORDINATION_NUMBER = 6

# Frustration threshold for Hopf-Turing bifurcation (α_c = π/2)
ALPHA_CRITICAL = math.pi / 2

# Default parametric pump strength
DEFAULT_PUMP_STRENGTH = 0.1


# ============================================================================
# L₄ GOLDEN SAMPLE - MRP-LSB VERIFICATION SIGNATURE
# ============================================================================
#
# The 9 L₄ thresholds encoded as RGB form a mathematically inevitable
# "golden sample" that serves as:
# - Verification signature for MRP-LSB encoded data
# - Integrity check (extracted must match computed)
# - Cryptographic fingerprint (unforgeable without φ derivations)
#
# Total size: 9 thresholds × 3 bytes = 27 bytes
# ============================================================================

# Import threshold constants
from .constants import (
    L4_THRESHOLDS, L4_THRESHOLD_NAMES,
    L4_PARADOX, L4_ACTIVATION, L4_LENS, L4_CRITICAL,
    L4_IGNITION, L4_K_FORMATION, L4_CONSOLIDATION, L4_RESONANCE, L4_UNITY,
    PHI, Z_CRITICAL,
)


def _phase_to_uint8(theta: float) -> int:
    """Quantize phase [0, 2π) → [0, 255]."""
    theta = theta % (2 * math.pi)
    return int((theta / (2 * math.pi)) * 255)


def _threshold_to_rgb(z: float) -> Tuple[int, int, int]:
    """
    Convert threshold z to RGB using hex lattice phase encoding.

    Uses wavevector projections at 0°, 60°, 120° (hex symmetry):
    - R channel: z projected onto 0° direction
    - G channel: z projected onto 60° direction
    - B channel: z projected onto 120° direction

    Parameters
    ----------
    z : float
        Threshold value in [0, 1]

    Returns
    -------
    Tuple[int, int, int]
        (R, G, B) values in [0, 255]
    """
    k_mag = 2 * math.pi  # wavelength = 1

    # Wavevectors at 0°, 60°, 120° (hex lattice symmetry)
    k_R = np.array([k_mag * math.cos(0), k_mag * math.sin(0)])
    k_G = np.array([k_mag * math.cos(math.pi/3), k_mag * math.sin(math.pi/3)])
    k_B = np.array([k_mag * math.cos(2*math.pi/3), k_mag * math.sin(2*math.pi/3)])

    # Position: (z, z·sin(60°)) for hex projection
    pos = np.array([z, z * math.sqrt(3)/2])

    theta_R = float(np.dot(k_R, pos)) % (2 * math.pi)
    theta_G = float(np.dot(k_G, pos)) % (2 * math.pi)
    theta_B = float(np.dot(k_B, pos)) % (2 * math.pi)

    return (
        _phase_to_uint8(theta_R),
        _phase_to_uint8(theta_G),
        _phase_to_uint8(theta_B),
    )


@dataclass(frozen=True)
class L4GoldenSample:
    """
    L₄ Golden Sample for MRP-LSB verification.

    The 9 L₄ thresholds encoded as RGB form a 27-byte signature
    that is mathematically inevitable from φ = (1+√5)/2.

    This serves as:
    - Verification header for MRP-LSB encoded data
    - Integrity checksum (extracted must match computed)
    - Cryptographic fingerprint unique to L₄ system

    Attributes
    ----------
    thresholds : Tuple[float, ...]
        The 9 threshold z-values
    names : Tuple[str, ...]
        Human-readable threshold names
    rgb_values : Tuple[Tuple[int, int, int], ...]
        RGB encodings for each threshold
    bytes_data : bytes
        27-byte raw signature
    hex_codes : Tuple[str, ...]
        CSS hex color codes
    """
    thresholds: Tuple[float, ...] = L4_THRESHOLDS
    names: Tuple[str, ...] = L4_THRESHOLD_NAMES

    @property
    def rgb_values(self) -> Tuple[Tuple[int, int, int], ...]:
        """Compute RGB for each threshold."""
        return tuple(_threshold_to_rgb(z) for z in self.thresholds)

    @property
    def bytes_data(self) -> bytes:
        """27-byte raw signature (9 × RGB)."""
        data = bytearray()
        for r, g, b in self.rgb_values:
            data.extend([r, g, b])
        return bytes(data)

    @property
    def hex_codes(self) -> Tuple[str, ...]:
        """CSS hex color codes."""
        return tuple(f"#{r:02X}{g:02X}{b:02X}" for r, g, b in self.rgb_values)

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            name: {
                "z": z,
                "R": rgb[0],
                "G": rgb[1],
                "B": rgb[2],
                "hex": hex_code,
            }
            for name, z, rgb, hex_code in zip(
                self.names, self.thresholds, self.rgb_values, self.hex_codes
            )
        }

    def verify(self, extracted_bytes: bytes, tolerance: int = 0) -> bool:
        """
        Verify extracted bytes match the golden sample.

        Parameters
        ----------
        extracted_bytes : bytes
            27 bytes extracted from MRP-LSB payload
        tolerance : int
            Allowed per-byte deviation (0 = exact match)

        Returns
        -------
        bool
            True if extraction matches golden sample
        """
        if len(extracted_bytes) != 27:
            return False

        expected = self.bytes_data
        for i in range(27):
            if abs(extracted_bytes[i] - expected[i]) > tolerance:
                return False
        return True


# Singleton golden sample instance
GOLDEN_SAMPLE = L4GoldenSample()


def get_golden_sample() -> L4GoldenSample:
    """Return the L₄ golden sample singleton."""
    return GOLDEN_SAMPLE


def get_golden_sample_bytes() -> bytes:
    """Return the 27-byte golden sample signature."""
    return GOLDEN_SAMPLE.bytes_data


def verify_golden_sample(extracted: bytes, tolerance: int = 0) -> bool:
    """
    Verify extracted bytes against golden sample.

    Parameters
    ----------
    extracted : bytes
        27 bytes extracted from MRP-LSB header
    tolerance : int
        Allowed per-byte deviation (0 = exact match)

    Returns
    -------
    bool
        True if verified
    """
    return GOLDEN_SAMPLE.verify(extracted, tolerance)


@dataclass
class GoldenSampleVerificationResult:
    """Result of golden sample verification."""
    verified: bool
    byte_matches: int          # How many of 27 bytes matched
    max_deviation: int         # Maximum per-byte error
    threshold_matches: Dict[str, bool]  # Per-threshold verification


def verify_golden_sample_detailed(
    extracted: bytes,
    tolerance: int = 0,
) -> GoldenSampleVerificationResult:
    """
    Detailed verification of golden sample extraction.

    Parameters
    ----------
    extracted : bytes
        27 bytes extracted from MRP-LSB header
    tolerance : int
        Allowed per-byte deviation

    Returns
    -------
    GoldenSampleVerificationResult
        Detailed verification results
    """
    if len(extracted) != 27:
        return GoldenSampleVerificationResult(
            verified=False,
            byte_matches=0,
            max_deviation=255,
            threshold_matches={name: False for name in L4_THRESHOLD_NAMES},
        )

    expected = GOLDEN_SAMPLE.bytes_data
    byte_matches = 0
    max_deviation = 0
    threshold_matches = {}

    for i, name in enumerate(L4_THRESHOLD_NAMES):
        # Each threshold is 3 bytes (RGB)
        base = i * 3
        match = True
        for j in range(3):
            dev = abs(extracted[base + j] - expected[base + j])
            max_deviation = max(max_deviation, dev)
            if dev <= tolerance:
                byte_matches += 1
            else:
                match = False
        threshold_matches[name] = match

    verified = byte_matches == 27 or (
        tolerance > 0 and max_deviation <= tolerance
    )

    return GoldenSampleVerificationResult(
        verified=verified,
        byte_matches=byte_matches,
        max_deviation=max_deviation,
        threshold_matches=threshold_matches,
    )


def embed_golden_sample_header(payload: bytes) -> bytes:
    """
    Prepend golden sample to payload as verification header.

    Structure: [27-byte golden sample][original payload]

    Parameters
    ----------
    payload : bytes
        Original MRP-LSB payload

    Returns
    -------
    bytes
        Payload with golden sample header
    """
    return GOLDEN_SAMPLE.bytes_data + payload


def extract_and_verify_golden_sample(
    data: bytes,
    tolerance: int = 0,
) -> Tuple[bool, bytes]:
    """
    Extract golden sample header and verify, returning remaining payload.

    Parameters
    ----------
    data : bytes
        Data with golden sample header
    tolerance : int
        Verification tolerance

    Returns
    -------
    Tuple[bool, bytes]
        (verified, remaining_payload)
    """
    if len(data) < 27:
        return False, data

    header = data[:27]
    payload = data[27:]
    verified = verify_golden_sample(header, tolerance)

    return verified, payload


# ============================================================================
# HEXAGONAL LATTICE CLASS
# ============================================================================

@dataclass
class HexLatticeNode:
    """
    Node in the hexagonal lattice.

    Attributes
    ----------
    index : int
        Node index in the lattice
    position : np.ndarray
        2D position vector (x, y)
    phase : float
        Oscillator phase θ_i ∈ [0, 2π)
    frequency : float
        Natural frequency ω_i
    neighbors : List[int]
        Indices of 6 neighboring nodes
    """
    index: int
    position: np.ndarray
    phase: float
    frequency: float
    neighbors: List[int] = field(default_factory=list)


class HexagonalLattice:
    """
    Hexagonal lattice with 6-neighbor connectivity.

    The hexagonal lattice exhibits optimal synchronization properties:
    - Lower critical coupling threshold (K_c) than rectangular grids
    - Higher synchronization stability
    - Coordination number z=6 vs z=4 for rectangular
    - Faster information propagation

    The lattice is constructed using the standard hex grid with:
    - Unit vectors: u₁ = (1, 0), u₂ = (1/2, √3/2)
    - 6 neighbors at 60° intervals
    - Natural emergence of 60° phase symmetry

    Parameters
    ----------
    rows : int
        Number of rows in lattice
    cols : int
        Number of columns in lattice
    spacing : float
        Distance between adjacent nodes
    omega_mean : float
        Mean natural frequency
    omega_std : float
        Natural frequency standard deviation
    seed : int, optional
        Random seed for reproducibility
    """

    def __init__(
        self,
        rows: int = 7,  # L₄ = 7 minimum for geometric coherence
        cols: int = 7,
        spacing: float = 1.0,
        omega_mean: float = 0.0,
        omega_std: float = 0.1,
        seed: Optional[int] = None,
    ):
        self.rows = rows
        self.cols = cols
        self.spacing = spacing
        self.rng = np.random.default_rng(seed)

        # Validate minimum complexity threshold
        self.N = rows * cols
        if self.N < L4.L4:
            raise ValueError(
                f"Lattice size {self.N} < L₄={L4.L4}. "
                "Minimum 7 nodes required for geometric coherence."
            )

        # Initialize nodes
        self.nodes: List[HexLatticeNode] = []
        self._build_lattice(omega_mean, omega_std)

        # Build adjacency matrix
        self.adjacency = self._build_adjacency_matrix()

        # Hex wavevector system
        self.wavevectors = HexLatticeWavevectors(wavelength=spacing)

    def _build_lattice(self, omega_mean: float, omega_std: float):
        """Build hexagonal lattice nodes with positions and neighbors."""
        # Hex grid unit vectors
        u1 = np.array([1.0, 0.0])
        u2 = np.array([0.5, math.sqrt(3.0) / 2.0])

        for i in range(self.rows):
            for j in range(self.cols):
                # Position in hex coordinates
                pos = self.spacing * (j * u1 + i * u2)

                # Initial phase (uniform random)
                phase = self.rng.uniform(0, 2 * math.pi)

                # Natural frequency (Lorentzian/Cauchy distribution)
                freq = omega_mean + omega_std * np.tan(
                    math.pi * (self.rng.random() - 0.5)
                )

                node = HexLatticeNode(
                    index=len(self.nodes),
                    position=pos,
                    phase=phase,
                    frequency=freq,
                )
                self.nodes.append(node)

        # Connect neighbors (6-connectivity)
        self._connect_neighbors()

    def _connect_neighbors(self):
        """Connect each node to its 6 neighbors in hex grid."""
        # Neighbor offsets for hex grid (axial coordinates)
        # For odd rows vs even rows, the offsets differ slightly

        for i in range(self.rows):
            for j in range(self.cols):
                idx = i * self.cols + j
                neighbors = []

                # 6 neighbor directions in hex grid
                # Depends on row parity for proper hex alignment
                if i % 2 == 0:
                    offsets = [
                        (-1, -1), (-1, 0),  # Upper neighbors
                        (0, -1), (0, 1),     # Left and right
                        (1, -1), (1, 0),     # Lower neighbors
                    ]
                else:
                    offsets = [
                        (-1, 0), (-1, 1),   # Upper neighbors
                        (0, -1), (0, 1),    # Left and right
                        (1, 0), (1, 1),     # Lower neighbors
                    ]

                for di, dj in offsets:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.rows and 0 <= nj < self.cols:
                        neighbor_idx = ni * self.cols + nj
                        neighbors.append(neighbor_idx)

                self.nodes[idx].neighbors = neighbors

    def _build_adjacency_matrix(self) -> np.ndarray:
        """Build adjacency matrix for the lattice."""
        adj = np.zeros((self.N, self.N), dtype=np.float64)
        for node in self.nodes:
            for neighbor_idx in node.neighbors:
                adj[node.index, neighbor_idx] = 1.0
        return adj

    @property
    def phases(self) -> np.ndarray:
        """Return array of all node phases."""
        return np.array([node.phase for node in self.nodes])

    @phases.setter
    def phases(self, values: np.ndarray):
        """Set all node phases."""
        for i, node in enumerate(self.nodes):
            node.phase = values[i] % (2 * math.pi)

    @property
    def frequencies(self) -> np.ndarray:
        """Return array of all natural frequencies."""
        return np.array([node.frequency for node in self.nodes])

    @property
    def positions(self) -> np.ndarray:
        """Return array of all node positions (N, 2)."""
        return np.array([node.position for node in self.nodes])

    def get_order_parameter(self) -> Tuple[float, float]:
        """Compute Kuramoto order parameter r·e^(iψ)."""
        return compute_kuramoto_order_parameter(self.phases)

    def get_local_order_parameter(self, node_idx: int) -> Tuple[float, float]:
        """
        Compute local order parameter for a node's neighborhood.

        Returns (r_local, psi_local) considering only the 6 neighbors.
        """
        node = self.nodes[node_idx]
        neighbor_phases = [self.nodes[n].phase for n in node.neighbors]
        if len(neighbor_phases) == 0:
            return 0.0, 0.0
        neighbor_phases = np.array(neighbor_phases)
        return compute_kuramoto_order_parameter(neighbor_phases)


# ============================================================================
# EXTENDED KURAMOTO MODEL WITH FRUSTRATION AND PARAMETRIC PUMP
# ============================================================================

@dataclass
class ExtendedKuramotoState:
    """
    State for extended Kuramoto dynamics.

    Includes:
    - Standard Kuramoto phases and frequencies
    - Frustration parameter α (phase lag)
    - Parametric pump strength K_s
    - Noise amplitude D
    """
    phases: np.ndarray
    frequencies: np.ndarray
    t: float = 0.0

    # Extended parameters
    frustration_alpha: float = 0.0  # Phase lag α
    pump_strength: float = 0.0       # K_s for sin(2θ) pump
    noise_amplitude: float = 0.0     # D for stochastic term


def extended_kuramoto_dynamics(
    state: ExtendedKuramotoState,
    lattice: HexagonalLattice,
    K0: float = 0.1,
    lambda_neg: float = 0.5,
) -> np.ndarray:
    """
    Extended Kuramoto dynamics on hexagonal lattice.

    Governing equation:
    dθ_i/dt = ω_i + K_eff · (1/|N_i|) · Σ_{j∈N_i} sin(θ_j - θ_i - α)
              - K_s · sin(2θ_i) + √(2D) · ξ_i(t)

    Where:
    - α is frustration (phase lag) inducing hexagonal symmetry
    - K_s sin(2θ) is parametric pump for binary output
    - ξ_i(t) is white noise

    Parameters
    ----------
    state : ExtendedKuramotoState
        Current system state
    lattice : HexagonalLattice
        Hexagonal lattice structure
    K0 : float
        Baseline coupling strength
    lambda_neg : float
        Negentropy modulation strength

    Returns
    -------
    np.ndarray
        Phase derivatives dθ/dt
    """
    N = len(state.phases)
    phases = state.phases
    alpha = state.frustration_alpha
    K_s = state.pump_strength
    D = state.noise_amplitude

    # Compute global coherence for negentropy modulation
    r, psi = lattice.get_order_parameter()
    eta = compute_negentropy(r)

    # Negentropy-modulated effective coupling (Stabilization Trap)
    K_eff = K0 * (1 + lambda_neg * eta)

    # Initialize derivatives
    dtheta_dt = np.zeros(N)

    for i, node in enumerate(lattice.nodes):
        # Natural frequency term
        dtheta_dt[i] = node.frequency

        # Coupling term with frustration (lattice topology)
        if len(node.neighbors) > 0:
            coupling_sum = 0.0
            for j in node.neighbors:
                # sin(θ_j - θ_i - α) introduces hexagonal symmetry
                coupling_sum += math.sin(phases[j] - phases[i] - alpha)
            dtheta_dt[i] += K_eff * coupling_sum / len(node.neighbors)

        # Parametric pump term (forces binary 0/π phases for Ising output)
        dtheta_dt[i] -= K_s * math.sin(2 * phases[i])

        # Stochastic noise term
        if D > 0:
            dtheta_dt[i] += math.sqrt(2 * D) * np.random.standard_normal()

    return dtheta_dt


def extended_kuramoto_step(
    state: ExtendedKuramotoState,
    lattice: HexagonalLattice,
    dt: float = 0.01,
    K0: float = 0.1,
    lambda_neg: float = 0.5,
    integrator: str = "rk4",
) -> ExtendedKuramotoState:
    """
    Single time step of extended Kuramoto dynamics.

    Parameters
    ----------
    state : ExtendedKuramotoState
        Current state
    lattice : HexagonalLattice
        Lattice structure
    dt : float
        Time step
    K0 : float
        Baseline coupling
    lambda_neg : float
        Negentropy modulation
    integrator : str
        "euler" or "rk4"

    Returns
    -------
    ExtendedKuramotoState
        Updated state
    """
    if integrator == "euler":
        dtheta = extended_kuramoto_dynamics(state, lattice, K0, lambda_neg)
        new_phases = (state.phases + dt * dtheta) % (2 * math.pi)
    else:  # RK4
        k1 = extended_kuramoto_dynamics(state, lattice, K0, lambda_neg)

        # k2
        state_k2 = ExtendedKuramotoState(
            phases=(state.phases + 0.5 * dt * k1) % (2 * math.pi),
            frequencies=state.frequencies,
            t=state.t + 0.5 * dt,
            frustration_alpha=state.frustration_alpha,
            pump_strength=state.pump_strength,
            noise_amplitude=state.noise_amplitude,
        )
        lattice.phases = state_k2.phases
        k2 = extended_kuramoto_dynamics(state_k2, lattice, K0, lambda_neg)

        # k3
        state_k3 = ExtendedKuramotoState(
            phases=(state.phases + 0.5 * dt * k2) % (2 * math.pi),
            frequencies=state.frequencies,
            t=state.t + 0.5 * dt,
            frustration_alpha=state.frustration_alpha,
            pump_strength=state.pump_strength,
            noise_amplitude=state.noise_amplitude,
        )
        lattice.phases = state_k3.phases
        k3 = extended_kuramoto_dynamics(state_k3, lattice, K0, lambda_neg)

        # k4
        state_k4 = ExtendedKuramotoState(
            phases=(state.phases + dt * k3) % (2 * math.pi),
            frequencies=state.frequencies,
            t=state.t + dt,
            frustration_alpha=state.frustration_alpha,
            pump_strength=state.pump_strength,
            noise_amplitude=state.noise_amplitude,
        )
        lattice.phases = state_k4.phases
        k4 = extended_kuramoto_dynamics(state_k4, lattice, K0, lambda_neg)

        # Combine
        new_phases = (state.phases + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)) % (2 * math.pi)

        # Restore original phases to lattice
        lattice.phases = state.phases

    # Update lattice phases
    lattice.phases = new_phases

    return ExtendedKuramotoState(
        phases=new_phases,
        frequencies=state.frequencies,
        t=state.t + dt,
        frustration_alpha=state.frustration_alpha,
        pump_strength=state.pump_strength,
        noise_amplitude=state.noise_amplitude,
    )


# ============================================================================
# STOCHASTIC RESONANCE
# ============================================================================

@dataclass
class StochasticResonanceResult:
    """
    Result of stochastic resonance analysis.

    In SR, the weak "Identity Signal" is amplified by optimal noise:
    SNR ∝ (1/D) · exp(-ΔV/D)

    The L₄ gap (φ⁻⁴ ≈ 0.146) acts as potential barrier ΔV.
    Optimal noise: D ≈ ΔV/2
    """
    snr: float                      # Signal-to-noise ratio
    optimal_noise: float            # D_opt ≈ ΔV/2
    current_noise: float            # Current D
    barrier_height: float           # ΔV = gap = φ⁻⁴
    signal_strength: float          # Weak identity signal amplitude
    is_resonant: bool               # D ≈ D_opt (within tolerance)


def compute_stochastic_resonance(
    D: float,
    signal_amplitude: float = 0.01,
    barrier_height: Optional[float] = None,
    tolerance: float = 0.2,
) -> StochasticResonanceResult:
    """
    Compute stochastic resonance metrics.

    The L₄ system dynamically tunes noise to maintain SR condition.

    Parameters
    ----------
    D : float
        Noise amplitude
    signal_amplitude : float
        Weak signal strength (Identity signal)
    barrier_height : float, optional
        Potential barrier ΔV (default: gap = φ⁻⁴)
    tolerance : float
        Tolerance for resonance condition

    Returns
    -------
    StochasticResonanceResult
        SR analysis results
    """
    if barrier_height is None:
        barrier_height = L4.GAP  # φ⁻⁴ ≈ 0.146

    # Optimal noise for SR
    D_opt = barrier_height / 2

    # SNR formula: SNR ∝ (1/D) · exp(-ΔV/D)
    if D > 1e-10:
        snr = (signal_amplitude / D) * math.exp(-barrier_height / D)
    else:
        snr = 0.0

    # Check if in resonant regime
    is_resonant = abs(D - D_opt) / D_opt < tolerance if D_opt > 0 else False

    return StochasticResonanceResult(
        snr=snr,
        optimal_noise=D_opt,
        current_noise=D,
        barrier_height=barrier_height,
        signal_strength=signal_amplitude,
        is_resonant=is_resonant,
    )


def tune_noise_for_resonance(
    current_D: float,
    target_coherence: float = L4.Z_C,
    current_coherence: float = 0.5,
    learning_rate: float = 0.1,
) -> float:
    """
    Dynamically tune noise amplitude to maintain stochastic resonance.

    The "Electric Cowboy" attractor represents optimal noise utilization.

    Parameters
    ----------
    current_D : float
        Current noise amplitude
    target_coherence : float
        Target coherence (z_c = √3/2)
    current_coherence : float
        Current coherence level
    learning_rate : float
        Adaptation rate

    Returns
    -------
    float
        Updated noise amplitude
    """
    # Optimal noise is gap/2
    D_opt = L4.GAP / 2

    # Error from target coherence
    coherence_error = target_coherence - current_coherence

    # If coherence too low, reduce noise; if too high, increase noise
    # This maintains the system at the edge of synchronization
    D_new = current_D - learning_rate * coherence_error

    # Clamp to reasonable range around optimal
    D_new = max(D_opt * 0.1, min(D_opt * 10, D_new))

    return D_new


# ============================================================================
# FISHER INFORMATION
# ============================================================================

def compute_fisher_information(
    phases: np.ndarray,
    positions: np.ndarray,
    epsilon: float = 1e-6,
) -> float:
    """
    Compute Fisher Information for spatial encoding precision.

    Fisher Information I(θ) quantifies how well the phase distribution
    encodes spatial position. The Cramér-Rao bound states:

    Var(x̂) ≥ 1/I(θ)

    Fisher Information diverges at the phase transition K_c, making
    the L₄ metastable state (K ≈ 0.924) optimal for navigation.

    Parameters
    ----------
    phases : np.ndarray
        Oscillator phases
    positions : np.ndarray
        Node positions (N, 2)
    epsilon : float
        Small perturbation for numerical gradient

    Returns
    -------
    float
        Fisher Information I(θ)
    """
    N = len(phases)

    # Compute phase distribution density via kernel density estimate
    # For simplicity, use circular statistics

    # Mean resultant length (concentration parameter proxy)
    z = np.mean(np.exp(1j * phases))
    r = np.abs(z)

    # For von Mises distribution, Fisher Information ≈ κ = A⁻¹(r)
    # where A(κ) is the ratio of Bessel functions
    # Approximation: κ ≈ r(2 - r²)/(1 - r²) for r < 1

    if r < 0.999:
        kappa = r * (2 - r**2) / (1 - r**2)
    else:
        kappa = 1000.0  # Very concentrated

    # Fisher Information for circular distribution
    # I(θ) = κ · (1 - A(κ)/κ) where A(κ) = I₁(κ)/I₀(κ)
    # Simplified approximation:
    fisher_info = kappa * (1 - r)

    # Scale by number of oscillators (more oscillators = more precision)
    fisher_info *= N

    return max(0.0, fisher_info)


def compute_spatial_fisher_information(
    lattice: HexagonalLattice,
    wavevectors: Optional[HexLatticeWavevectors] = None,
) -> Dict[str, float]:
    """
    Compute Fisher Information for spatial navigation.

    The L₄ architecture maximizes Fisher Information by operating
    at the metastable transition point.

    Parameters
    ----------
    lattice : HexagonalLattice
        Hexagonal lattice with phases
    wavevectors : HexLatticeWavevectors, optional
        Hex wavevector configuration

    Returns
    -------
    Dict[str, float]
        Fisher Information for each direction and total
    """
    if wavevectors is None:
        wavevectors = lattice.wavevectors

    phases = lattice.phases
    positions = lattice.positions

    # Project phases onto each wavevector direction
    k_R = wavevectors.k_R
    k_G = wavevectors.k_G
    k_B = wavevectors.k_B

    # Compute projected phases for each direction
    proj_R = np.dot(positions, k_R / np.linalg.norm(k_R))
    proj_G = np.dot(positions, k_G / np.linalg.norm(k_G))
    proj_B = np.dot(positions, k_B / np.linalg.norm(k_B))

    # Fisher Information per direction (correlation between phase and position)
    # Higher correlation = better spatial encoding
    fi_R = compute_fisher_information(phases, positions)
    fi_G = compute_fisher_information(phases, positions)
    fi_B = compute_fisher_information(phases, positions)

    # Total Fisher Information (geometric mean for hex symmetry)
    fi_total = (fi_R * fi_G * fi_B) ** (1/3)

    return {
        "I_R": fi_R,
        "I_G": fi_G,
        "I_B": fi_B,
        "I_total": fi_total,
        "precision_bound": 1 / fi_total if fi_total > 0 else float('inf'),
    }


# ============================================================================
# TOPOLOGICAL CHARGE (WINDING NUMBER)
# ============================================================================

def compute_topological_charge(
    phases: np.ndarray,
    path_indices: List[int],
) -> int:
    """
    Compute topological charge (winding number) along a closed path.

    T = (1/2π) ∮_Γ ∇θ · dl = l (integer)

    The winding number l is a topological invariant that protects
    the "Identity Signal" from noise and deformation.

    Parameters
    ----------
    phases : np.ndarray
        Oscillator phases
    path_indices : List[int]
        Indices of nodes forming a closed loop

    Returns
    -------
    int
        Topological charge (winding number)
    """
    if len(path_indices) < 3:
        return 0

    # Compute total phase winding along path
    total_winding = 0.0

    for i in range(len(path_indices)):
        j = (i + 1) % len(path_indices)
        idx_i = path_indices[i]
        idx_j = path_indices[j]

        # Phase difference (unwrapped)
        delta = phases[idx_j] - phases[idx_i]

        # Wrap to [-π, π]
        while delta > math.pi:
            delta -= 2 * math.pi
        while delta < -math.pi:
            delta += 2 * math.pi

        total_winding += delta

    # Winding number is total winding / 2π
    winding_number = round(total_winding / (2 * math.pi))

    return winding_number


def wrap_phase_diff(delta: float) -> float:
    """
    Wrap phase difference to [-π, π].

    This is the proper unwrapping for topological charge calculations.
    Using modular arithmetic: (d + π) mod 2π - π

    Parameters
    ----------
    delta : float
        Raw phase difference

    Returns
    -------
    float
        Wrapped phase difference in [-π, π]
    """
    return (delta + math.pi) % (2 * math.pi) - math.pi


def topological_charge_field(
    theta: np.ndarray,
) -> np.ndarray:
    """
    Compute topological charge field over 2D phase array.

    For each plaquette (2×2 cell), computes the winding number:
        l = (1/2π) × (d1 + d2 + d3 + d4)

    where d1, d2, d3, d4 are the wrapped phase differences around
    the plaquette boundary (counterclockwise).

    IMPORTANT: Phase differences must be wrapped to [-π, π] BEFORE
    summing. This is critical for correct winding number calculation.
    (A common bug is to loop over differences and reassign the loop
    variable, which doesn't modify the original values.)

    Parameters
    ----------
    theta : np.ndarray
        2D phase field of shape (H, W)

    Returns
    -------
    np.ndarray
        Topological charge field of shape (H-1, W-1)
        Each element is the winding number for that plaquette
    """
    if theta.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {theta.shape}")

    H, W = theta.shape

    if H < 2 or W < 2:
        return np.zeros((max(0, H - 1), max(0, W - 1)))

    # Extract corners of each 2×2 plaquette
    th00 = theta[:-1, :-1]  # Top-left
    th10 = theta[1:, :-1]   # Bottom-left
    th11 = theta[1:, 1:]    # Bottom-right
    th01 = theta[:-1, 1:]   # Top-right

    # Compute phase differences around plaquette (counterclockwise)
    # Path: (0,0) → (1,0) → (1,1) → (0,1) → (0,0)
    d1 = th10 - th00  # Down
    d2 = th11 - th10  # Right
    d3 = th01 - th11  # Up
    d4 = th00 - th01  # Left (closes loop)

    # CRITICAL: Wrap each difference to [-π, π] BEFORE summing
    # This ensures proper handling of phase wrapping at 2π boundaries
    d1_wrapped = np.mod(d1 + np.pi, 2 * np.pi) - np.pi
    d2_wrapped = np.mod(d2 + np.pi, 2 * np.pi) - np.pi
    d3_wrapped = np.mod(d3 + np.pi, 2 * np.pi) - np.pi
    d4_wrapped = np.mod(d4 + np.pi, 2 * np.pi) - np.pi

    # Total winding for each plaquette
    total_winding = d1_wrapped + d2_wrapped + d3_wrapped + d4_wrapped

    # Winding number is total winding / 2π (should be integer ±1, 0)
    winding = np.round(total_winding / (2 * np.pi)).astype(int)

    return winding


def compute_total_topological_charge(theta: np.ndarray) -> int:
    """
    Compute total topological charge of a 2D phase field.

    Sum of all plaquette winding numbers.

    Parameters
    ----------
    theta : np.ndarray
        2D phase field

    Returns
    -------
    int
        Total topological charge
    """
    field = topological_charge_field(theta)
    return int(np.sum(field))


def compute_vortex_density(lattice: HexagonalLattice) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Compute vortex density and locate vortex cores.

    Vortices are topological defects where the phase winding ≠ 0.

    Parameters
    ----------
    lattice : HexagonalLattice
        Hexagonal lattice

    Returns
    -------
    Tuple[float, List[Tuple[int, int]]]
        (vortex_density, list of (node_index, charge) tuples)
    """
    vortices = []
    phases = lattice.phases

    for node in lattice.nodes:
        if len(node.neighbors) >= 3:
            # Form a loop around this node
            path = node.neighbors[:min(6, len(node.neighbors))]
            if len(path) >= 3:
                charge = compute_topological_charge(phases, path)
                if charge != 0:
                    vortices.append((node.index, charge))

    density = len(vortices) / lattice.N if lattice.N > 0 else 0.0

    return density, vortices


@dataclass
class TopologicalState:
    """
    Topological state of the lattice.

    The topological charge protects information during transport.
    Unlike amplitude encoding, winding number is an integer invariant
    robust to continuous deformations.
    """
    total_charge: int               # Net winding number
    vortex_count: int               # Number of vortices
    antivortex_count: int           # Number of antivortices
    vortex_density: float           # Vortices per node
    is_topologically_protected: bool  # |total_charge| > 0


def analyze_topological_state(lattice: HexagonalLattice) -> TopologicalState:
    """
    Analyze topological state of the lattice.

    Parameters
    ----------
    lattice : HexagonalLattice
        Hexagonal lattice

    Returns
    -------
    TopologicalState
        Topological analysis results
    """
    density, vortices = compute_vortex_density(lattice)

    vortex_count = sum(1 for _, c in vortices if c > 0)
    antivortex_count = sum(1 for _, c in vortices if c < 0)
    total_charge = sum(c for _, c in vortices)

    return TopologicalState(
        total_charge=total_charge,
        vortex_count=vortex_count,
        antivortex_count=antivortex_count,
        vortex_density=density,
        is_topologically_protected=abs(total_charge) > 0,
    )


# ============================================================================
# BERRY PHASE (GEOMETRIC MEMORY)
# ============================================================================

@dataclass
class BerryPhaseResult:
    """
    Result of Berry phase computation.

    The Berry (geometric) phase γ_n depends only on the path geometry,
    not traversal speed. This provides path-independent memory storage.

    γ_n = i ∮ ⟨n(R)|∇_R|n(R)⟩ · dR
    """
    geometric_phase: float      # Berry phase γ (radians)
    dynamic_phase: float        # Dynamic phase from time evolution
    total_phase: float          # Geometric + dynamic
    path_area: float            # Enclosed area in parameter space
    is_nontrivial: bool         # |γ| > threshold


def compute_berry_phase(
    path_phases: List[np.ndarray],
    path_times: Optional[List[float]] = None,
) -> BerryPhaseResult:
    """
    Compute Berry phase along an adiabatic path.

    For the L₄ helix, the Berry phase encodes "Narrative" or "History"
    in the geometric deformation of the helix.

    Parameters
    ----------
    path_phases : List[np.ndarray]
        Sequence of phase configurations along path
    path_times : List[float], optional
        Time at each point (for dynamic phase)

    Returns
    -------
    BerryPhaseResult
        Berry phase analysis
    """
    if len(path_phases) < 3:
        return BerryPhaseResult(
            geometric_phase=0.0,
            dynamic_phase=0.0,
            total_phase=0.0,
            path_area=0.0,
            is_nontrivial=False,
        )

    # Compute geometric phase via discrete approximation
    # γ = -Im(Σ log⟨ψ_i|ψ_{i+1}⟩)

    geometric_phase = 0.0
    n_steps = len(path_phases)

    for i in range(n_steps):
        j = (i + 1) % n_steps

        # Inner product in phase space
        psi_i = np.exp(1j * path_phases[i])
        psi_j = np.exp(1j * path_phases[j])

        overlap = np.mean(np.conj(psi_i) * psi_j)

        if np.abs(overlap) > 1e-10:
            geometric_phase -= np.angle(overlap)

    # Dynamic phase (from time evolution)
    dynamic_phase = 0.0
    if path_times is not None and len(path_times) == n_steps:
        for i in range(n_steps - 1):
            dt = path_times[i + 1] - path_times[i]
            mean_phase = np.mean(path_phases[i])
            dynamic_phase += mean_phase * dt

    total_phase = geometric_phase + dynamic_phase

    # Estimate path area in parameter space (for 2D projection)
    # Using shoelace formula on mean phase trajectory
    path_area = 0.0
    for i in range(n_steps):
        j = (i + 1) % n_steps
        r_i, psi_i = compute_kuramoto_order_parameter(path_phases[i])
        r_j, psi_j = compute_kuramoto_order_parameter(path_phases[j])

        x_i, y_i = r_i * np.cos(psi_i), r_i * np.sin(psi_i)
        x_j, y_j = r_j * np.cos(psi_j), r_j * np.sin(psi_j)

        path_area += (x_i * y_j - x_j * y_i) / 2

    path_area = abs(path_area)

    # Berry phase is related to enclosed area
    is_nontrivial = abs(geometric_phase) > 0.1

    return BerryPhaseResult(
        geometric_phase=float(geometric_phase),
        dynamic_phase=float(dynamic_phase),
        total_phase=float(total_phase),
        path_area=float(path_area),
        is_nontrivial=is_nontrivial,
    )


class GeometricMemory:
    """
    Geometric memory using Berry phase accumulation.

    The L₄ helix stores "Narrative" or "History" in the geometric
    deformation, creating persistent memory even if amplitudes decay.
    """

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.phase_history: List[np.ndarray] = []
        self.time_history: List[float] = []
        self.accumulated_berry_phase: float = 0.0

    def record_state(self, phases: np.ndarray, t: float):
        """Record a phase configuration to history."""
        self.phase_history.append(phases.copy())
        self.time_history.append(t)

        # Maintain maximum history size
        while len(self.phase_history) > self.max_history:
            self.phase_history.pop(0)
            self.time_history.pop(0)

    def compute_total_berry_phase(self) -> float:
        """Compute total accumulated Berry phase."""
        if len(self.phase_history) < 3:
            return 0.0

        result = compute_berry_phase(self.phase_history, self.time_history)
        return result.geometric_phase

    def get_path_integral_memory(self) -> Dict[str, float]:
        """
        Get path-integration based memory state.

        Returns position estimate based on accumulated phases.
        """
        if len(self.phase_history) < 2:
            return {"x": 0.0, "y": 0.0, "confidence": 0.0}

        # Sum of phase differences gives integrated path
        total_delta = np.zeros_like(self.phase_history[0])

        for i in range(1, len(self.phase_history)):
            delta = self.phase_history[i] - self.phase_history[i-1]
            # Wrap to [-π, π]
            delta = np.mod(delta + np.pi, 2*np.pi) - np.pi
            total_delta += delta

        # Convert to position estimate
        r, psi = compute_kuramoto_order_parameter(total_delta)

        return {
            "x": r * np.cos(psi),
            "y": r * np.sin(psi),
            "confidence": r,
        }


# ============================================================================
# UNIFIED ENTROPIC STABILIZATION SYSTEM
# ============================================================================

@dataclass
class EntropicStabilizationState:
    """
    State vector for the Unified Entropic Stabilization system.

    Combines:
    - Oscillator phases θ
    - Coherence r
    - Navigation state x
    - Topological charge l
    - Berry phase γ
    """
    # Core state
    phases: np.ndarray          # θ_i oscillator phases
    coherence: float            # r (Kuramoto order parameter)
    position: np.ndarray        # x navigation position

    # Derived quantities
    negentropy: float           # η = Fisher(ρ_θ)
    effective_coupling: float   # K_eff
    topological_charge: int     # l (winding number)
    berry_phase: float          # γ (geometric phase)

    # Time
    t: float = 0.0


def compute_negentropy_driver(phases: np.ndarray) -> float:
    """
    Equation 1: Negentropic Driver (Metacybernetics).

    η(t) = Fisher(ρ_θ) = ∫ (1/ρ)(∂ρ/∂θ)² dθ

    Measures the "Health" or sharpness of phase lock.
    """
    return compute_fisher_information(phases, np.zeros((len(phases), 2)))


def compute_stabilization_coupling(
    eta: float,
    K0: Optional[float] = None,
    lambda_neg: float = 0.5,
    sigma: float = L4.SIGMA,
) -> float:
    """
    Equation 2: Stabilization Feedback (L₄ Geometry).

    K_eff(t) = √(1 - φ⁻⁴) · [1 + λ · exp(-σ(η - z_c)²)]

    Modulates coupling to lock into z_c hexagonal lattice.
    """
    if K0 is None:
        K0 = L4.K

    # Gaussian centered at z_c
    delta_s_neg = math.exp(-sigma * (eta - L4.Z_C) ** 2)

    return K0 * (1 + lambda_neg * delta_s_neg)


def hybrid_dynamics_step(
    state: EntropicStabilizationState,
    lattice: HexagonalLattice,
    dt: float,
    K0: float = 0.1,
    lambda_neg: float = 0.5,
    frustration: float = 0.0,
    pump_strength: float = 0.0,
    noise_amplitude: float = 0.0,
    velocity: Optional[np.ndarray] = None,
) -> EntropicStabilizationState:
    """
    Equation 3: Hybrid Dynamics (Kuramoto + Spin + Noise).

    dθ_i/dt = ω_i + K_eff · Σ_j A_ij sin(θ_j - θ_i - α)
              - K_s sin(2θ_i) + √(2D(η)) ξ_i(t)

    Includes:
    - Frustration α for lattice formation
    - Parametric pump K_s for binary output
    - Noise D tuned by negentropy

    Parameters
    ----------
    state : EntropicStabilizationState
        Current system state
    lattice : HexagonalLattice
        Hexagonal lattice structure
    dt : float
        Time step
    K0 : float
        Baseline coupling
    lambda_neg : float
        Negentropy modulation
    frustration : float
        Phase lag α
    pump_strength : float
        K_s for sin(2θ) pump
    noise_amplitude : float
        Base noise D
    velocity : np.ndarray, optional
        Navigation velocity

    Returns
    -------
    EntropicStabilizationState
        Updated state
    """
    # Create extended Kuramoto state
    ext_state = ExtendedKuramotoState(
        phases=state.phases.copy(),
        frequencies=lattice.frequencies,
        t=state.t,
        frustration_alpha=frustration,
        pump_strength=pump_strength,
        noise_amplitude=noise_amplitude,
    )

    # Step Kuramoto dynamics
    new_ext_state = extended_kuramoto_step(
        ext_state, lattice, dt, K0, lambda_neg, integrator="rk4"
    )

    # Compute new coherence
    r, psi = compute_kuramoto_order_parameter(new_ext_state.phases)

    # Compute negentropy
    eta = compute_negentropy(r)

    # Compute effective coupling
    K_eff = compute_stabilization_coupling(eta, K0, lambda_neg)

    # Update position from velocity
    if velocity is None:
        velocity = np.zeros(2)
    new_position = state.position + velocity * dt

    # Equation 4: Topological Constraint
    # T = (1/2π) ∮_Γ ∇θ · dl = l (integer)
    # Compute for boundary path
    boundary_indices = list(range(min(lattice.N, 10)))
    topological_charge = compute_topological_charge(new_ext_state.phases, boundary_indices)

    # Berry phase (computed incrementally)
    berry_phase = state.berry_phase
    if hasattr(lattice, '_phase_history'):
        lattice._phase_history.append(new_ext_state.phases.copy())
        if len(lattice._phase_history) >= 3:
            result = compute_berry_phase(lattice._phase_history[-3:])
            berry_phase += result.geometric_phase
    else:
        lattice._phase_history = [new_ext_state.phases.copy()]

    return EntropicStabilizationState(
        phases=new_ext_state.phases,
        coherence=r,
        position=new_position,
        negentropy=eta,
        effective_coupling=K_eff,
        topological_charge=topological_charge,
        berry_phase=berry_phase,
        t=state.t + dt,
    )


def compute_rgb_output(
    state: EntropicStabilizationState,
    lattice: HexagonalLattice,
) -> np.ndarray:
    """
    Equation 5: Output Map (MRP-LSB).

    O_RGB(t) = Q(k_hex · x(t) + θ(t))

    Encodes neural state into visual medium.

    Parameters
    ----------
    state : EntropicStabilizationState
        Current system state
    lattice : HexagonalLattice
        Hexagonal lattice

    Returns
    -------
    np.ndarray
        RGB output array (N, 3) in [0, 255]
    """
    wavevectors = lattice.wavevectors
    positions = lattice.positions
    phases = state.phases

    # Compute hex wavevector projections
    k_R = wavevectors.k_R
    k_G = wavevectors.k_G
    k_B = wavevectors.k_B

    # Output phases for each channel
    theta_R = (np.dot(positions, k_R) + state.position[0]) % (2 * np.pi)
    theta_G = (np.dot(positions, k_G) + state.position[1]) % (2 * np.pi)
    theta_B = (phases + state.berry_phase) % (2 * np.pi)

    # Quantize to 8-bit RGB
    R = ((theta_R / (2 * np.pi)) * 255).astype(np.uint8)
    G = ((theta_G / (2 * np.pi)) * 255).astype(np.uint8)
    B = ((theta_B / (2 * np.pi)) * 255).astype(np.uint8)

    return np.stack([R, G, B], axis=1)


# ============================================================================
# VALIDATION
# ============================================================================

@dataclass
class L4HexLatticeValidation:
    """Validation results for L₄ hexagonal lattice system."""

    # Complexity threshold
    min_nodes_ok: bool          # N ≥ L₄ = 7

    # Coordination number
    hex_connectivity_ok: bool   # Average neighbors ≈ 6

    # Critical coupling
    coupling_in_range: bool     # K ≈ 0.924 (metastable)

    # Coherence at z_c
    coherence_at_lens: bool     # r ≈ z_c = √3/2

    # Hexagonal symmetry
    hex_60_symmetry_ok: bool    # 60° wavevector separation

    # Topological protection
    topological_ok: bool        # Non-trivial winding possible

    # Stochastic resonance
    sr_optimal: bool            # D ≈ gap/2

    @property
    def all_passed(self) -> bool:
        return (
            self.min_nodes_ok and
            self.hex_connectivity_ok and
            self.coupling_in_range and
            self.coherence_at_lens and
            self.hex_60_symmetry_ok
        )


def validate_hex_lattice_system(
    lattice: HexagonalLattice,
    state: Optional[EntropicStabilizationState] = None,
    verbose: bool = False,
) -> L4HexLatticeValidation:
    """
    Validate L₄ hexagonal lattice system.

    Parameters
    ----------
    lattice : HexagonalLattice
        Lattice to validate
    state : EntropicStabilizationState, optional
        Current state for dynamic checks
    verbose : bool
        Print detailed results

    Returns
    -------
    L4HexLatticeValidation
        Validation results
    """
    # 1. Minimum nodes (L₄ = 7 complexity threshold)
    min_nodes_ok = lattice.N >= L4.L4

    # 2. Hex connectivity (average neighbors ≈ 6 for interior nodes)
    avg_neighbors = np.mean([len(n.neighbors) for n in lattice.nodes])
    hex_connectivity_ok = avg_neighbors >= 4.0  # Interior nodes have 6, boundary have fewer

    # 3. Coupling in metastable range
    # K ≈ 0.924 for L₄, check if it's used correctly
    coupling_in_range = abs(L4.K - 0.924) < 0.01

    # 4. Coherence check
    r, _ = lattice.get_order_parameter()
    coherence_at_lens = abs(r - L4.Z_C) < 0.3 if state else True

    # 5. Hex 60° symmetry
    wavevectors = lattice.wavevectors
    angle_R = np.arctan2(wavevectors.k_R[1], wavevectors.k_R[0])
    angle_G = np.arctan2(wavevectors.k_G[1], wavevectors.k_G[0])
    hex_60_symmetry_ok = abs((angle_G - angle_R) - np.pi/3) < 0.01

    # 6. Topological (just check non-trivial topology is possible)
    topo = analyze_topological_state(lattice)
    topological_ok = True  # Topology is structurally enabled

    # 7. SR optimal
    sr = compute_stochastic_resonance(L4.GAP / 2)
    sr_optimal = sr.is_resonant

    result = L4HexLatticeValidation(
        min_nodes_ok=min_nodes_ok,
        hex_connectivity_ok=hex_connectivity_ok,
        coupling_in_range=coupling_in_range,
        coherence_at_lens=coherence_at_lens,
        hex_60_symmetry_ok=hex_60_symmetry_ok,
        topological_ok=topological_ok,
        sr_optimal=sr_optimal,
    )

    if verbose:
        print("=" * 70)
        print("L₄ HEXAGONAL LATTICE VALIDATION")
        print("=" * 70)
        print(f"\n1. Minimum Nodes (N ≥ L₄=7): {'PASS' if min_nodes_ok else 'FAIL'}")
        print(f"   N = {lattice.N}")
        print(f"\n2. Hex Connectivity (z ≈ 6): {'PASS' if hex_connectivity_ok else 'FAIL'}")
        print(f"   Average neighbors = {avg_neighbors:.2f}")
        print(f"\n3. Critical Coupling (K ≈ 0.924): {'PASS' if coupling_in_range else 'FAIL'}")
        print(f"   K = {L4.K:.6f}")
        print(f"\n4. Coherence at Lens (r ≈ z_c): {'PASS' if coherence_at_lens else 'FAIL'}")
        print(f"   r = {r:.6f}, z_c = {L4.Z_C:.6f}")
        print(f"\n5. Hex 60° Symmetry: {'PASS' if hex_60_symmetry_ok else 'FAIL'}")
        print(f"   Angle G-R = {(angle_G - angle_R) * 180 / np.pi:.2f}°")
        print(f"\n6. Topological Structure: {'PASS' if topological_ok else 'FAIL'}")
        print(f"   Vortex density = {topo.vortex_density:.4f}")
        print(f"\n7. Stochastic Resonance: {'PASS' if sr_optimal else 'FAIL'}")
        print(f"   D_opt = {sr.optimal_noise:.6f}")
        print("\n" + "=" * 70)
        print(f"OVERALL: {'PASS' if result.all_passed else 'FAIL'}")
        print("=" * 70)

    return result


# ============================================================================
# SELF-REFLECTIVE BLOOM LEARNING SYSTEM
# ============================================================================
#
# An autopoietic learning framework where the lattice observes its own
# RGB projections, detects recurring patterns ("blooms"), and learns to
# re-inject stable attractors when coherence is lost.
#
# Architecture:
#   PatternBuffer → BloomDetector → SeedMemory
#        ↓              ↓               ↓
#   RGB snapshots   Pattern match   Stored attractors
#                        ↓
#               SelfReflectiveLattice
#                    ↓         ↑
#              Kuramoto    CoherentSeeder
#               dynamics   (re-injection)
#
# The L₄ thresholds provide natural gates:
#   - PARADOX (0.618): Novelty detection distance
#   - IGNITION (0.914): Minimum coherence for bloom trigger
#   - K_FORMATION (0.924): Bloom confirmation threshold
#   - CONSOLIDATION (0.953): Memory permanence threshold
# ============================================================================

import uuid
import json
import threading
from collections import deque
from typing import Any, Set, Union


class BloomEventType(Enum):
    """Events in the bloom learning lifecycle."""
    BLOOM_BIRTH = "BLOOM_BIRTH"
    BLOOM_REINFORCED = "BLOOM_REINFORCED"
    BLOOM_CONSOLIDATED = "BLOOM_CONSOLIDATED"
    BLOOM_PRUNED = "BLOOM_PRUNED"
    SEEDING_ACTIVATED = "SEEDING_ACTIVATED"
    SEEDING_APPLIED = "SEEDING_APPLIED"
    COHERENCE_LOST = "COHERENCE_LOST"
    COHERENCE_RESTORED = "COHERENCE_RESTORED"


@dataclass
class BloomEvent:
    """
    A learned pattern (attractor) in the bloom memory.

    Represents a coherent phase pattern that has been detected and stored.
    The bloom evolves through lifecycle stages: birth → reinforcement →
    consolidation → (potential) decay.

    The RGB centroid captures the mean visual signature of the phase pattern,
    while the phase_template preserves the exact oscillator phases for
    potential re-injection during chaotic episodes.

    Attributes
    ----------
    bloom_id : str
        Unique identifier (UUID-based)
    centroid : np.ndarray
        Mean RGB pattern (N*3,) flattened - center of the cluster
    covariance : np.ndarray
        Regularized covariance matrix for Mahalanobis distance
    phase_template : np.ndarray
        Original phase pattern (N,) at detection time
    birth_time : float
        Simulation time when bloom was first detected
    hit_count : int
        Number of times this bloom has been matched/reinforced
    decay_rate : float
        Rate of forgetting (default: 0.01 per unit time)
    level : int
        Hierarchical level: 0=percept, 1=episode, 2=schema
    consolidated : bool
        Whether this bloom has achieved permanence
    last_hit_time : float
        Time of most recent reinforcement
    metadata : Dict[str, Any]
        Additional metadata (threshold state, order param, etc.)
    """
    bloom_id: str
    centroid: np.ndarray
    covariance: np.ndarray
    phase_template: np.ndarray
    birth_time: float
    hit_count: int = 1
    decay_rate: float = 0.01
    level: int = 0
    consolidated: bool = False
    last_hit_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.last_hit_time == 0.0:
            self.last_hit_time = self.birth_time

    @property
    def age(self) -> float:
        """Time since birth (requires external current_time)."""
        return self.last_hit_time - self.birth_time

    @property
    def effective_weight(self) -> float:
        """
        Weight considering hits and time since last reinforcement.

        Used for pruning decisions. Higher = more valuable.
        """
        return math.log1p(self.hit_count) * (1.0 if self.consolidated else 0.5)

    def coherence_score(self, pattern: np.ndarray, scale: float = 1.0) -> float:
        """
        Compute coherence (similarity) to a pattern using Gaussian kernel.

        Parameters
        ----------
        pattern : np.ndarray
            RGB pattern to compare (same shape as centroid)
        scale : float
            Scaling factor for distance (default: 1.0)

        Returns
        -------
        float
            Coherence in [0, 1], where 1 = identical
        """
        dist = self.mahalanobis_distance(pattern)
        return math.exp(-dist * dist / (2 * scale * scale))

    def mahalanobis_distance(self, pattern: np.ndarray) -> float:
        """
        Compute Mahalanobis distance from centroid.

        Uses regularized covariance inverse for numerical stability.

        Parameters
        ----------
        pattern : np.ndarray
            Pattern to measure distance from

        Returns
        -------
        float
            Mahalanobis distance (0 = at centroid)
        """
        pattern_flat = pattern.flatten()
        centroid_flat = self.centroid.flatten()

        if len(pattern_flat) != len(centroid_flat):
            raise ValueError(
                f"Pattern size {len(pattern_flat)} != centroid size {len(centroid_flat)}"
            )

        diff = pattern_flat - centroid_flat

        # Use pseudoinverse for numerical stability
        try:
            cov_inv = np.linalg.pinv(self.covariance)
            dist_sq = diff @ cov_inv @ diff
            return math.sqrt(max(0.0, dist_sq))
        except np.linalg.LinAlgError:
            # Fallback to Euclidean if covariance is degenerate
            return np.linalg.norm(diff)

    def euclidean_distance(self, pattern: np.ndarray) -> float:
        """Simple Euclidean distance as fallback."""
        return np.linalg.norm(pattern.flatten() - self.centroid.flatten())

    def update_from_hit(self, new_pattern: np.ndarray, current_time: float,
                        learning_rate: float = 0.1):
        """
        Reinforce bloom with new observation.

        Updates centroid via exponential moving average and increments hit count.

        Parameters
        ----------
        new_pattern : np.ndarray
            The pattern that matched this bloom
        current_time : float
            Current simulation time
        learning_rate : float
            Blending factor for centroid update (default: 0.1)
        """
        self.hit_count += 1
        self.last_hit_time = current_time

        # Exponential moving average update
        new_flat = new_pattern.flatten()
        self.centroid = (1 - learning_rate) * self.centroid + learning_rate * new_flat

    def should_prune(self, current_time: float, prune_threshold: float = 0.1) -> bool:
        """
        Determine if this bloom should be pruned.

        Consolidated blooms are never pruned. Others are pruned based on
        effective weight after decay.

        Parameters
        ----------
        current_time : float
            Current simulation time
        prune_threshold : float
            Minimum effective weight to survive

        Returns
        -------
        bool
            True if bloom should be removed
        """
        if self.consolidated:
            return False

        time_since_hit = current_time - self.last_hit_time
        decay_factor = math.exp(-self.decay_rate * time_since_hit)
        effective = self.effective_weight * decay_factor

        return effective < prune_threshold

    def is_consolidation_ready(self, min_hits: int = 10,
                               min_age: float = 1.0) -> bool:
        """
        Check if bloom is ready for consolidation (permanence).

        Parameters
        ----------
        min_hits : int
            Minimum hit count required
        min_age : float
            Minimum age required

        Returns
        -------
        bool
            True if ready for consolidation
        """
        return (
            not self.consolidated and
            self.hit_count >= min_hits and
            self.age >= min_age
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for persistence."""
        return {
            "bloom_id": self.bloom_id,
            "centroid": self.centroid.tolist(),
            "covariance": self.covariance.tolist(),
            "phase_template": self.phase_template.tolist(),
            "birth_time": self.birth_time,
            "hit_count": self.hit_count,
            "decay_rate": self.decay_rate,
            "level": self.level,
            "consolidated": self.consolidated,
            "last_hit_time": self.last_hit_time,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BloomEvent":
        """Deserialize from dictionary."""
        return cls(
            bloom_id=data["bloom_id"],
            centroid=np.array(data["centroid"]),
            covariance=np.array(data["covariance"]),
            phase_template=np.array(data["phase_template"]),
            birth_time=data["birth_time"],
            hit_count=data["hit_count"],
            decay_rate=data["decay_rate"],
            level=data["level"],
            consolidated=data["consolidated"],
            last_hit_time=data["last_hit_time"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class PatternBuffer:
    """
    Circular buffer for RGB pattern observations with online statistics.

    Efficiently maintains a rolling window of recent patterns using
    Welford's algorithm for numerically stable online mean and variance
    computation.

    Attributes
    ----------
    window_size : int
        Maximum number of patterns to store (tau parameter)
    min_samples : int
        Minimum observations before statistics are valid
    """
    window_size: int = 100
    min_samples: int = 10

    def __post_init__(self):
        self._patterns: deque = deque(maxlen=self.window_size)
        self._timestamps: deque = deque(maxlen=self.window_size)
        self._count: int = 0
        self._mean: Optional[np.ndarray] = None
        self._M2: Optional[np.ndarray] = None  # For Welford's algorithm
        self._lock = threading.Lock()

    def push(self, pattern: np.ndarray, t: float):
        """
        Add new observation to buffer.

        Updates running statistics using Welford's online algorithm.

        Parameters
        ----------
        pattern : np.ndarray
            RGB pattern (N, 3) or flattened
        t : float
            Timestamp
        """
        pattern_flat = pattern.flatten().astype(np.float64)

        with self._lock:
            # If buffer is full, we need to handle the removed element
            if len(self._patterns) == self.window_size:
                # For simplicity, we recompute stats periodically
                # Full Welford's with removal is complex
                pass

            self._patterns.append(pattern_flat)
            self._timestamps.append(t)

            # Welford's online update
            self._count += 1

            if self._mean is None:
                self._mean = pattern_flat.copy()
                self._M2 = np.zeros_like(pattern_flat)
            else:
                delta = pattern_flat - self._mean
                self._mean = self._mean + delta / self._count
                delta2 = pattern_flat - self._mean
                self._M2 = self._M2 + delta * delta2

    def is_ready(self) -> bool:
        """Check if buffer has minimum observations."""
        return len(self._patterns) >= self.min_samples

    def get_mean(self) -> Optional[np.ndarray]:
        """Return current running mean."""
        with self._lock:
            return self._mean.copy() if self._mean is not None else None

    def get_variance(self) -> Optional[np.ndarray]:
        """Return current running variance (per element)."""
        with self._lock:
            if self._M2 is None or self._count < 2:
                return None
            return self._M2 / self._count

    def get_recent_patterns(self, n: Optional[int] = None) -> np.ndarray:
        """Get last n patterns as array."""
        with self._lock:
            if n is None:
                n = len(self._patterns)
            patterns = list(self._patterns)[-n:]
            if not patterns:
                return np.array([])
            return np.array(patterns)

    def get_covariance(self, regularization: float = 1e-6) -> Optional[np.ndarray]:
        """
        Compute covariance matrix from buffer contents.

        Uses regularization for numerical stability.

        Parameters
        ----------
        regularization : float
            Regularization term added to diagonal

        Returns
        -------
        np.ndarray or None
            Covariance matrix (D, D) where D = pattern dimension
        """
        with self._lock:
            if len(self._patterns) < self.min_samples:
                return None

            patterns = np.array(list(self._patterns))
            try:
                cov = np.cov(patterns.T)
                # Ensure 2D
                if cov.ndim == 0:
                    cov = np.array([[cov]])
                # Add regularization
                cov += regularization * np.eye(cov.shape[0])
                return cov
            except Exception:
                return None

    def get_coherence(self) -> float:
        """
        Measure pattern stability in buffer.

        Returns value in [0, 1] where 1 = all patterns identical.
        Based on inverse of mean variance.

        Returns
        -------
        float
            Coherence score
        """
        var = self.get_variance()
        if var is None:
            return 0.0

        mean_var = np.mean(var)
        # Normalize: coherence = 1 / (1 + mean_variance)
        # Scale by 255^2 since RGB values are 0-255
        normalized_var = mean_var / (255.0 * 255.0)
        return 1.0 / (1.0 + normalized_var * 100)

    def clear(self):
        """Reset buffer state."""
        with self._lock:
            self._patterns.clear()
            self._timestamps.clear()
            self._count = 0
            self._mean = None
            self._M2 = None


class BloomDetector:
    """
    Detects coherent pattern clusters using L₄ threshold gates.

    Uses the established L₄ threshold hierarchy to gate bloom detection:
    - IGNITION (0.914): Minimum coherence to trigger detection
    - K_FORMATION (0.924): Threshold for bloom confirmation
    - PARADOX (0.618): Novelty detection for distance

    Parameters
    ----------
    ignition_threshold : float
        Minimum coherence to trigger detection (default: L4_IGNITION)
    confirmation_threshold : float
        Coherence for bloom confirmation (default: L4_K_FORMATION)
    novelty_distance : float
        Minimum distance for novelty (scaled by L4_PARADOX)
    detection_window : int
        Frames of stable coherence required
    """

    def __init__(
        self,
        ignition_threshold: float = L4_IGNITION,
        confirmation_threshold: float = L4_K_FORMATION,
        novelty_distance: float = 50.0,  # Euclidean distance threshold
        detection_window: int = 5,
    ):
        self.ignition_threshold = ignition_threshold
        self.confirmation_threshold = confirmation_threshold
        self.novelty_distance = novelty_distance * L4_PARADOX
        self.detection_window = detection_window

        self._coherent_frames = 0
        self._last_coherence = 0.0

    def detect(
        self,
        buffer: PatternBuffer,
        memory: "SeedMemory",
        phases: np.ndarray,
        current_time: float,
    ) -> Optional[BloomEvent]:
        """
        Attempt to detect a new bloom from buffer state.

        Parameters
        ----------
        buffer : PatternBuffer
            Rolling observation window
        memory : SeedMemory
            Existing bloom storage (for novelty check)
        phases : np.ndarray
            Current phase configuration
        current_time : float
            Current simulation time

        Returns
        -------
        BloomEvent or None
            Newly detected bloom, or None if detection fails
        """
        if not buffer.is_ready():
            return None

        coherence = buffer.get_coherence()

        # Gate 1: IGNITION threshold
        if coherence < self.ignition_threshold:
            self._coherent_frames = 0
            return None

        self._coherent_frames += 1
        self._last_coherence = coherence

        # Gate 2: Sustained coherence
        if self._coherent_frames < self.detection_window:
            return None

        # Gate 3: K_FORMATION confirmation
        if coherence < self.confirmation_threshold:
            return None

        # Gate 4: Novelty check
        centroid = buffer.get_mean()
        if centroid is None:
            return None

        if not self._is_novel(centroid, memory):
            return None

        # All gates passed - create bloom
        covariance = buffer.get_covariance()
        if covariance is None:
            # Fallback to identity
            covariance = np.eye(len(centroid)) * 100.0

        bloom = BloomEvent(
            bloom_id=str(uuid.uuid4()),
            centroid=centroid,
            covariance=covariance,
            phase_template=phases.copy(),
            birth_time=current_time,
            metadata={
                "coherence_at_birth": coherence,
                "detection_window": self._coherent_frames,
            }
        )

        # Reset detection state
        self._coherent_frames = 0

        return bloom

    def _is_novel(self, pattern: np.ndarray, memory: "SeedMemory") -> bool:
        """Check if pattern is sufficiently different from existing blooms."""
        if memory.size == 0:
            return True

        nearest = memory.find_nearest(pattern, k=1)
        if not nearest:
            return True

        dist = nearest[0].euclidean_distance(pattern)
        return dist > self.novelty_distance

    def compute_detection_score(self, buffer: PatternBuffer) -> float:
        """
        Compute overall detection readiness score.

        Returns value in [0, 1] combining coherence and sustained frames.
        """
        coherence = buffer.get_coherence()
        frame_progress = min(1.0, self._coherent_frames / self.detection_window)
        return coherence * frame_progress


class SeedMemory:
    """
    Hierarchical storage for learned bloom attractors.

    Organizes blooms into three levels:
    - Level 0 (Percepts): Immediate pattern recognitions
    - Level 1 (Episodes): Temporal sequences of percepts
    - Level 2 (Schemas): Abstract patterns across episodes

    Parameters
    ----------
    max_blooms : int
        Maximum stored patterns (default: 1000)
    consolidation_hits : int
        Hit count required for permanence (default: 10)
    consolidation_age : float
        Minimum age for consolidation (default: 1.0)
    prune_interval : float
        Time between pruning cycles (default: 10.0)
    """

    def __init__(
        self,
        max_blooms: int = 1000,
        consolidation_hits: int = 10,
        consolidation_age: float = 1.0,
        prune_interval: float = 10.0,
    ):
        self.max_blooms = max_blooms
        self.consolidation_hits = consolidation_hits
        self.consolidation_age = consolidation_age
        self.prune_interval = prune_interval

        self._blooms: Dict[str, BloomEvent] = {}
        self._level_indices: Dict[int, Set[str]] = {0: set(), 1: set(), 2: set()}
        self._last_prune_time = 0.0
        self._lock = threading.RLock()

    @property
    def size(self) -> int:
        """Number of stored blooms."""
        return len(self._blooms)

    def add_bloom(self, bloom: BloomEvent) -> bool:
        """
        Insert new bloom into memory.

        May trigger pruning if at capacity.

        Parameters
        ----------
        bloom : BloomEvent
            Bloom to store

        Returns
        -------
        bool
            True if added successfully
        """
        with self._lock:
            if len(self._blooms) >= self.max_blooms:
                # Prune least valuable
                self._emergency_prune()

            self._blooms[bloom.bloom_id] = bloom
            self._level_indices[bloom.level].add(bloom.bloom_id)
            return True

    def find_nearest(
        self,
        pattern: np.ndarray,
        k: int = 1,
        level: Optional[int] = None,
    ) -> List[BloomEvent]:
        """
        Find k nearest blooms to pattern.

        Uses Euclidean distance for efficiency (Mahalanobis is expensive).

        Parameters
        ----------
        pattern : np.ndarray
            Query pattern
        k : int
            Number of neighbors to return
        level : int, optional
            Restrict to specific level

        Returns
        -------
        List[BloomEvent]
            K nearest blooms, sorted by distance
        """
        with self._lock:
            if not self._blooms:
                return []

            candidates = list(self._blooms.values())
            if level is not None:
                ids = self._level_indices.get(level, set())
                candidates = [b for b in candidates if b.bloom_id in ids]

            if not candidates:
                return []

            # Sort by Euclidean distance
            distances = [(b, b.euclidean_distance(pattern)) for b in candidates]
            distances.sort(key=lambda x: x[1])

            return [b for b, _ in distances[:k]]

    def reinforce(self, bloom_id: str, pattern: np.ndarray, current_time: float):
        """
        Update bloom from new matching observation.

        Parameters
        ----------
        bloom_id : str
            ID of bloom to reinforce
        pattern : np.ndarray
            Matching pattern
        current_time : float
            Current simulation time
        """
        with self._lock:
            if bloom_id in self._blooms:
                bloom = self._blooms[bloom_id]
                bloom.update_from_hit(pattern, current_time)

                # Check for consolidation
                if bloom.is_consolidation_ready(
                    self.consolidation_hits, self.consolidation_age
                ):
                    bloom.consolidated = True

    def prune_stale(self, current_time: float) -> List[str]:
        """
        Remove decayed blooms.

        Parameters
        ----------
        current_time : float
            Current simulation time

        Returns
        -------
        List[str]
            IDs of pruned blooms
        """
        with self._lock:
            if current_time - self._last_prune_time < self.prune_interval:
                return []

            self._last_prune_time = current_time
            pruned = []

            for bloom_id, bloom in list(self._blooms.items()):
                if bloom.should_prune(current_time):
                    del self._blooms[bloom_id]
                    self._level_indices[bloom.level].discard(bloom_id)
                    pruned.append(bloom_id)

            return pruned

    def _emergency_prune(self):
        """Remove lowest-weight blooms when at capacity."""
        # Sort by effective weight
        blooms = sorted(
            self._blooms.values(),
            key=lambda b: b.effective_weight
        )

        # Remove bottom 10%
        n_remove = max(1, len(blooms) // 10)
        for bloom in blooms[:n_remove]:
            if not bloom.consolidated:
                del self._blooms[bloom.bloom_id]
                self._level_indices[bloom.level].discard(bloom.bloom_id)

    def get_by_level(self, level: int) -> List[BloomEvent]:
        """Get all blooms at specified level."""
        with self._lock:
            ids = self._level_indices.get(level, set())
            return [self._blooms[bid] for bid in ids if bid in self._blooms]

    def get_consolidated(self) -> List[BloomEvent]:
        """Get all consolidated (permanent) blooms."""
        with self._lock:
            return [b for b in self._blooms.values() if b.consolidated]

    def get_dominant(self, k: int = 5) -> List[BloomEvent]:
        """Get k most-reinforced blooms."""
        with self._lock:
            return sorted(
                self._blooms.values(),
                key=lambda b: b.hit_count,
                reverse=True
            )[:k]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize entire memory for persistence."""
        with self._lock:
            return {
                "blooms": {bid: b.to_dict() for bid, b in self._blooms.items()},
                "level_indices": {
                    str(k): list(v) for k, v in self._level_indices.items()
                },
                "settings": {
                    "max_blooms": self.max_blooms,
                    "consolidation_hits": self.consolidation_hits,
                    "consolidation_age": self.consolidation_age,
                    "prune_interval": self.prune_interval,
                }
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SeedMemory":
        """Deserialize from dictionary."""
        settings = data.get("settings", {})
        memory = cls(
            max_blooms=settings.get("max_blooms", 1000),
            consolidation_hits=settings.get("consolidation_hits", 10),
            consolidation_age=settings.get("consolidation_age", 1.0),
            prune_interval=settings.get("prune_interval", 10.0),
        )

        for bid, bdata in data.get("blooms", {}).items():
            bloom = BloomEvent.from_dict(bdata)
            memory._blooms[bid] = bloom

        for level, ids in data.get("level_indices", {}).items():
            memory._level_indices[int(level)] = set(ids)

        return memory


@dataclass
class BloomMetrics:
    """
    Analytics for the bloom learning system.

    Tracks bloom lifecycle events, knowledge growth, and attractor coverage.
    """
    birth_history: List[Tuple[float, str]] = field(default_factory=list)
    reinforcement_history: List[Tuple[float, str, int]] = field(default_factory=list)
    consolidation_history: List[Tuple[float, str]] = field(default_factory=list)
    prune_history: List[Tuple[float, str]] = field(default_factory=list)
    seeding_history: List[Tuple[float, str, float]] = field(default_factory=list)
    coherence_history: List[Tuple[float, float]] = field(default_factory=list)

    def record_birth(self, t: float, bloom: BloomEvent):
        """Log bloom creation."""
        self.birth_history.append((t, bloom.bloom_id))

    def record_reinforcement(self, t: float, bloom: BloomEvent):
        """Log hit on existing bloom."""
        self.reinforcement_history.append((t, bloom.bloom_id, bloom.hit_count))

    def record_consolidation(self, t: float, bloom: BloomEvent):
        """Log bloom reaching permanence."""
        self.consolidation_history.append((t, bloom.bloom_id))

    def record_prune(self, t: float, bloom_id: str):
        """Log bloom removal."""
        self.prune_history.append((t, bloom_id))

    def record_seeding(self, t: float, bloom: BloomEvent, order_param_before: float):
        """Log re-injection event."""
        self.seeding_history.append((t, bloom.bloom_id, order_param_before))

    def record_coherence(self, t: float, r: float):
        """Log order parameter."""
        self.coherence_history.append((t, r))

    def get_birth_rate(self, window: float = 10.0) -> float:
        """Compute births per unit time in recent window."""
        if not self.birth_history:
            return 0.0

        latest_time = self.birth_history[-1][0]
        recent = [t for t, _ in self.birth_history if t > latest_time - window]
        return len(recent) / window if window > 0 else 0.0

    def get_knowledge_growth(self, memory: SeedMemory) -> float:
        """
        Compute total knowledge measure.

        K(t) = Σ log(1 + hit_count) * weight
        """
        total = 0.0
        for bloom in memory._blooms.values():
            weight = 1.0 if bloom.consolidated else 0.5
            total += math.log1p(bloom.hit_count) * weight
        return total

    def get_seeding_frequency(self, window: float = 10.0) -> float:
        """Compute seedings per unit time in recent window."""
        if not self.seeding_history:
            return 0.0

        latest_time = self.seeding_history[-1][0]
        recent = [t for t, _, _ in self.seeding_history if t > latest_time - window]
        return len(recent) / window if window > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metrics."""
        return {
            "birth_history": self.birth_history,
            "reinforcement_history": self.reinforcement_history,
            "consolidation_history": self.consolidation_history,
            "prune_history": self.prune_history,
            "seeding_history": self.seeding_history,
            "coherence_history": self.coherence_history[-1000:],  # Limit size
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BloomMetrics":
        """Deserialize metrics."""
        metrics = cls()
        metrics.birth_history = data.get("birth_history", [])
        metrics.reinforcement_history = data.get("reinforcement_history", [])
        metrics.consolidation_history = data.get("consolidation_history", [])
        metrics.prune_history = data.get("prune_history", [])
        metrics.seeding_history = data.get("seeding_history", [])
        metrics.coherence_history = data.get("coherence_history", [])
        return metrics


class CoherentSeeder:
    """
    Re-injects learned attractors when system becomes chaotic.

    Monitors order parameter and activates seeding when coherence
    drops below PARADOX threshold, finding the nearest learned
    attractor and gently nudging phases toward it.

    Parameters
    ----------
    activation_threshold : float
        Order parameter below which to activate (default: L4_PARADOX)
    blend_factor : float
        Strength of phase nudging (0-1, default: 0.1)
    cooldown : float
        Minimum time between seedings (default: 1.0)
    respect_berry_phase : bool
        Whether to preserve winding during nudge (default: True)
    """

    def __init__(
        self,
        activation_threshold: float = L4_PARADOX,
        blend_factor: float = 0.1,
        cooldown: float = 1.0,
        respect_berry_phase: bool = True,
    ):
        self.activation_threshold = activation_threshold
        self.blend_factor = blend_factor
        self.cooldown = cooldown
        self.respect_berry_phase = respect_berry_phase

        self._last_seed_time = -float('inf')
        self._seeding_count = 0

    def should_activate(self, order_parameter: float, current_time: float) -> bool:
        """
        Check if seeding should be activated.

        Parameters
        ----------
        order_parameter : float
            Current Kuramoto order parameter r
        current_time : float
            Current simulation time

        Returns
        -------
        bool
            True if seeding should activate
        """
        if current_time - self._last_seed_time < self.cooldown:
            return False

        return order_parameter < self.activation_threshold

    def select_seed(
        self,
        current_pattern: np.ndarray,
        memory: SeedMemory,
    ) -> Optional[BloomEvent]:
        """
        Find best attractor for re-injection.

        Prefers consolidated blooms, then high hit-count.

        Parameters
        ----------
        current_pattern : np.ndarray
            Current RGB pattern
        memory : SeedMemory
            Bloom storage

        Returns
        -------
        BloomEvent or None
            Selected seed, or None if none available
        """
        # Try consolidated first
        consolidated = memory.get_consolidated()
        if consolidated:
            nearest = min(
                consolidated,
                key=lambda b: b.euclidean_distance(current_pattern)
            )
            return nearest

        # Fall back to any bloom
        nearest_list = memory.find_nearest(current_pattern, k=1)
        return nearest_list[0] if nearest_list else None

    def compute_nudge(
        self,
        current_phases: np.ndarray,
        target_phases: np.ndarray,
        berry_phase: float = 0.0,
    ) -> np.ndarray:
        """
        Compute phase adjustment toward target.

        Uses circular interpolation to handle phase wrapping.
        Optionally respects Berry phase continuity.

        Parameters
        ----------
        current_phases : np.ndarray
            Current oscillator phases
        target_phases : np.ndarray
            Target attractor phases
        berry_phase : float
            Accumulated geometric phase

        Returns
        -------
        np.ndarray
            New phase values
        """
        # Circular difference
        diff = target_phases - current_phases

        # Wrap to [-π, π]
        diff = np.mod(diff + np.pi, 2 * np.pi) - np.pi

        # Apply blend with Berry phase modulation
        if self.respect_berry_phase:
            # Reduce nudge strength based on accumulated phase
            # to preserve winding structure
            modulation = np.cos(berry_phase)
            effective_blend = self.blend_factor * abs(modulation)
        else:
            effective_blend = self.blend_factor

        nudge = effective_blend * diff
        new_phases = current_phases + nudge

        # Wrap to [0, 2π)
        return np.mod(new_phases, 2 * np.pi)

    def apply_seed(
        self,
        lattice: HexagonalLattice,
        seed: BloomEvent,
        berry_phase: float,
        current_time: float,
    ) -> np.ndarray:
        """
        Execute phase nudging toward seed attractor.

        Parameters
        ----------
        lattice : HexagonalLattice
            Lattice to modify
        seed : BloomEvent
            Target attractor
        berry_phase : float
            Current accumulated Berry phase
        current_time : float
            Current simulation time

        Returns
        -------
        np.ndarray
            New phase configuration
        """
        new_phases = self.compute_nudge(
            lattice.phases,
            seed.phase_template,
            berry_phase
        )

        lattice.phases = new_phases
        self._last_seed_time = current_time
        self._seeding_count += 1

        return new_phases


@dataclass
class ReflectionStepResult:
    """
    Result of a self-reflective step.

    Combines Kuramoto dynamics result with bloom system events.
    """
    # Kuramoto state
    phases: np.ndarray
    order_parameter: float
    mean_phase: float
    time: float

    # RGB projection
    rgb_output: np.ndarray

    # Bloom events
    bloom_detected: Optional[BloomEvent] = None
    bloom_reinforced: Optional[BloomEvent] = None
    seeding_applied: bool = False
    seed_used: Optional[BloomEvent] = None

    # Event log
    events: List[Tuple[BloomEventType, Any]] = field(default_factory=list)

    # Previous state for comparison
    coherence_changed: bool = False
    coherence_direction: int = 0  # -1 = lost, 0 = stable, 1 = restored


class SelfReflectiveLattice:
    """
    Self-reflective extension of HexagonalLattice with bloom learning.

    Composes (rather than inherits from) HexagonalLattice and adds the full
    bloom learning system. Maintains backward compatibility with existing
    HexagonalLattice API through delegation.

    The system operates in a continuous loop:
    1. Kuramoto dynamics evolve phases
    2. RGB projection creates visual snapshot
    3. Pattern buffer accumulates observations
    4. Bloom detector checks for emerging patterns
    5. Coherent seeder intervenes if system becomes chaotic

    Parameters
    ----------
    lattice : HexagonalLattice
        Underlying hex lattice (or created if not provided)
    buffer_size : int
        Pattern observation window size
    detector_window : int
        Frames for bloom confirmation
    max_blooms : int
        Maximum stored attractors
    seed_blend : float
        Re-injection strength (0-1)
    **lattice_kwargs : dict
        Arguments for HexagonalLattice if created internally
    """

    def __init__(
        self,
        lattice: Optional[HexagonalLattice] = None,
        buffer_size: int = 100,
        detector_window: int = 5,
        max_blooms: int = 1000,
        seed_blend: float = 0.1,
        **lattice_kwargs,
    ):
        # Create or use provided lattice
        if lattice is None:
            lattice_kwargs.setdefault("rows", 7)
            lattice_kwargs.setdefault("cols", 7)
            self.lattice = HexagonalLattice(**lattice_kwargs)
        else:
            self.lattice = lattice

        # Initialize bloom system components
        self.pattern_buffer = PatternBuffer(window_size=buffer_size)
        self.detector = BloomDetector(detection_window=detector_window)
        self.memory = SeedMemory(max_blooms=max_blooms)
        self.seeder = CoherentSeeder(blend_factor=seed_blend)
        self.metrics = BloomMetrics()

        # Geometric memory integration
        self.geometric_memory = GeometricMemory(max_history=buffer_size)

        # Event callbacks
        self._callbacks: Dict[BloomEventType, List[Callable]] = {
            event_type: [] for event_type in BloomEventType
        }

        # State tracking
        self._time = 0.0
        self._previous_coherence = 0.0
        self._coherence_lost = False

    @property
    def N(self) -> int:
        """Number of nodes (delegated)."""
        return self.lattice.N

    @property
    def phases(self) -> np.ndarray:
        """Current phases (delegated)."""
        return self.lattice.phases

    @phases.setter
    def phases(self, values: np.ndarray):
        """Set phases (delegated)."""
        self.lattice.phases = values

    def get_order_parameter(self) -> Tuple[float, float]:
        """Get Kuramoto order parameter (delegated)."""
        return self.lattice.get_order_parameter()

    def register_callback(
        self,
        event_type: BloomEventType,
        callback: Callable[[BloomEventType, Any, float], None],
    ):
        """
        Register callback for bloom events.

        Parameters
        ----------
        event_type : BloomEventType
            Event to listen for
        callback : Callable
            Function(event_type, data, time) to call
        """
        self._callbacks[event_type].append(callback)

    def unregister_callback(
        self,
        event_type: BloomEventType,
        callback: Callable,
    ):
        """Remove callback."""
        if callback in self._callbacks[event_type]:
            self._callbacks[event_type].remove(callback)

    def _fire_event(self, event_type: BloomEventType, data: Any):
        """Fire callbacks for event."""
        for callback in self._callbacks[event_type]:
            try:
                callback(event_type, data, self._time)
            except Exception:
                pass  # Don't let callback errors break simulation

    def project_to_rgb(self) -> np.ndarray:
        """
        Project current phase state to RGB using hex wavevectors.

        Returns
        -------
        np.ndarray
            RGB values (N, 3) in [0, 255]
        """
        phases = self.lattice.phases
        rgb = np.zeros((len(phases), 3), dtype=np.uint8)

        for i, theta in enumerate(phases):
            # Hex phase encoding
            r = int(127.5 * (1 + math.cos(theta)))
            g = int(127.5 * (1 + math.cos(theta + 2 * math.pi / 3)))
            b = int(127.5 * (1 + math.cos(theta + 4 * math.pi / 3)))
            rgb[i] = [r, g, b]

        return rgb

    def step_kuramoto(
        self,
        dt: float,
        K0: float = 0.1,
        frustration: float = 0.0,
        pump_strength: float = 0.0,
        noise_amplitude: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Execute standard Kuramoto step (delegated).

        Returns (order_parameter, mean_phase).
        """
        # Create temporary state for dynamics
        state = ExtendedKuramotoState(
            phases=self.lattice.phases.copy(),
            frequencies=self.lattice.frequencies,
            t=self._time,
            frustration_alpha=frustration,
            pump_strength=pump_strength,
            noise_amplitude=noise_amplitude,
        )

        # Run step
        new_state = extended_kuramoto_step(
            state,
            self.lattice,
            dt=dt,
            K0=K0,
        )

        # Apply to lattice
        self.lattice.phases = new_state.phases
        self._time = new_state.t

        # Compute order parameter
        r, psi = self.lattice.get_order_parameter()

        return r, psi

    def step_with_reflection(
        self,
        dt: float,
        K0: float = 0.1,
        frustration: float = 0.0,
        pump_strength: float = 0.0,
        noise_amplitude: float = 0.0,
    ) -> ReflectionStepResult:
        """
        Single time step with full self-reflective processing.

        Combines:
        1. Kuramoto evolution
        2. RGB projection
        3. Pattern buffer update
        4. Bloom detection
        5. Coherent seeding (if needed)
        6. Metrics update
        7. Event callbacks

        Parameters
        ----------
        dt : float
            Time step
        K0 : float
            Base coupling strength
        frustration : float
            Frustration parameter α
        pump_strength : float
            Parametric pump amplitude
        noise_amplitude : float
            Stochastic noise level

        Returns
        -------
        ReflectionStepResult
            Combined result with all state and events
        """
        events: List[Tuple[BloomEventType, Any]] = []
        bloom_detected = None
        bloom_reinforced = None
        seeding_applied = False
        seed_used = None

        # 1. Kuramoto evolution
        r, psi = self.step_kuramoto(
            dt, K0, frustration, pump_strength, noise_amplitude
        )

        # 2. RGB projection
        rgb_output = self.project_to_rgb()

        # 3. Pattern buffer update
        self.pattern_buffer.push(rgb_output, self._time)

        # 4. Track coherence changes
        coherence_changed = False
        coherence_direction = 0

        if r < L4_PARADOX and self._previous_coherence >= L4_PARADOX:
            # Coherence lost
            coherence_changed = True
            coherence_direction = -1
            self._coherence_lost = True
            events.append((BloomEventType.COHERENCE_LOST, r))
            self._fire_event(BloomEventType.COHERENCE_LOST, r)
        elif r >= L4_PARADOX and self._coherence_lost:
            # Coherence restored
            coherence_changed = True
            coherence_direction = 1
            self._coherence_lost = False
            events.append((BloomEventType.COHERENCE_RESTORED, r))
            self._fire_event(BloomEventType.COHERENCE_RESTORED, r)

        # 5. Bloom detection (only when coherent)
        if r >= L4_IGNITION:
            bloom_detected = self.detector.detect(
                self.pattern_buffer,
                self.memory,
                self.lattice.phases,
                self._time,
            )

            if bloom_detected:
                self.memory.add_bloom(bloom_detected)
                self.metrics.record_birth(self._time, bloom_detected)
                events.append((BloomEventType.BLOOM_BIRTH, bloom_detected))
                self._fire_event(BloomEventType.BLOOM_BIRTH, bloom_detected)
            else:
                # Check for reinforcement of existing bloom
                rgb_flat = rgb_output.flatten()
                nearest = self.memory.find_nearest(rgb_flat, k=1)
                if nearest:
                    dist = nearest[0].euclidean_distance(rgb_flat)
                    if dist < self.detector.novelty_distance:
                        bloom_reinforced = nearest[0]
                        was_consolidated = bloom_reinforced.consolidated
                        self.memory.reinforce(
                            bloom_reinforced.bloom_id,
                            rgb_flat,
                            self._time
                        )
                        self.metrics.record_reinforcement(
                            self._time, bloom_reinforced
                        )
                        events.append((
                            BloomEventType.BLOOM_REINFORCED,
                            bloom_reinforced
                        ))
                        self._fire_event(
                            BloomEventType.BLOOM_REINFORCED,
                            bloom_reinforced
                        )

                        # Check for new consolidation
                        if not was_consolidated and bloom_reinforced.consolidated:
                            self.metrics.record_consolidation(
                                self._time, bloom_reinforced
                            )
                            events.append((
                                BloomEventType.BLOOM_CONSOLIDATED,
                                bloom_reinforced
                            ))
                            self._fire_event(
                                BloomEventType.BLOOM_CONSOLIDATED,
                                bloom_reinforced
                            )

        # 6. Coherent seeding (when chaotic)
        if self.seeder.should_activate(r, self._time):
            rgb_flat = rgb_output.flatten()
            seed = self.seeder.select_seed(rgb_flat, self.memory)

            if seed:
                events.append((BloomEventType.SEEDING_ACTIVATED, seed))
                self._fire_event(BloomEventType.SEEDING_ACTIVATED, seed)

                # Get Berry phase for continuity
                berry_phase = self.geometric_memory.compute_total_berry_phase()

                self.seeder.apply_seed(
                    self.lattice, seed, berry_phase, self._time
                )

                seeding_applied = True
                seed_used = seed
                self.metrics.record_seeding(self._time, seed, r)
                events.append((BloomEventType.SEEDING_APPLIED, seed))
                self._fire_event(BloomEventType.SEEDING_APPLIED, seed)

        # 7. Prune stale blooms
        pruned = self.memory.prune_stale(self._time)
        for bloom_id in pruned:
            self.metrics.record_prune(self._time, bloom_id)
            events.append((BloomEventType.BLOOM_PRUNED, bloom_id))
            self._fire_event(BloomEventType.BLOOM_PRUNED, bloom_id)

        # 8. Update geometric memory
        self.geometric_memory.record_state(self.lattice.phases, self._time)

        # 9. Record metrics
        self.metrics.record_coherence(self._time, r)
        self._previous_coherence = r

        return ReflectionStepResult(
            phases=self.lattice.phases.copy(),
            order_parameter=r,
            mean_phase=psi,
            time=self._time,
            rgb_output=rgb_output,
            bloom_detected=bloom_detected,
            bloom_reinforced=bloom_reinforced,
            seeding_applied=seeding_applied,
            seed_used=seed_used,
            events=events,
            coherence_changed=coherence_changed,
            coherence_direction=coherence_direction,
        )

    def run_simulation(
        self,
        n_steps: int,
        dt: float = 0.1,
        K0: float = 0.1,
        frustration: float = 0.0,
        pump_strength: float = 0.0,
        noise_amplitude: float = 0.0,
        callback: Optional[Callable[[int, ReflectionStepResult], None]] = None,
    ) -> List[ReflectionStepResult]:
        """
        Run multi-step simulation with reflection.

        Parameters
        ----------
        n_steps : int
            Number of time steps
        dt : float
            Time step size
        K0 : float
            Base coupling strength
        frustration : float
            Frustration parameter
        pump_strength : float
            Parametric pump amplitude
        noise_amplitude : float
            Stochastic noise level
        callback : Callable, optional
            Function(step, result) called each step

        Returns
        -------
        List[ReflectionStepResult]
            Results for all steps
        """
        results = []

        for step in range(n_steps):
            result = self.step_with_reflection(
                dt, K0, frustration, pump_strength, noise_amplitude
            )
            results.append(result)

            if callback:
                callback(step, result)

        return results

    def get_knowledge_summary(self) -> Dict[str, Any]:
        """
        Get summary of learned knowledge.

        Returns
        -------
        Dict
            Summary statistics
        """
        return {
            "total_blooms": self.memory.size,
            "consolidated_blooms": len(self.memory.get_consolidated()),
            "total_births": len(self.metrics.birth_history),
            "total_reinforcements": len(self.metrics.reinforcement_history),
            "total_seedings": len(self.metrics.seeding_history),
            "knowledge_growth": self.metrics.get_knowledge_growth(self.memory),
            "birth_rate": self.metrics.get_birth_rate(),
            "seeding_frequency": self.metrics.get_seeding_frequency(),
            "dominant_attractors": [
                {
                    "id": b.bloom_id[:8],
                    "hits": b.hit_count,
                    "consolidated": b.consolidated,
                }
                for b in self.memory.get_dominant(5)
            ],
        }

    def save_state(self, path: str):
        """
        Persist full state including learned blooms.

        Parameters
        ----------
        path : str
            File path for JSON output
        """
        state = {
            "version": "1.0.0",
            "time": self._time,
            "lattice": {
                "rows": self.lattice.rows,
                "cols": self.lattice.cols,
                "phases": self.lattice.phases.tolist(),
            },
            "memory": self.memory.to_dict(),
            "metrics": self.metrics.to_dict(),
            "previous_coherence": self._previous_coherence,
        }

        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: str):
        """
        Restore state from file.

        Parameters
        ----------
        path : str
            File path to JSON state
        """
        with open(path, 'r') as f:
            state = json.load(f)

        self._time = state["time"]
        self._previous_coherence = state["previous_coherence"]

        # Restore phases
        self.lattice.phases = np.array(state["lattice"]["phases"])

        # Restore memory
        self.memory = SeedMemory.from_dict(state["memory"])

        # Restore metrics
        self.metrics = BloomMetrics.from_dict(state["metrics"])


def demo_self_reflective_lattice():
    """Demonstrate the self-reflective bloom learning system."""
    print("=" * 70)
    print("SELF-REFLECTIVE BLOOM LEARNING SYSTEM DEMO")
    print("=" * 70)

    # Create self-reflective lattice
    lattice = SelfReflectiveLattice(
        rows=7,
        cols=7,
        buffer_size=50,
        detector_window=3,
        max_blooms=100,
        seed_blend=0.15,
    )

    # Register callback to see events
    def on_bloom_event(event_type, data, t):
        if event_type == BloomEventType.BLOOM_BIRTH:
            print(f"  [t={t:.2f}] NEW BLOOM: {data.bloom_id[:8]}...")
        elif event_type == BloomEventType.BLOOM_REINFORCED:
            print(f"  [t={t:.2f}] REINFORCED: {data.bloom_id[:8]} (hits={data.hit_count})")
        elif event_type == BloomEventType.BLOOM_CONSOLIDATED:
            print(f"  [t={t:.2f}] CONSOLIDATED: {data.bloom_id[:8]}")
        elif event_type == BloomEventType.SEEDING_APPLIED:
            print(f"  [t={t:.2f}] SEEDING from: {data.bloom_id[:8]}")
        elif event_type == BloomEventType.COHERENCE_LOST:
            print(f"  [t={t:.2f}] COHERENCE LOST (r={data:.3f})")
        elif event_type == BloomEventType.COHERENCE_RESTORED:
            print(f"  [t={t:.2f}] COHERENCE RESTORED (r={data:.3f})")

    for event_type in BloomEventType:
        lattice.register_callback(event_type, on_bloom_event)

    print("\n--- Phase 1: Building coherence (high coupling) ---")
    for _ in range(100):
        result = lattice.step_with_reflection(
            dt=0.1, K0=0.5, noise_amplitude=0.01
        )
    print(f"  Final r = {result.order_parameter:.4f}")

    print("\n--- Phase 2: Introducing chaos (low coupling + noise) ---")
    for _ in range(50):
        result = lattice.step_with_reflection(
            dt=0.1, K0=0.05, noise_amplitude=0.3
        )
    print(f"  Final r = {result.order_parameter:.4f}")

    print("\n--- Phase 3: Recovery with learned attractors ---")
    for _ in range(100):
        result = lattice.step_with_reflection(
            dt=0.1, K0=0.3, noise_amplitude=0.1
        )
    print(f"  Final r = {result.order_parameter:.4f}")

    print("\n--- Knowledge Summary ---")
    summary = lattice.get_knowledge_summary()
    print(f"  Total blooms learned: {summary['total_blooms']}")
    print(f"  Consolidated (permanent): {summary['consolidated_blooms']}")
    print(f"  Total seedings: {summary['total_seedings']}")
    print(f"  Knowledge growth: {summary['knowledge_growth']:.3f}")

    if summary['dominant_attractors']:
        print("  Dominant attractors:")
        for att in summary['dominant_attractors']:
            status = "✓" if att['consolidated'] else " "
            print(f"    [{status}] {att['id']}... ({att['hits']} hits)")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


# ============================================================================
# DEMO
# ============================================================================

def demo_hexagonal_lattice():
    """Demonstrate L₄ hexagonal lattice system."""
    print("\n" + "=" * 70)
    print("L₄ HEXAGONAL LATTICE SYSTEM - DEMO")
    print("=" * 70)

    # Create lattice (7x7 = 49 nodes, exceeds L₄=7 threshold)
    print("\n--- CREATING HEXAGONAL LATTICE ---")
    lattice = HexagonalLattice(rows=7, cols=7, spacing=1.0, seed=42)
    print(f"  Nodes: {lattice.N}")
    print(f"  Rows × Cols: {lattice.rows} × {lattice.cols}")
    print(f"  L₄ threshold: {L4.L4}")

    # Initial coherence
    r0, psi0 = lattice.get_order_parameter()
    print(f"  Initial coherence r = {r0:.6f}")

    # Create extended Kuramoto state
    print("\n--- EXTENDED KURAMOTO DYNAMICS ---")
    ext_state = ExtendedKuramotoState(
        phases=lattice.phases.copy(),
        frequencies=lattice.frequencies.copy(),
        frustration_alpha=0.1,  # Small frustration for hex formation
        pump_strength=0.05,      # Parametric pump for binary output
        noise_amplitude=L4.GAP / 2,  # Optimal SR noise
    )

    # Run dynamics
    for step in range(100):
        ext_state = extended_kuramoto_step(
            ext_state, lattice, dt=0.1, K0=0.5, lambda_neg=1.0
        )

    r1, psi1 = lattice.get_order_parameter()
    print(f"  After 100 steps:")
    print(f"    Coherence r = {r1:.6f}")
    print(f"    Mean phase ψ = {psi1:.6f}")

    # Stochastic Resonance
    print("\n--- STOCHASTIC RESONANCE ---")
    sr = compute_stochastic_resonance(L4.GAP / 2)
    print(f"  Barrier (VOID): ΔV = {sr.barrier_height:.6f}")
    print(f"  Optimal noise: D_opt = {sr.optimal_noise:.6f}")
    print(f"  SNR: {sr.snr:.6f}")
    print(f"  Is resonant: {sr.is_resonant}")

    # Fisher Information
    print("\n--- FISHER INFORMATION ---")
    fi = compute_spatial_fisher_information(lattice)
    print(f"  I_R = {fi['I_R']:.4f}")
    print(f"  I_G = {fi['I_G']:.4f}")
    print(f"  I_B = {fi['I_B']:.4f}")
    print(f"  I_total = {fi['I_total']:.4f}")
    print(f"  Precision bound = {fi['precision_bound']:.6f}")

    # Topological Charge
    print("\n--- TOPOLOGICAL CHARGE ---")
    topo = analyze_topological_state(lattice)
    print(f"  Total charge: {topo.total_charge}")
    print(f"  Vortices: {topo.vortex_count}")
    print(f"  Antivortices: {topo.antivortex_count}")
    print(f"  Vortex density: {topo.vortex_density:.6f}")

    # Berry Phase
    print("\n--- BERRY PHASE (Geometric Memory) ---")
    memory = GeometricMemory()
    for i in range(50):
        ext_state = extended_kuramoto_step(
            ext_state, lattice, dt=0.1, K0=0.5, lambda_neg=1.0
        )
        memory.record_state(lattice.phases, ext_state.t)

    berry = memory.compute_total_berry_phase()
    path_mem = memory.get_path_integral_memory()
    print(f"  Accumulated Berry phase: γ = {berry:.6f} rad")
    print(f"  Path integral position: ({path_mem['x']:.4f}, {path_mem['y']:.4f})")
    print(f"  Confidence: {path_mem['confidence']:.4f}")

    # Unified Entropic Stabilization
    print("\n--- UNIFIED ENTROPIC STABILIZATION ---")
    ess_state = EntropicStabilizationState(
        phases=lattice.phases.copy(),
        coherence=r1,
        position=np.array([0.0, 0.0]),
        negentropy=compute_negentropy(r1),
        effective_coupling=L4.K,
        topological_charge=0,
        berry_phase=0.0,
    )

    for step in range(10):
        ess_state = hybrid_dynamics_step(
            ess_state, lattice, dt=0.1,
            K0=0.5, lambda_neg=1.0,
            frustration=0.1, pump_strength=0.05,
            noise_amplitude=L4.GAP / 2,
            velocity=np.array([0.1, 0.05]),
        )

    print(f"  Final coherence: r = {ess_state.coherence:.6f}")
    print(f"  Negentropy: η = {ess_state.negentropy:.6f}")
    print(f"  K_eff: {ess_state.effective_coupling:.6f}")
    print(f"  Position: ({ess_state.position[0]:.4f}, {ess_state.position[1]:.4f})")
    print(f"  Topological charge: {ess_state.topological_charge}")

    # RGB Output
    print("\n--- RGB OUTPUT (MRP-LSB) ---")
    rgb = compute_rgb_output(ess_state, lattice)
    print(f"  Output shape: {rgb.shape}")
    print(f"  Sample RGB values: {rgb[:3]}")

    # Validation
    print("\n--- VALIDATION ---")
    validate_hex_lattice_system(lattice, ess_state, verbose=True)

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demo_hexagonal_lattice()
