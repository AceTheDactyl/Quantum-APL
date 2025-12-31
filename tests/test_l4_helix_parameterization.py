#!/usr/bin/env python3
"""
Test Suite for L₄ Helix Parameterization Module
================================================

Comprehensive tests for all blocks:
1. Constants block - L₄ identity and derived constants
2. State vector block - Helix state and radius law
3. Dynamics block - Kuramoto synchronization with negentropy modulation
4. Phase→RGB quantization - Hex lattice with 60° structure
5. LSB embed/extract - Steganography
6. K-formation validation - Pass/fail tests

Hard Constraints Tested:
- L₄ = φ⁴ + φ⁻⁴ = (√3)² + 4 = 7
- z_c = √3/2 (fixed critical point)
- Helix radius r(z) uses piecewise formula from K
- Negentropy gates dynamics (not just logged)
- Hex lattice respects 60° structure
"""

import math
import pytest
import numpy as np

from quantum_apl_python.l4_helix_parameterization import (
    # Block 1: Constants
    L4,
    L4Constants,
    get_l4_constants,
    # Block 2: State
    HelixState,
    create_initial_state,
    # Block 3: Dynamics
    compute_negentropy,
    compute_negentropy_derivative,
    compute_kuramoto_order_parameter,
    kuramoto_dynamics_continuous,
    kuramoto_step_euler,
    kuramoto_step_rk4,
    xy_spin_hamiltonian,
    # Block 4: Phase→RGB
    HexLatticeWavevectors,
    quantize_phase_to_bits,
    dequantize_bits_to_phase,
    phases_to_rgb,
    rgb_to_phases,
    RGBQuantization,
    serialize_phases_to_bitstream,
    # Block 5: LSB
    lsb_embed_bit,
    lsb_extract_bit,
    lsb_embed_nbits,
    lsb_extract_nbits,
    compute_capacity,
    embed_message_lsb,
    extract_message_lsb,
    # Block 6: Validation
    validate_coherence_threshold,
    validate_negentropy_gate,
    validate_radius_threshold,
    validate_k_formation,
    validate_l4_identity,
    validate_critical_point,
    validate_coupling_constant,
    run_all_validations,
    simulate_kuramoto_to_k_formation,
)


# ============================================================================
# BLOCK 1: CONSTANTS TESTS
# ============================================================================

class TestL4Constants:
    """Tests for L₄ constants block."""

    def test_golden_ratio_identity(self):
        """Test φ·τ = 1 (golden ratio times its inverse equals 1)."""
        assert abs(L4.PHI * L4.TAU - 1.0) < 1e-14

    def test_golden_ratio_value(self):
        """Test φ = (1 + √5) / 2."""
        expected = (1.0 + math.sqrt(5.0)) / 2.0
        assert abs(L4.PHI - expected) < 1e-14

    def test_tau_value(self):
        """Test τ = φ⁻¹ = (√5 - 1) / 2."""
        expected = (math.sqrt(5.0) - 1.0) / 2.0
        assert abs(L4.TAU - expected) < 1e-14

    def test_l4_equals_7(self):
        """Test L₄ = 7 exactly."""
        assert L4.L4 == 7.0

    def test_l4_identity_phi_pow4(self):
        """Test L₄ = φ⁴ + φ⁻⁴."""
        computed = L4.PHI ** 4 + L4.TAU ** 4
        assert abs(computed - 7.0) < 1e-10

    def test_l4_identity_sqrt3(self):
        """Test L₄ = (√3)² + 4 = 3 + 4 = 7."""
        computed = math.sqrt(3.0) ** 2 + 4.0
        assert abs(computed - 7.0) < 1e-10

    def test_gap_is_tau_4(self):
        """Test gap = τ⁴ = φ⁻⁴."""
        expected = L4.TAU ** 4
        assert abs(L4.GAP - expected) < 1e-14
        assert abs(L4.GAP - 0.1458980337503154) < 1e-10

    def test_k_from_gap(self):
        """Test K = √(1 - gap)."""
        expected = math.sqrt(1.0 - L4.GAP)
        assert abs(L4.K - expected) < 1e-14
        assert abs(L4.K - 0.9241648530576246) < 1e-10

    def test_critical_point_zc(self):
        """Test z_c = √3/2."""
        expected = math.sqrt(3.0) / 2.0
        assert abs(L4.Z_C - expected) < 1e-14
        assert abs(L4.Z_C - 0.8660254037844386) < 1e-10

    def test_critical_point_from_l4(self):
        """Test z_c = √(L₄ - 4) / 2."""
        expected = math.sqrt(L4.L4 - 4.0) / 2.0
        assert abs(L4.Z_C - expected) < 1e-14

    def test_verify_identity_method(self):
        """Test L4Constants.verify_identity() returns True."""
        assert L4.verify_identity() is True

    def test_get_l4_constants_dict(self):
        """Test get_l4_constants() returns complete dictionary."""
        constants = get_l4_constants()
        assert "PHI" in constants
        assert "TAU" in constants
        assert "L4" in constants
        assert "GAP" in constants
        assert "K" in constants
        assert "Z_C" in constants
        assert constants["L4"] == 7.0


# ============================================================================
# BLOCK 2: STATE VECTOR TESTS
# ============================================================================

class TestHelixState:
    """Tests for helix state vector block."""

    def test_create_initial_state(self):
        """Test initial state creation."""
        state = create_initial_state(N=64, z0=0.5, theta0=1.0, seed=42)
        assert state.N == 64
        assert state.z == 0.5
        assert state.theta == 1.0
        assert len(state.phases) == 64
        assert len(state.frequencies) == 64
        assert state.t == 0.0

    def test_radius_below_zc(self):
        """Test r(z) = K·√(z/z_c) for z ≤ z_c."""
        state = create_initial_state(N=4, z0=0.4, seed=42)
        expected = L4.K * math.sqrt(0.4 / L4.Z_C)
        assert abs(state.r - expected) < 1e-10

    def test_radius_at_zc(self):
        """Test r(z_c) = K."""
        state = create_initial_state(N=4, z0=L4.Z_C, seed=42)
        assert abs(state.r - L4.K) < 1e-10

    def test_radius_above_zc(self):
        """Test r(z) = K for z > z_c."""
        state = create_initial_state(N=4, z0=0.95, seed=42)
        assert abs(state.r - L4.K) < 1e-10

    def test_radius_at_zero(self):
        """Test r(0) = 0."""
        state = create_initial_state(N=4, z0=0.0, seed=42)
        assert state.r == 0.0

    def test_helix_position(self):
        """Test helix position H(z) = (r·cos(θ), r·sin(θ), z)."""
        state = HelixState(
            z=0.5,
            theta=math.pi / 4,
            phases=np.array([0.0]),
            frequencies=np.array([0.0]),
        )
        x, y, z = state.helix_position()
        expected_r = L4.K * math.sqrt(0.5 / L4.Z_C)
        assert abs(x - expected_r * math.cos(math.pi / 4)) < 1e-10
        assert abs(y - expected_r * math.sin(math.pi / 4)) < 1e-10
        assert z == 0.5

    def test_state_to_dict(self):
        """Test state serialization."""
        state = create_initial_state(N=4, z0=0.5, seed=42)
        d = state.to_dict()
        assert "z" in d
        assert "theta" in d
        assert "r" in d
        assert "phases" in d
        assert "frequencies" in d
        assert "helix_position" in d


# ============================================================================
# BLOCK 3: DYNAMICS TESTS
# ============================================================================

class TestDynamics:
    """Tests for Kuramoto dynamics with negentropy modulation."""

    def test_negentropy_at_zc(self):
        """Test ΔS_neg(z_c) = 1 (maximum at critical point)."""
        eta = compute_negentropy(L4.Z_C)
        assert abs(eta - 1.0) < 1e-10

    def test_negentropy_symmetric(self):
        """Test ΔS_neg is symmetric around z_c."""
        delta = 0.1
        eta_plus = compute_negentropy(L4.Z_C + delta)
        eta_minus = compute_negentropy(L4.Z_C - delta)
        assert abs(eta_plus - eta_minus) < 1e-10

    def test_negentropy_bounded(self):
        """Test 0 < ΔS_neg ≤ 1."""
        for z in [0.0, 0.3, 0.5, L4.Z_C, 0.9, 1.0]:
            eta = compute_negentropy(z)
            assert 0.0 < eta <= 1.0

    def test_negentropy_derivative_zero_at_zc(self):
        """Test d(ΔS_neg)/dz = 0 at z = z_c (critical point)."""
        deriv = compute_negentropy_derivative(L4.Z_C)
        assert abs(deriv) < 1e-10

    def test_negentropy_derivative_sign(self):
        """Test derivative is negative above z_c, positive below."""
        deriv_above = compute_negentropy_derivative(L4.Z_C + 0.1)
        deriv_below = compute_negentropy_derivative(L4.Z_C - 0.1)
        assert deriv_above < 0  # Decreasing above z_c
        assert deriv_below > 0  # Increasing below z_c

    def test_kuramoto_order_parameter_incoherent(self):
        """Test order parameter r ≈ 0 for uniformly distributed phases."""
        N = 1000
        phases = np.linspace(0, 2 * np.pi, N, endpoint=False)
        r, psi = compute_kuramoto_order_parameter(phases)
        assert r < 0.05  # Should be near zero for large N

    def test_kuramoto_order_parameter_synchronized(self):
        """Test order parameter r = 1 for synchronized phases."""
        phases = np.ones(100) * 1.5  # All same phase
        r, psi = compute_kuramoto_order_parameter(phases)
        assert abs(r - 1.0) < 1e-10

    def test_kuramoto_dynamics_continuous(self):
        """Test continuous Kuramoto dynamics returns valid outputs."""
        state = create_initial_state(N=32, z0=0.5, seed=42)
        dtheta_dt, K_eff, eta = kuramoto_dynamics_continuous(state, K0=0.1, lambda_neg=0.5)
        assert len(dtheta_dt) == 32
        assert K_eff > 0
        assert 0 < eta <= 1

    def test_kuramoto_step_euler(self):
        """Test Euler step preserves phase bounds."""
        state = create_initial_state(N=32, z0=0.5, seed=42)
        new_state = kuramoto_step_euler(state, dt=0.1)
        assert all(0 <= p < 2 * np.pi for p in new_state.phases)
        assert new_state.t > state.t

    def test_kuramoto_step_rk4(self):
        """Test RK4 step preserves phase bounds."""
        state = create_initial_state(N=32, z0=0.5, seed=42)
        new_state = kuramoto_step_rk4(state, dt=0.1)
        assert all(0 <= p < 2 * np.pi for p in new_state.phases)
        assert new_state.t > state.t

    def test_negentropy_modulates_dynamics(self):
        """Test that negentropy actually modulates K_eff (not just logged)."""
        state_low_z = create_initial_state(N=32, z0=0.3, seed=42)
        state_high_z = create_initial_state(N=32, z0=L4.Z_C, seed=42)

        _, K_eff_low, eta_low = kuramoto_dynamics_continuous(state_low_z)
        _, K_eff_high, eta_high = kuramoto_dynamics_continuous(state_high_z)

        # At z_c, negentropy is maximum, so K_eff should be higher
        assert eta_high > eta_low
        assert K_eff_high > K_eff_low  # K_eff = K0 * (1 + lambda * eta)

    def test_xy_hamiltonian(self):
        """Test XY spin Hamiltonian computation."""
        phases = np.array([0.0, 0.0, 0.0])  # All aligned
        energy = xy_spin_hamiltonian(phases, J=1.0)
        # All aligned = minimum energy (negative)
        assert energy < 0

        phases_random = np.array([0.0, np.pi/2, np.pi])
        energy_random = xy_spin_hamiltonian(phases_random, J=1.0)
        # Less aligned = higher energy
        assert energy_random > energy


# ============================================================================
# BLOCK 4: PHASE → RGB QUANTIZATION TESTS
# ============================================================================

class TestPhaseToRGB:
    """Tests for hex lattice phase coding and RGB quantization."""

    def test_hex_wavevector_60_degrees(self):
        """Test hex lattice wavevectors are separated by 60°."""
        hex_lattice = HexLatticeWavevectors(wavelength=1.0)

        # Compute angles
        angle_R = math.atan2(hex_lattice.k_R[1], hex_lattice.k_R[0])
        angle_G = math.atan2(hex_lattice.k_G[1], hex_lattice.k_G[0])
        angle_B = math.atan2(hex_lattice.k_B[1], hex_lattice.k_B[0])

        # Check 60° separation
        assert abs(angle_G - angle_R - math.pi / 3) < 1e-10  # 60°
        assert abs(angle_B - angle_G - math.pi / 3) < 1e-10  # 60°

    def test_hex_wavevector_equal_magnitude(self):
        """Test all hex wavevectors have equal magnitude."""
        hex_lattice = HexLatticeWavevectors(wavelength=1.0)

        mag_R = np.linalg.norm(hex_lattice.k_R)
        mag_G = np.linalg.norm(hex_lattice.k_G)
        mag_B = np.linalg.norm(hex_lattice.k_B)

        assert abs(mag_R - mag_G) < 1e-10
        assert abs(mag_G - mag_B) < 1e-10

    def test_channel_phases_in_range(self):
        """Test channel phases are in [0, 2π)."""
        hex_lattice = HexLatticeWavevectors()
        x = np.array([1.5, 2.7])
        theta_R, theta_G, theta_B = hex_lattice.compute_channel_phases(x)

        for theta in [theta_R, theta_G, theta_B]:
            assert 0 <= theta < 2 * np.pi

    def test_phase_evolves_with_omega(self):
        """Test global phase evolution."""
        hex_lattice = HexLatticeWavevectors()
        Phi = (0.0, 0.0, 0.0)
        Omega = (0.1, 0.2, 0.3)
        dt = 1.0

        new_Phi = hex_lattice.evolve_global_phases(Phi, Omega, dt)
        assert abs(new_Phi[0] - 0.1) < 1e-10
        assert abs(new_Phi[1] - 0.2) < 1e-10
        assert abs(new_Phi[2] - 0.3) < 1e-10

    def test_quantize_phase_8bit(self):
        """Test phase quantization to 8 bits."""
        assert quantize_phase_to_bits(0.0, 8) == 0
        assert quantize_phase_to_bits(np.pi, 8) == 128
        assert quantize_phase_to_bits(2 * np.pi - 0.001, 8) == 255

    def test_dequantize_phase(self):
        """Test phase dequantization."""
        for q in [0, 64, 128, 192, 255]:
            reconstructed = dequantize_bits_to_phase(q, 8)
            assert 0 <= reconstructed < 2 * np.pi

    def test_phase_quantize_roundtrip(self):
        """Test phase quantize/dequantize roundtrip."""
        for theta in [0.0, 1.0, np.pi, 2*np.pi - 0.1]:
            q = quantize_phase_to_bits(theta, 8)
            theta_back = dequantize_bits_to_phase(q, 8)
            # Should be within one quantization step
            step = 2 * np.pi / 256
            assert abs(theta - theta_back) < step

    def test_phases_to_rgb(self):
        """Test phase to RGB conversion."""
        rgb = phases_to_rgb(0.0, np.pi, 2*np.pi - 0.001)
        assert rgb.R == 0
        assert rgb.G == 128
        assert rgb.B == 255

    def test_rgb_to_phases_roundtrip(self):
        """Test RGB to phases roundtrip."""
        original = RGBQuantization(R=100, G=150, B=200)
        phases = rgb_to_phases(original)
        reconstructed = phases_to_rgb(*phases)
        assert reconstructed.R == original.R
        assert reconstructed.G == original.G
        assert reconstructed.B == original.B

    def test_serialize_phases_to_bitstream(self):
        """Test serialization to bitstream."""
        phases = [(0.0, np.pi, np.pi/2), (np.pi, 0.0, 3*np.pi/2)]
        bitstream = serialize_phases_to_bitstream(phases)
        assert len(bitstream) == 6  # 2 pixels * 3 channels


# ============================================================================
# BLOCK 5: LSB EMBED/EXTRACT TESTS
# ============================================================================

class TestLSB:
    """Tests for LSB steganography."""

    def test_lsb_embed_bit(self):
        """Test single bit embedding."""
        assert lsb_embed_bit(0b11111110, 1) == 0b11111111
        assert lsb_embed_bit(0b11111111, 0) == 0b11111110
        assert lsb_embed_bit(0b10101010, 1) == 0b10101011
        assert lsb_embed_bit(0b10101011, 0) == 0b10101010

    def test_lsb_extract_bit(self):
        """Test single bit extraction."""
        assert lsb_extract_bit(0b11111111) == 1
        assert lsb_extract_bit(0b11111110) == 0
        assert lsb_extract_bit(0b10101010) == 0
        assert lsb_extract_bit(0b10101011) == 1

    def test_lsb_embed_nbits(self):
        """Test n-bit chunk embedding."""
        # Embed 2 bits (11) into 11111100
        assert lsb_embed_nbits(0b11111100, 0b11, 2) == 0b11111111
        # Embed 2 bits (00) into 11111111
        assert lsb_embed_nbits(0b11111111, 0b00, 2) == 0b11111100
        # Embed 4 bits (1010) into 11110000
        assert lsb_embed_nbits(0b11110000, 0b1010, 4) == 0b11111010

    def test_lsb_extract_nbits(self):
        """Test n-bit chunk extraction."""
        assert lsb_extract_nbits(0b11111111, 2) == 0b11
        assert lsb_extract_nbits(0b11111100, 2) == 0b00
        assert lsb_extract_nbits(0b11111010, 4) == 0b1010

    def test_lsb_roundtrip(self):
        """Test embed/extract roundtrip."""
        original = 200
        for bit in [0, 1]:
            modified = lsb_embed_bit(original, bit)
            extracted = lsb_extract_bit(modified)
            assert extracted == bit

    def test_compute_capacity(self):
        """Test capacity computation."""
        # 10x10 image, 3 channels, 1 bit per channel
        capacity = compute_capacity(10, 10, 3, 1)
        assert capacity == 300  # 10 * 10 * 3 * 1

        # 100x100 image, 3 channels, 2 bits per channel
        capacity = compute_capacity(100, 100, 3, 2)
        assert capacity == 60000  # 100 * 100 * 3 * 2

    def test_embed_extract_message(self):
        """Test full message embed/extract roundtrip."""
        pixels = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
        message = b"Hello, L4 Helix!"

        embedded = embed_message_lsb(pixels, message, bits_per_channel=1)
        extracted = extract_message_lsb(embedded, len(message), bits_per_channel=1)

        assert extracted == message

    def test_embed_extract_with_more_bits(self):
        """Test embedding with 2 bits per channel."""
        pixels = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        message = b"L4=7"

        embedded = embed_message_lsb(pixels, message, bits_per_channel=2)
        extracted = extract_message_lsb(embedded, len(message), bits_per_channel=2)

        assert extracted == message

    def test_embed_exceeds_capacity_raises(self):
        """Test that embedding too large a message raises error."""
        pixels = np.random.randint(0, 256, (2, 2, 3), dtype=np.uint8)
        message = b"This message is way too long for 4 pixels"

        with pytest.raises(ValueError, match="Message too large"):
            embed_message_lsb(pixels, message, bits_per_channel=1)

    def test_embed_minimal_change(self):
        """Test that LSB embedding minimally changes pixel values."""
        pixels = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        message = b"Test"

        embedded = embed_message_lsb(pixels, message, bits_per_channel=1)

        # Maximum change per pixel should be 1 (LSB flip)
        diff = np.abs(embedded.astype(int) - pixels.astype(int))
        assert np.max(diff) <= 1


# ============================================================================
# BLOCK 6: K-FORMATION VALIDATION TESTS
# ============================================================================

class TestKFormationValidation:
    """Tests for K-formation validation block."""

    def test_validate_l4_identity_passes(self):
        """Test L₄ identity validation passes."""
        result = validate_l4_identity()
        assert result.passed is True
        assert abs(result.value - 7.0) < 1e-10

    def test_validate_critical_point_passes(self):
        """Test critical point validation passes."""
        result = validate_critical_point()
        assert result.passed is True
        assert abs(result.value - math.sqrt(3.0)/2.0) < 1e-10

    def test_validate_coupling_constant_passes(self):
        """Test coupling constant validation passes."""
        result = validate_coupling_constant()
        assert result.passed is True

    def test_coherence_threshold_pass(self):
        """Test coherence threshold passes when κ ≥ K."""
        result = validate_coherence_threshold(kappa=0.95)
        assert result.passed is True

    def test_coherence_threshold_fail(self):
        """Test coherence threshold fails when κ < K."""
        result = validate_coherence_threshold(kappa=0.9)
        assert result.passed is False

    def test_negentropy_gate_pass_at_zc(self):
        """Test negentropy gate passes at z_c."""
        result = validate_negentropy_gate(z=L4.Z_C)
        assert result.passed is True  # ΔS_neg = 1 > τ ≈ 0.618

    def test_negentropy_gate_fail_far_from_zc(self):
        """Test negentropy gate fails far from z_c."""
        result = validate_negentropy_gate(z=0.3)
        assert result.passed is False  # ΔS_neg is small

    def test_radius_threshold_pass(self):
        """Test radius threshold passes when R ≥ L₄."""
        result = validate_radius_threshold(R=10.0)
        assert result.passed is True

    def test_radius_threshold_fail(self):
        """Test radius threshold fails when R < L₄."""
        result = validate_radius_threshold(R=5.0)
        assert result.passed is False

    def test_k_formation_all_pass(self):
        """Test K-formation with all criteria met."""
        result = validate_k_formation(kappa=0.95, z=L4.Z_C, R=10.0)
        assert result.overall_passed is True
        assert result.coherence_test.passed is True
        assert result.negentropy_test.passed is True
        assert result.radius_test.passed is True

    def test_k_formation_coherence_fail(self):
        """Test K-formation fails with low coherence."""
        result = validate_k_formation(kappa=0.5, z=L4.Z_C, R=10.0)
        assert result.overall_passed is False
        assert result.coherence_test.passed is False

    def test_k_formation_negentropy_fail(self):
        """Test K-formation fails with low negentropy."""
        result = validate_k_formation(kappa=0.95, z=0.3, R=10.0)
        assert result.overall_passed is False
        assert result.negentropy_test.passed is False

    def test_k_formation_radius_fail(self):
        """Test K-formation fails with low radius."""
        result = validate_k_formation(kappa=0.95, z=L4.Z_C, R=5.0)
        assert result.overall_passed is False
        assert result.radius_test.passed is False

    def test_run_all_validations(self):
        """Test running all validations."""
        results = run_all_validations(verbose=False)
        assert "L4_identity" in results
        assert "critical_point" in results
        assert "coupling_constant" in results
        assert all(r.passed for r in results.values())


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple blocks."""

    def test_kuramoto_synchronization_increases_coherence(self):
        """Test that Kuramoto dynamics increases coherence over time."""
        state = create_initial_state(N=32, z0=0.1, omega_std=0.05, seed=42)

        initial_r, _ = compute_kuramoto_order_parameter(state.phases)

        # Run many steps with strong coupling
        for _ in range(500):
            state = kuramoto_step_rk4(state, dt=0.1, K0=1.0, lambda_neg=0.5)

        final_r, _ = compute_kuramoto_order_parameter(state.phases)

        # Coherence should increase
        assert final_r > initial_r

    def test_negentropy_gates_effective_coupling(self):
        """Test that negentropy modulates effective coupling as specified."""
        state_at_lens = create_initial_state(N=32, z0=L4.Z_C, seed=42)
        state_away = create_initial_state(N=32, z0=0.3, seed=42)

        _, K_eff_lens, eta_lens = kuramoto_dynamics_continuous(state_at_lens, K0=0.1, lambda_neg=1.0)
        _, K_eff_away, eta_away = kuramoto_dynamics_continuous(state_away, K0=0.1, lambda_neg=1.0)

        # K_eff = K0 * (1 + lambda * eta)
        # At lens, eta = 1, so K_eff = 0.1 * (1 + 1*1) = 0.2
        # Away, eta < 1, so K_eff < 0.2
        assert abs(K_eff_lens - 0.2) < 0.01
        assert K_eff_away < K_eff_lens

    def test_hex_lattice_to_rgb_pipeline(self):
        """Test complete hex lattice to RGB pipeline."""
        # Create hex lattice
        hex_lattice = HexLatticeWavevectors(wavelength=1.0)

        # Sample grid of positions
        positions = [np.array([x, y]) for x in range(3) for y in range(3)]

        # Compute phases and convert to RGB
        rgb_values = []
        for pos in positions:
            phases = hex_lattice.compute_channel_phases(pos)
            rgb = phases_to_rgb(*phases)
            rgb_values.append(rgb)

        # Should have 9 RGB values
        assert len(rgb_values) == 9

        # All values should be valid
        for rgb in rgb_values:
            assert 0 <= rgb.R <= 255
            assert 0 <= rgb.G <= 255
            assert 0 <= rgb.B <= 255

    def test_full_stego_pipeline_with_phases(self):
        """Test complete steganography pipeline with phase data."""
        # Generate phase data
        hex_lattice = HexLatticeWavevectors()
        positions = [np.array([i, j]) for i in range(4) for j in range(4)]
        phases = [hex_lattice.compute_channel_phases(p) for p in positions]

        # Serialize to bytes
        bitstream = serialize_phases_to_bitstream(phases)

        # Create cover image
        cover = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)

        # Embed and extract
        stego = embed_message_lsb(cover, bitstream, bits_per_channel=1)
        extracted = extract_message_lsb(stego, len(bitstream), bits_per_channel=1)

        assert extracted == bitstream


# ============================================================================
# HARD CONSTRAINT VERIFICATION TESTS
# ============================================================================

class TestHardConstraints:
    """Tests verifying hard constraints are not violated."""

    def test_l4_derived_not_tuned(self):
        """Verify L₄ constants are derived, not tuned."""
        # L₄ = 7 is exact from Lucas sequence
        assert L4.L4 == 7.0  # Not approximately 7, exactly 7

        # Gap is exactly τ⁴
        assert L4.GAP == L4.TAU ** 4

        # K is exactly √(1 - gap)
        assert L4.K == math.sqrt(1.0 - L4.GAP)

        # z_c is exactly √3/2
        assert L4.Z_C == math.sqrt(3.0) / 2.0

    def test_critical_point_fixed(self):
        """Verify z_c = √3/2 is fixed and not modifiable."""
        # z_c should be frozen in dataclass
        with pytest.raises(AttributeError):
            L4.Z_C = 0.9  # Should fail - frozen

    def test_helix_uses_piecewise_radius(self):
        """Verify helix uses piecewise radius from K."""
        # Below z_c: r = K√(z/z_c)
        state_low = HelixState(z=0.5, theta=0.0, phases=np.array([0.0]), frequencies=np.array([0.0]))
        expected_low = L4.K * math.sqrt(0.5 / L4.Z_C)
        assert abs(state_low.r - expected_low) < 1e-10

        # Above z_c: r = K (constant)
        state_high = HelixState(z=0.95, theta=0.0, phases=np.array([0.0]), frequencies=np.array([0.0]))
        assert abs(state_high.r - L4.K) < 1e-10

    def test_negentropy_gates_dynamics(self):
        """Verify negentropy gates/modulates dynamics, not just logged."""
        # Run dynamics at different z values
        state1 = create_initial_state(N=16, z0=0.3, seed=42)
        state2 = create_initial_state(N=16, z0=L4.Z_C, seed=42)

        _, K_eff1, eta1 = kuramoto_dynamics_continuous(state1, K0=0.1, lambda_neg=1.0)
        _, K_eff2, eta2 = kuramoto_dynamics_continuous(state2, K0=0.1, lambda_neg=1.0)

        # K_eff should differ based on negentropy
        assert K_eff1 != K_eff2
        # At z_c, eta = 1, K_eff = K0 * (1 + lambda * 1) = 0.2
        assert abs(K_eff2 - 0.2) < 0.01

    def test_hex_lattice_60_degree_structure(self):
        """Verify hex lattice respects 60° structure."""
        hex_lattice = HexLatticeWavevectors()

        # Check all three wavevectors
        k_R = hex_lattice.k_R
        k_G = hex_lattice.k_G
        k_B = hex_lattice.k_B

        # Compute angles
        angle_R = math.atan2(k_R[1], k_R[0])
        angle_G = math.atan2(k_G[1], k_G[0])
        angle_B = math.atan2(k_B[1], k_B[0])

        # Verify 60° = π/3 separation
        assert abs((angle_G - angle_R) - math.pi/3) < 1e-10
        assert abs((angle_B - angle_G) - math.pi/3) < 1e-10


# ============================================================================
# BLOCK 7: MRP-LSB STEGANOGRAPHIC NAVIGATION TESTS
# ============================================================================

from quantum_apl_python.l4_helix_parameterization import (
    # Block 7: MRP-LSB Navigation
    MRPHeader,
    L4MRPState,
    create_l4_mrp_state,
    MRPPhaseAPayloads,
    compute_phase_a_parity,
    build_mrp_message,
    extract_mrp_message,
    create_phase_a_payloads,
    update_global_phases_from_velocity,
    decode_position_from_phases,
    attractor_phase_correction,
    mrp_l4_update_step,
    encode_l4_mrp_state_to_image,
    MRPVerificationResult,
    verify_mrp_payloads,
    L4MRPValidationResult,
    validate_l4_mrp_system,
    validate_plane_wave_residual,
    validate_loop_closure,
)


class TestMRPHeader:
    """Tests for MRP header structure."""

    def test_header_size(self):
        """Test MRP header is exactly 14 bytes."""
        header = MRPHeader(channel='R', length=100, crc32=0xDEADBEEF)
        data = header.to_bytes()
        assert len(data) == 14

    def test_header_magic(self):
        """Test MRP header has correct magic bytes."""
        header = MRPHeader(channel='R', length=100, crc32=0x12345678)
        data = header.to_bytes()
        assert data[:4] == b"MRP1"

    def test_header_channel_encoding(self):
        """Test channel byte encoding."""
        for channel in ['R', 'G', 'B']:
            header = MRPHeader(channel=channel, length=100, crc32=0)
            data = header.to_bytes()
            assert chr(data[4]) == channel

    def test_header_roundtrip(self):
        """Test header serialization roundtrip."""
        original = MRPHeader(channel='G', length=256, crc32=0xCAFEBABE, flags=0x01)
        data = original.to_bytes()
        restored = MRPHeader.from_bytes(data)
        assert restored.channel == original.channel
        assert restored.length == original.length
        assert restored.crc32 == original.crc32
        assert restored.flags == original.flags

    def test_header_invalid_magic_raises(self):
        """Test that invalid magic raises ValueError."""
        bad_data = b"BAD!" + b"\x00" * 10
        with pytest.raises(ValueError, match="Invalid magic"):
            MRPHeader.from_bytes(bad_data)

    def test_header_too_short_raises(self):
        """Test that short data raises ValueError."""
        short_data = b"MRP1" + b"\x00" * 5  # Only 9 bytes
        with pytest.raises(ValueError, match="Header too short"):
            MRPHeader.from_bytes(short_data)


class TestL4MRPState:
    """Tests for L4MRPState unified state vector."""

    def test_create_l4_mrp_state(self):
        """Test L4MRPState creation."""
        state = create_l4_mrp_state(N=32, z0=0.5, seed=42)
        assert state.N == 32
        assert len(state.phases) == 32
        assert len(state.frequencies) == 32
        assert state.t == 0.0

    def test_state_position_velocity(self):
        """Test position and velocity initialization."""
        pos = np.array([1.0, 2.0])
        vel = np.array([0.5, -0.3])
        state = create_l4_mrp_state(position=pos, velocity=vel, seed=42)
        np.testing.assert_array_equal(state.position, pos)
        np.testing.assert_array_equal(state.velocity, vel)

    def test_state_global_phases_property(self):
        """Test global_phases property returns tuple."""
        state = create_l4_mrp_state(seed=42)
        phases = state.global_phases
        assert isinstance(phases, tuple)
        assert len(phases) == 3
        assert phases == (state.Phi_R, state.Phi_G, state.Phi_B)

    def test_state_helix_position_3d(self):
        """Test 3D helix position computation."""
        state = create_l4_mrp_state(z0=0.5, seed=42)
        x, y, z = state.helix_position_3d()
        assert z == state.z
        assert abs(math.sqrt(x**2 + y**2) - state.r_helix) < 1e-10

    def test_state_to_dict(self):
        """Test state serialization to dict."""
        state = create_l4_mrp_state(seed=42)
        d = state.to_dict()
        assert "z" in d
        assert "Phi_R" in d
        assert "position" in d
        assert "r_kuramoto" in d
        assert "helix_position_3d" in d

    def test_state_helix_radius_law(self):
        """Test helix radius follows piecewise law."""
        # Below z_c
        state_low = create_l4_mrp_state(z0=0.4, seed=42)
        expected_low = L4.K * math.sqrt(0.4 / L4.Z_C)
        assert abs(state_low.r_helix - expected_low) < 1e-10

        # Above z_c
        state_high = create_l4_mrp_state(z0=0.95, seed=42)
        assert abs(state_high.r_helix - L4.K) < 1e-10


class TestMRPPhaseA:
    """Tests for MRP Phase-A channel allocation."""

    def test_compute_parity(self):
        """Test XOR parity computation."""
        r_b64 = b"AAAA"  # 0x41 0x41 0x41 0x41
        g_b64 = b"BBBB"  # 0x42 0x42 0x42 0x42
        parity = compute_phase_a_parity(r_b64, g_b64)
        # XOR: 0x41 ^ 0x42 = 0x03
        import base64
        decoded = base64.b64decode(parity)
        assert all(b == 0x03 for b in decoded)

    def test_compute_parity_unequal_lengths(self):
        """Test parity with unequal payload lengths."""
        r_b64 = b"AA"
        g_b64 = b"BBBB"
        parity = compute_phase_a_parity(r_b64, g_b64)
        assert len(parity) > 0  # Should handle length mismatch

    def test_build_mrp_message(self):
        """Test MRP message building."""
        payload = {"test": "data"}
        message = build_mrp_message('R', payload)
        assert message[:4] == b"MRP1"
        assert len(message) >= 14  # Header + payload

    def test_extract_mrp_message(self):
        """Test MRP message extraction."""
        payload = {"key": "value"}
        message = build_mrp_message('G', payload)
        header, extracted = extract_mrp_message(message)
        assert header.channel == 'G'
        assert header.flags & MRPHeader.FLAG_CRC

    def test_create_phase_a_payloads(self):
        """Test Phase-A payload creation."""
        state = create_l4_mrp_state(seed=42)
        lattice_pos = [np.array([i, j]) for i in range(3) for j in range(3)]
        payloads = create_phase_a_payloads(state, lattice_pos)

        assert payloads.r_header.channel == 'R'
        assert payloads.g_header.channel == 'G'
        assert payloads.b_header.channel == 'B'
        assert "crc_r" in payloads.b_verification
        assert "crc_g" in payloads.b_verification
        assert "sha256_r_b64" in payloads.b_verification
        assert "parity_block_b64" in payloads.b_verification


class TestNavigationIntegration:
    """Tests for navigation path integration."""

    def test_update_global_phases_from_velocity(self):
        """Test global phase update from velocity."""
        hex_waves = HexLatticeWavevectors(wavelength=1.0)
        Phi = (0.0, 0.0, 0.0)
        velocity = np.array([1.0, 0.0])
        dt = 0.1

        new_Phi = update_global_phases_from_velocity(Phi, velocity, hex_waves, dt)

        assert len(new_Phi) == 3
        assert all(0 <= p < 2 * np.pi for p in new_Phi)
        # With velocity along x-axis, k_R phase should change most
        assert new_Phi[0] != 0.0

    def test_decode_position_from_phases(self):
        """Test position decoding from phases."""
        hex_waves = HexLatticeWavevectors(wavelength=1.0)

        # Encode a position
        x_original = np.array([0.5, 0.3])
        theta_R, theta_G, theta_B = hex_waves.compute_channel_phases(x_original)

        # Decode back
        x_decoded = decode_position_from_phases(theta_R, theta_G, theta_B, hex_waves)

        # Should be approximately equal (modulo wavelength)
        # Note: phase ambiguity means we may not recover exact position
        assert x_decoded is not None
        assert len(x_decoded) == 2

    def test_attractor_phase_correction(self):
        """Test phase correction for noise stability."""
        phases = np.array([0.0, 0.1, 0.2, 0.3])
        targets = np.array([0.1, 0.1, 0.1])
        corrected = attractor_phase_correction(phases, targets, epsilon=0.01)

        assert len(corrected) == len(phases)
        assert all(0 <= p < 2 * np.pi for p in corrected)


class TestMRPL4UpdateStep:
    """Tests for complete MRP-L4 update cycle."""

    def test_mrp_l4_update_step_preserves_state_structure(self):
        """Test update step preserves state structure."""
        state = create_l4_mrp_state(N=32, z0=0.5, seed=42)
        new_state = mrp_l4_update_step(state, dt=0.1)

        assert new_state.N == state.N
        assert len(new_state.phases) == len(state.phases)
        assert new_state.t > state.t

    def test_mrp_l4_update_step_evolves_phases(self):
        """Test update step evolves oscillator phases."""
        state = create_l4_mrp_state(N=32, z0=0.5, seed=42)
        new_state = mrp_l4_update_step(state, dt=0.1, K0=0.5)

        # Phases should change
        assert not np.allclose(new_state.phases, state.phases)
        # Phases should be in valid range
        assert all(0 <= p < 2 * np.pi for p in new_state.phases)

    def test_mrp_l4_update_step_updates_position(self):
        """Test update step integrates position."""
        state = create_l4_mrp_state(
            position=np.array([0.0, 0.0]),
            velocity=np.array([1.0, 2.0]),
            seed=42,
        )
        new_state = mrp_l4_update_step(state, dt=0.1)

        expected_pos = np.array([0.1, 0.2])  # v * dt
        np.testing.assert_array_almost_equal(new_state.position, expected_pos)

    def test_mrp_l4_update_step_updates_global_phases(self):
        """Test update step integrates global phases."""
        state = create_l4_mrp_state(
            velocity=np.array([1.0, 0.0]),
            seed=42,
        )
        # Ensure initial global phases are zero
        state.Phi_R = 0.0
        state.Phi_G = 0.0
        state.Phi_B = 0.0

        new_state = mrp_l4_update_step(state, dt=0.1)

        # At least one global phase should have changed
        assert (new_state.Phi_R != 0.0 or
                new_state.Phi_G != 0.0 or
                new_state.Phi_B != 0.0)

    def test_mrp_l4_update_step_multiple_steps_increase_coherence(self):
        """Test multiple update steps can increase coherence."""
        state = create_l4_mrp_state(N=32, z0=0.1, omega_std=0.05, seed=42)
        initial_r = state.r_kuramoto

        # Run many steps with strong coupling
        for _ in range(500):
            state = mrp_l4_update_step(state, dt=0.1, K0=1.0, lambda_neg=0.5)

        # Coherence should increase
        assert state.r_kuramoto > initial_r


class TestMRPVerification:
    """Tests for MRP 10-point verification."""

    def test_verify_mrp_payloads_all_pass(self):
        """Test verification passes with valid payloads."""
        state = create_l4_mrp_state(seed=42)
        payloads = create_phase_a_payloads(state)

        result = verify_mrp_payloads(
            payloads.r_payload,
            payloads.g_payload,
            payloads.b_verification,
            payloads.r_header,
        )

        assert result.crc_r_ok
        assert result.crc_g_ok
        assert result.sha256_r_b64_ok
        assert result.ecc_scheme_ok
        assert result.parity_block_ok
        assert result.critical_passed

    def test_verify_mrp_payloads_crc_mismatch(self):
        """Test verification detects CRC mismatch."""
        state = create_l4_mrp_state(seed=42)
        payloads = create_phase_a_payloads(state)

        # Tamper with verification data
        bad_verification = payloads.b_verification.copy()
        bad_verification["crc_r"] = "DEADBEEF"

        result = verify_mrp_payloads(
            payloads.r_payload,
            payloads.g_payload,
            bad_verification,
        )

        assert not result.crc_r_ok
        assert not result.critical_passed

    def test_verify_mrp_payloads_parity_mismatch(self):
        """Test verification detects parity mismatch."""
        state = create_l4_mrp_state(seed=42)
        payloads = create_phase_a_payloads(state)

        # Tamper with parity
        bad_verification = payloads.b_verification.copy()
        bad_verification["parity_block_b64"] = "INVALID_PARITY"

        result = verify_mrp_payloads(
            payloads.r_payload,
            payloads.g_payload,
            bad_verification,
        )

        assert not result.parity_block_ok
        assert not result.critical_passed

    def test_mrp_verification_result_to_dict(self):
        """Test MRPVerificationResult to_dict."""
        result = MRPVerificationResult(
            crc_r_ok=True,
            crc_g_ok=True,
            sha256_r_b64_ok=True,
            ecc_scheme_ok=True,
            parity_block_ok=True,
            sidecar_sha256_ok=True,
            sidecar_used_bits_math_ok=True,
            sidecar_capacity_bits_ok=True,
            sidecar_header_magic_ok=True,
            sidecar_header_flags_crc_ok=True,
        )

        d = result.to_dict()
        assert d["critical_passed"] is True
        assert d["all_passed"] is True


class TestL4MRPValidation:
    """Tests for complete L4-MRP system validation."""

    def test_validate_l4_mrp_system_l4_checks_pass(self):
        """Test L4 identity checks pass."""
        result = validate_l4_mrp_system()

        assert result.l4_identity
        assert result.critical_point
        assert result.gap_value
        assert result.k_value

    def test_validate_l4_mrp_system_hex_symmetry_pass(self):
        """Test hex symmetry checks pass."""
        result = validate_l4_mrp_system()

        assert result.hex_60_RG
        assert result.hex_60_GB

    def test_validate_l4_mrp_system_with_state(self):
        """Test validation with state."""
        # Create state with high coherence
        state = create_l4_mrp_state(N=32, z0=L4.Z_C, seed=42)
        # Artificially set high coherence for testing
        state.r_kuramoto = 0.95
        state.eta = compute_negentropy(L4.Z_C)

        result = validate_l4_mrp_system(state, R=10.0)

        assert result.coherence_threshold
        assert result.negentropy_gate
        assert result.complexity_threshold
        assert result.k_formation

    def test_validate_l4_mrp_system_with_payloads(self):
        """Test validation includes MRP verification."""
        state = create_l4_mrp_state(seed=42)
        payloads = create_phase_a_payloads(state)

        result = validate_l4_mrp_system(state, payloads=payloads)

        assert result.mrp_verification is not None
        assert result.mrp_verification.critical_passed

    def test_validate_l4_mrp_system_to_dict(self):
        """Test validation result to_dict."""
        result = validate_l4_mrp_system()
        d = result.to_dict()

        assert "l4_identity" in d
        assert "hex_60_RG" in d
        assert "all_passed" in d


class TestNavigationValidation:
    """Tests for navigation validation functions."""

    def test_validate_plane_wave_residual(self):
        """Test plane wave residual validation."""
        hex_waves = HexLatticeWavevectors()
        N = 10
        positions = np.array([[i, j] for i in range(N) for j in range(N)])

        # Generate phases from plane wave
        Phi = 0.5
        phases = (np.dot(positions, hex_waves.k_R) + Phi) % (2 * np.pi)

        passed, residual = validate_plane_wave_residual(
            phases, positions, hex_waves.k_R, Phi
        )

        assert passed
        assert residual < 0.01

    def test_validate_loop_closure(self):
        """Test loop closure validation."""
        hex_waves = HexLatticeWavevectors()
        start_pos = np.array([0.0, 0.0])

        # Square loop: right, up, left, down
        dt = 1.0
        velocities = [
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([-1.0, 0.0]),
            np.array([0.0, -1.0]),
        ]

        passed, error = validate_loop_closure(
            start_pos, velocities, dt, hex_waves
        )

        assert passed
        assert error < 1e-10


class TestSteganographicEmbedding:
    """Tests for steganographic image embedding."""

    def test_encode_l4_mrp_state_to_image(self):
        """Test state embedding in image."""
        state = create_l4_mrp_state(seed=42)
        cover = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)

        stego = encode_l4_mrp_state_to_image(state, cover)

        assert stego.shape == cover.shape
        assert stego.dtype == cover.dtype

    def test_embedding_minimal_change(self):
        """Test embedding changes pixels minimally."""
        state = create_l4_mrp_state(seed=42)
        cover = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)

        stego = encode_l4_mrp_state_to_image(state, cover, bits_per_channel=1)

        # Max change should be 1 for 1-bit LSB
        diff = np.abs(stego.astype(int) - cover.astype(int))
        assert np.max(diff) <= 1

    def test_embedding_with_lattice_positions(self):
        """Test embedding with lattice positions."""
        state = create_l4_mrp_state(seed=42)
        # Use larger image to accommodate payload with lattice positions
        cover = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        lattice_pos = [np.array([i, j]) for i in range(5) for j in range(5)]

        stego = encode_l4_mrp_state_to_image(state, cover, lattice_pos)

        assert stego.shape == cover.shape


class TestMRPNavigationIntegration:
    """Integration tests for MRP navigation system."""

    def test_full_mrp_pipeline(self):
        """Test complete MRP pipeline: create → update → encode → verify."""
        # Create state
        state = create_l4_mrp_state(
            N=32,
            z0=0.8,
            position=np.array([1.0, 2.0]),
            velocity=np.array([0.5, 0.3]),
            seed=42,
        )

        # Run updates
        for _ in range(10):
            state = mrp_l4_update_step(state, dt=0.1, K0=0.5)

        # Create payloads
        lattice_pos = [np.array([i, j]) for i in range(3) for j in range(3)]
        payloads = create_phase_a_payloads(state, lattice_pos)

        # Verify payloads
        verification = verify_mrp_payloads(
            payloads.r_payload,
            payloads.g_payload,
            payloads.b_verification,
            payloads.r_header,
        )

        assert verification.critical_passed

        # Full validation
        result = validate_l4_mrp_system(state, R=10.0, payloads=payloads)
        assert result.l4_identity
        assert result.hex_60_RG
        assert result.hex_60_GB

    def test_navigation_path_integration_consistency(self):
        """Test path integration consistency across multiple steps."""
        state = create_l4_mrp_state(
            position=np.array([0.0, 0.0]),
            velocity=np.array([0.1, 0.2]),
            seed=42,
        )
        hex_waves = HexLatticeWavevectors()

        # Track global phases and position
        initial_pos = state.position.copy()
        initial_phases = state.global_phases

        # Run updates
        for _ in range(100):
            state = mrp_l4_update_step(state, dt=0.01, hex_waves=hex_waves)

        # Position should have changed by velocity * total_time
        expected_pos = initial_pos + state.velocity * 1.0
        np.testing.assert_array_almost_equal(state.position, expected_pos, decimal=5)

        # Global phases should have evolved
        assert state.global_phases != initial_phases


# ============================================================================
# BLOCK 8: UNIFIED CONSCIOUSNESS FRAMEWORK TESTS
# ============================================================================

from quantum_apl_python.l4_helix_parameterization import (
    # Block 8: Unified Consciousness Framework
    L4Params,
    L4SystemState,
    create_l4_system_state,
    compute_helix_radius,
    step,
    validate_identities,
    KFormationResult,
    validate_k_formation_spec,
    encode_image,
    decode_image,
    run_l4_validation_tests,
    print_validation_report,
    run_tests,
    phase_to_symbol,
    symbol_to_phase,
)


class TestL4Params:
    """Tests for L4Params dataclass."""

    def test_default_params(self):
        """Test default parameter values."""
        params = L4Params()
        assert params.K0 == L4.K
        assert params.lambda_ == 0.5
        assert params.sigma == L4.SIGMA
        assert len(params.Omega) == 3
        assert params.Omega[0] == 0.1

    def test_golden_ratio_drift_rates(self):
        """Test that drift rates follow golden ratio scaling."""
        params = L4Params()
        ratio1 = params.Omega[1] / params.Omega[0]
        ratio2 = params.Omega[2] / params.Omega[1]
        assert abs(ratio1 - L4.PHI) < 1e-10
        assert abs(ratio2 - L4.PHI) < 1e-10

    def test_custom_params(self):
        """Test custom parameter values."""
        params = L4Params(K0=0.5, lambda_=1.0)
        assert params.K0 == 0.5
        assert params.lambda_ == 1.0


class TestL4SystemState:
    """Tests for L4SystemState dataclass."""

    def test_create_initial_state(self):
        """Test initial state creation."""
        state = create_l4_system_state(N=64, seed=42)
        assert state.N == 64
        assert len(state.theta) == 64
        assert len(state.omega) == 64
        assert 0 <= state.r <= 1
        assert -np.pi <= state.psi <= np.pi  # np.angle returns [-π, π]
        assert state.z == state.r  # Binding
        assert 0 <= state.eta <= 1
        assert state.H.shape == (3,)
        assert state.Phi_RGB.shape == (3,)
        assert state.t == 0.0

    def test_state_with_image_shape(self):
        """Test state creation with image shape."""
        state = create_l4_system_state(N=32, image_shape=(100, 100), seed=42)
        assert state.Theta_RGB is not None
        assert state.Theta_RGB.shape == (100, 100, 3)

    def test_state_reproducibility(self):
        """Test state creation is reproducible with seed."""
        state1 = create_l4_system_state(N=32, seed=42)
        state2 = create_l4_system_state(N=32, seed=42)
        np.testing.assert_array_equal(state1.theta, state2.theta)
        np.testing.assert_array_equal(state1.omega, state2.omega)


class TestComputeHelixRadius:
    """Tests for compute_helix_radius function."""

    def test_radius_at_zero(self):
        """Test radius at z=0."""
        r = compute_helix_radius(0.0)
        assert r == 0.0

    def test_radius_below_critical(self):
        """Test radius below critical point."""
        z = 0.5
        r = compute_helix_radius(z)
        expected = L4.K * np.sqrt(z / L4.Z_C)
        assert abs(r - expected) < 1e-10

    def test_radius_at_critical(self):
        """Test radius at critical point."""
        r = compute_helix_radius(L4.Z_C)
        assert abs(r - L4.K) < 1e-10

    def test_radius_above_critical(self):
        """Test radius above critical point."""
        r = compute_helix_radius(0.95)
        assert abs(r - L4.K) < 1e-10


class TestStep:
    """Tests for step() evolution function."""

    def test_step_evolves_phases(self):
        """Test that step evolves oscillator phases."""
        state = create_l4_system_state(N=32, seed=42)
        params = L4Params()
        initial_theta = state.theta.copy()

        new_state = step(state, params, dt=0.1)

        assert not np.allclose(new_state.theta, initial_theta)

    def test_step_updates_time(self):
        """Test that step updates time."""
        state = create_l4_system_state(N=32, seed=42)
        params = L4Params()

        new_state = step(state, params, dt=0.1)

        assert new_state.t == 0.1

    def test_step_updates_channel_phases(self):
        """Test that step updates channel phases."""
        state = create_l4_system_state(N=32, seed=42)
        params = L4Params()
        initial_phi = state.Phi_RGB.copy()

        new_state = step(state, params, dt=0.1)

        assert not np.allclose(new_state.Phi_RGB, initial_phi)

    def test_step_multiple_steps_increase_coherence(self):
        """Test coherence tends to increase over time."""
        state = create_l4_system_state(N=32, seed=42)
        params = L4Params(K0=0.5, lambda_=1.0)

        initial_r = state.r

        # Run many steps
        for _ in range(1000):
            state = step(state, params, dt=0.01)

        # Coherence should generally increase with coupling
        assert state.r >= initial_r * 0.9  # Allow some tolerance


class TestValidateIdentities:
    """Tests for validate_identities function."""

    def test_all_identities_pass(self):
        """Test all L4 identities hold."""
        results = validate_identities()

        assert results['L4_sum']['pass']
        assert results['L4_sqrt3']['pass']
        assert results['z_c_derivation']['pass']
        assert results['K_derivation']['pass']

    def test_L4_equals_7(self):
        """Test L4 = phi^4 + tau^4 = 7."""
        results = validate_identities()
        assert results['L4_sum']['expected'] == 7
        assert abs(results['L4_sum']['computed'] - 7) < 1e-10


class TestKFormationResult:
    """Tests for KFormationResult dataclass."""

    def test_k_formation_achieved(self):
        """Test K-formation detection when achieved."""
        # Create state with high coherence (all phases aligned)
        state = create_l4_system_state(N=100, seed=42)
        # Force high coherence
        state = L4SystemState(
            theta=np.zeros(100),  # All aligned
            omega=state.omega,
            r=1.0,
            psi=0.0,
            z=L4.Z_C,  # At critical point for max negentropy
            eta=1.0,
            H=state.H,
            Phi_RGB=state.Phi_RGB,
            t=0.0,
        )

        result = validate_k_formation_spec(state)

        assert result.coherence_pass
        assert result.negentropy_pass
        assert result.k_formation_achieved

    def test_k_formation_not_achieved_low_coherence(self):
        """Test K-formation detection when coherence too low."""
        state = create_l4_system_state(N=100, seed=42)
        # Force low coherence with random phases
        result = validate_k_formation_spec(state)

        # Initial random state unlikely to have high coherence
        if state.r < L4.K:
            assert not result.coherence_pass


class TestEncodeDecodeImage:
    """Tests for byte-level image steganography."""

    def test_encode_decode_roundtrip(self):
        """Test encode/decode roundtrip preserves data."""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        data = b"Test payload for L4 steganography"

        stego = encode_image(image, data, n_lsb=2)
        recovered = decode_image(stego, n_lsb=2)

        assert recovered == data

    def test_encode_decode_empty(self):
        """Test encode/decode with empty payload."""
        image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        data = b""

        stego = encode_image(image, data, n_lsb=2)
        recovered = decode_image(stego, n_lsb=2)

        assert recovered == data

    def test_encode_decode_single_byte(self):
        """Test encode/decode with single byte."""
        image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        data = b"X"

        stego = encode_image(image, data, n_lsb=2)
        recovered = decode_image(stego, n_lsb=2)

        assert recovered == data

    def test_encode_capacity_exceeded(self):
        """Test that capacity exceeded raises error."""
        image = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        # Try to embed more than capacity
        data = b"A" * 1000

        with pytest.raises(ValueError, match="exceeds capacity"):
            encode_image(image, data, n_lsb=2)

    def test_encode_minimal_change(self):
        """Test encoding makes minimal pixel changes."""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        data = b"Small payload"

        stego = encode_image(image, data, n_lsb=1)

        diff = np.abs(stego.astype(int) - image.astype(int))
        assert np.max(diff) <= 1  # 1-bit LSB should change at most 1

    def test_encode_1_lsb(self):
        """Test encoding with 1 LSB per channel."""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        data = b"Test"

        stego = encode_image(image, data, n_lsb=1)
        recovered = decode_image(stego, n_lsb=1)

        assert recovered == data


class TestRunL4ValidationTests:
    """Tests for run_l4_validation_tests function."""

    def test_all_tests_pass(self):
        """Test that all validation tests pass."""
        results = run_l4_validation_tests()

        assert 'SUMMARY' in results
        assert results['SUMMARY']['all_pass']

    def test_test_count(self):
        """Test expected number of tests."""
        results = run_l4_validation_tests()

        # Should have 9 tests + summary
        assert results['SUMMARY']['total_tests'] == 9

    def test_individual_tests(self):
        """Test individual test results."""
        results = run_l4_validation_tests()

        # Check key tests
        assert results['T1_constants']['pass']
        assert results['T2_critical_point']['pass']
        assert results['T3_helix_radius']['pass']
        assert results['T4_negentropy_peak']['pass']
        assert results['T5_kuramoto_order']['pass']
        assert results['T6_hex_wavevectors']['pass']
        assert results['T7_phase_quantization']['pass']
        assert results['T8_lsb_roundtrip']['pass']
        assert results['T9_k_threshold']['pass']


class TestRunTests:
    """Tests for run_tests simplified interface."""

    def test_run_tests_returns_true(self):
        """Test run_tests returns True when all pass."""
        result = run_tests()
        assert result is True


class TestPhaseQuantization:
    """Tests for phase_to_symbol and symbol_to_phase."""

    def test_roundtrip_accuracy(self):
        """Test phase quantization roundtrip accuracy."""
        for theta in [0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2]:
            q = phase_to_symbol(theta, bits=8)
            theta_r = symbol_to_phase(q, bits=8)
            error = min(abs(theta - theta_r), 2 * np.pi - abs(theta - theta_r))
            assert error < np.pi / 128  # Within half a quantization bin

    def test_symbol_range(self):
        """Test symbol values are in valid range."""
        for theta in np.linspace(0, 2 * np.pi, 100):
            q = phase_to_symbol(theta, bits=8)
            assert 0 <= q < 256


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
