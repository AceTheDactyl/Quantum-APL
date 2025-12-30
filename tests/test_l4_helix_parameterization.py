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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
