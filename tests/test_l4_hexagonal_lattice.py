#!/usr/bin/env python3
"""
Tests for L₄ Hexagonal Lattice Module
======================================

Comprehensive test suite for:
- HexagonalLattice class with 6-neighbor connectivity
- Extended Kuramoto dynamics with frustration and parametric pump
- Stochastic Resonance
- Fisher Information
- Topological Charge (winding number)
- Berry Phase geometric memory
- Unified Entropic Stabilization system

@version 1.0.0
"""

import math
import numpy as np
import pytest

from quantum_apl_python.l4_hexagonal_lattice import (
    # Hexagonal Lattice
    HexagonalLattice,
    HexLatticeNode,
    HEX_COORDINATION_NUMBER,
    ALPHA_CRITICAL,
    # Extended Kuramoto
    ExtendedKuramotoState,
    extended_kuramoto_dynamics,
    extended_kuramoto_step,
    # Stochastic Resonance
    StochasticResonanceResult,
    compute_stochastic_resonance,
    tune_noise_for_resonance,
    # Fisher Information
    compute_fisher_information,
    compute_spatial_fisher_information,
    # Topological Charge
    compute_topological_charge,
    compute_vortex_density,
    TopologicalState,
    analyze_topological_state,
    # Berry Phase
    BerryPhaseResult,
    compute_berry_phase,
    GeometricMemory,
    # Entropic Stabilization
    EntropicStabilizationState,
    compute_negentropy_driver,
    compute_stabilization_coupling,
    hybrid_dynamics_step,
    compute_rgb_output,
    # Validation
    L4HexLatticeValidation,
    validate_hex_lattice_system,
)

from quantum_apl_python.l4_helix_parameterization import L4


# ============================================================================
# HEXAGONAL LATTICE TESTS
# ============================================================================

class TestHexagonalLattice:
    """Tests for HexagonalLattice class."""

    def test_lattice_creation_default(self):
        """Test default lattice creation."""
        lattice = HexagonalLattice(rows=7, cols=7, seed=42)

        assert lattice.N == 49
        assert lattice.rows == 7
        assert lattice.cols == 7
        assert len(lattice.nodes) == 49

    def test_lattice_minimum_complexity_threshold(self):
        """Test that lattice enforces L₄=7 minimum nodes."""
        # Should fail: N=4 < L₄=7
        with pytest.raises(ValueError, match="Minimum 7 nodes"):
            HexagonalLattice(rows=2, cols=2)

        # Should succeed: N=9 >= L₄=7
        lattice = HexagonalLattice(rows=3, cols=3)
        assert lattice.N == 9 >= L4.L4

    def test_hex_connectivity(self):
        """Test 6-neighbor connectivity for interior nodes."""
        lattice = HexagonalLattice(rows=5, cols=5, seed=42)

        # Interior node should have 6 neighbors
        interior_node = lattice.nodes[12]  # Center of 5x5 grid
        assert len(interior_node.neighbors) == 6

        # Average connectivity should be between 4-6 (boundary effects)
        avg_neighbors = np.mean([len(n.neighbors) for n in lattice.nodes])
        assert 4.0 <= avg_neighbors <= 6.0

    def test_node_positions_hex_pattern(self):
        """Test that nodes are positioned in hex pattern."""
        lattice = HexagonalLattice(rows=3, cols=3, spacing=1.0, seed=42)

        # Second row should be offset by √3/2 in y
        y_row0 = lattice.nodes[0].position[1]
        y_row1 = lattice.nodes[3].position[1]

        expected_delta = math.sqrt(3.0) / 2.0
        assert abs(y_row1 - y_row0 - expected_delta) < 1e-10

    def test_phases_property(self):
        """Test phases getter and setter."""
        lattice = HexagonalLattice(rows=3, cols=3, seed=42)

        # Get phases
        phases = lattice.phases
        assert len(phases) == 9
        assert all(0 <= p < 2 * math.pi for p in phases)

        # Set phases
        new_phases = np.zeros(9)
        lattice.phases = new_phases
        assert np.allclose(lattice.phases, 0)

    def test_order_parameter(self):
        """Test Kuramoto order parameter computation."""
        lattice = HexagonalLattice(rows=3, cols=3, seed=42)

        r, psi = lattice.get_order_parameter()

        assert 0.0 <= r <= 1.0
        assert -math.pi <= psi <= math.pi

    def test_local_order_parameter(self):
        """Test local order parameter for neighborhoods."""
        lattice = HexagonalLattice(rows=5, cols=5, seed=42)

        # Interior node
        r_local, psi_local = lattice.get_local_order_parameter(12)

        assert 0.0 <= r_local <= 1.0

    def test_adjacency_matrix(self):
        """Test adjacency matrix construction."""
        lattice = HexagonalLattice(rows=3, cols=3, seed=42)

        adj = lattice.adjacency
        assert adj.shape == (9, 9)

        # Symmetric
        assert np.allclose(adj, adj.T)

        # Row sums match neighbor counts
        for i, node in enumerate(lattice.nodes):
            assert int(adj[i].sum()) == len(node.neighbors)


# ============================================================================
# EXTENDED KURAMOTO TESTS
# ============================================================================

class TestExtendedKuramoto:
    """Tests for extended Kuramoto dynamics."""

    def test_state_creation(self):
        """Test ExtendedKuramotoState creation."""
        phases = np.random.uniform(0, 2*np.pi, 10)
        freqs = np.random.randn(10) * 0.1

        state = ExtendedKuramotoState(
            phases=phases,
            frequencies=freqs,
            frustration_alpha=0.1,
            pump_strength=0.05,
            noise_amplitude=0.01,
        )

        assert len(state.phases) == 10
        assert state.frustration_alpha == 0.1
        assert state.pump_strength == 0.05
        assert state.noise_amplitude == 0.01

    def test_dynamics_no_frustration(self):
        """Test dynamics without frustration (standard Kuramoto)."""
        lattice = HexagonalLattice(rows=3, cols=3, seed=42)

        state = ExtendedKuramotoState(
            phases=lattice.phases.copy(),
            frequencies=lattice.frequencies.copy(),
            frustration_alpha=0.0,
            pump_strength=0.0,
            noise_amplitude=0.0,
        )

        dtheta = extended_kuramoto_dynamics(state, lattice, K0=0.5)

        assert dtheta.shape == (9,)
        # Dynamics should be finite
        assert np.all(np.isfinite(dtheta))

    def test_dynamics_with_frustration(self):
        """Test dynamics with frustration (hex symmetry inducer)."""
        lattice = HexagonalLattice(rows=3, cols=3, seed=42)

        state = ExtendedKuramotoState(
            phases=lattice.phases.copy(),
            frequencies=lattice.frequencies.copy(),
            frustration_alpha=0.1,
            pump_strength=0.0,
            noise_amplitude=0.0,
        )

        dtheta = extended_kuramoto_dynamics(state, lattice, K0=0.5)

        assert np.all(np.isfinite(dtheta))

    def test_dynamics_with_pump(self):
        """Test dynamics with parametric pump."""
        lattice = HexagonalLattice(rows=3, cols=3, seed=42)

        state = ExtendedKuramotoState(
            phases=lattice.phases.copy(),
            frequencies=lattice.frequencies.copy(),
            frustration_alpha=0.0,
            pump_strength=0.5,  # Strong pump
            noise_amplitude=0.0,
        )

        dtheta = extended_kuramoto_dynamics(state, lattice, K0=0.5)

        assert np.all(np.isfinite(dtheta))

    def test_step_euler(self):
        """Test Euler integration step."""
        lattice = HexagonalLattice(rows=3, cols=3, seed=42)

        state = ExtendedKuramotoState(
            phases=lattice.phases.copy(),
            frequencies=lattice.frequencies.copy(),
        )

        new_state = extended_kuramoto_step(
            state, lattice, dt=0.01, integrator="euler"
        )

        assert new_state.t == state.t + 0.01
        assert all(0 <= p < 2*np.pi for p in new_state.phases)

    def test_step_rk4(self):
        """Test RK4 integration step."""
        lattice = HexagonalLattice(rows=3, cols=3, seed=42)

        state = ExtendedKuramotoState(
            phases=lattice.phases.copy(),
            frequencies=lattice.frequencies.copy(),
        )

        new_state = extended_kuramoto_step(
            state, lattice, dt=0.01, integrator="rk4"
        )

        assert new_state.t == state.t + 0.01

    def test_synchronization_tendency(self):
        """Test that coupling drives synchronization."""
        lattice = HexagonalLattice(rows=5, cols=5, seed=42)

        state = ExtendedKuramotoState(
            phases=lattice.phases.copy(),
            frequencies=np.zeros(25),  # Zero frequencies
        )

        r0, _ = lattice.get_order_parameter()

        # Run many steps with strong coupling
        for _ in range(500):
            state = extended_kuramoto_step(
                state, lattice, dt=0.1, K0=2.0, lambda_neg=1.0
            )

        r1, _ = lattice.get_order_parameter()

        # Coherence should increase
        assert r1 > r0


# ============================================================================
# STOCHASTIC RESONANCE TESTS
# ============================================================================

class TestStochasticResonance:
    """Tests for stochastic resonance."""

    def test_sr_optimal_noise(self):
        """Test optimal noise computation."""
        sr = compute_stochastic_resonance(L4.GAP / 2)

        assert sr.optimal_noise == pytest.approx(L4.GAP / 2, rel=1e-6)
        assert sr.barrier_height == pytest.approx(L4.GAP, rel=1e-6)
        assert sr.is_resonant is True

    def test_sr_non_resonant(self):
        """Test non-resonant regime."""
        # Far from optimal noise
        sr = compute_stochastic_resonance(L4.GAP * 10)

        assert sr.is_resonant is False

    def test_sr_snr_formula(self):
        """Test SNR formula correctness."""
        D = 0.1
        signal = 0.01
        barrier = L4.GAP

        sr = compute_stochastic_resonance(D, signal, barrier)

        expected_snr = (signal / D) * math.exp(-barrier / D)
        assert sr.snr == pytest.approx(expected_snr, rel=1e-6)

    def test_noise_tuning(self):
        """Test adaptive noise tuning."""
        # If coherence too low, noise should decrease
        current_D = 0.1
        new_D = tune_noise_for_resonance(
            current_D,
            target_coherence=L4.Z_C,
            current_coherence=0.3,  # Low coherence
            learning_rate=0.1,
        )

        # Noise should decrease to allow more synchronization
        assert new_D < current_D

    def test_noise_tuning_high_coherence(self):
        """Test noise increases when coherence too high."""
        current_D = 0.01
        new_D = tune_noise_for_resonance(
            current_D,
            target_coherence=L4.Z_C,
            current_coherence=0.95,  # High coherence
            learning_rate=0.1,
        )

        # Noise should increase to prevent lock
        assert new_D > current_D


# ============================================================================
# FISHER INFORMATION TESTS
# ============================================================================

class TestFisherInformation:
    """Tests for Fisher Information."""

    def test_fisher_basic(self):
        """Test basic Fisher Information computation."""
        phases = np.random.uniform(0, 2*np.pi, 20)
        positions = np.random.randn(20, 2)

        fi = compute_fisher_information(phases, positions)

        assert fi >= 0  # Fisher Info is non-negative

    def test_fisher_high_coherence(self):
        """Test Fisher Info is high for coherent phases."""
        # Highly synchronized phases
        phases = np.ones(20) * 0.5  # All same phase
        positions = np.random.randn(20, 2)

        fi = compute_fisher_information(phases, positions)

        # Should be large for synchronized state
        assert fi > 0

    def test_spatial_fisher_info(self):
        """Test spatial Fisher Information."""
        lattice = HexagonalLattice(rows=5, cols=5, seed=42)

        fi = compute_spatial_fisher_information(lattice)

        assert "I_R" in fi
        assert "I_G" in fi
        assert "I_B" in fi
        assert "I_total" in fi
        assert "precision_bound" in fi

        assert fi["I_total"] >= 0


# ============================================================================
# TOPOLOGICAL CHARGE TESTS
# ============================================================================

class TestTopologicalCharge:
    """Tests for topological charge (winding number)."""

    def test_zero_winding_uniform(self):
        """Test zero winding for uniform phases."""
        phases = np.ones(10) * 0.5

        charge = compute_topological_charge(phases, list(range(10)))

        assert charge == 0

    def test_unit_winding(self):
        """Test unit winding number for 2π cycle."""
        # Phases that go through one full cycle
        N = 10
        phases = np.linspace(0, 2*np.pi * 0.99, N)

        charge = compute_topological_charge(phases, list(range(N)))

        # Should be close to 1 (one full winding)
        assert abs(charge) == 1

    def test_vortex_density(self):
        """Test vortex density computation."""
        lattice = HexagonalLattice(rows=5, cols=5, seed=42)

        density, vortices = compute_vortex_density(lattice)

        assert 0.0 <= density <= 1.0
        assert isinstance(vortices, list)

    def test_topological_state_analysis(self):
        """Test full topological state analysis."""
        lattice = HexagonalLattice(rows=5, cols=5, seed=42)

        topo = analyze_topological_state(lattice)

        assert isinstance(topo, TopologicalState)
        assert isinstance(topo.total_charge, int)
        assert topo.vortex_count >= 0
        assert topo.antivortex_count >= 0


# ============================================================================
# BERRY PHASE TESTS
# ============================================================================

class TestBerryPhase:
    """Tests for Berry phase (geometric memory)."""

    def test_berry_phase_computation(self):
        """Test Berry phase computation."""
        # Create a sequence of phase configurations
        path = []
        for i in range(20):
            phase = np.random.uniform(0, 2*np.pi, 10)
            path.append(phase)

        result = compute_berry_phase(path)

        assert isinstance(result, BerryPhaseResult)
        assert np.isfinite(result.geometric_phase)
        assert np.isfinite(result.path_area)

    def test_berry_phase_closed_loop(self):
        """Test Berry phase for closed parameter loop."""
        # Create a closed loop in phase space
        N = 10
        path = []
        for theta in np.linspace(0, 2*np.pi, 20):
            # Phases rotate together
            phases = np.ones(N) * theta
            path.append(phases)

        result = compute_berry_phase(path)

        assert np.isfinite(result.geometric_phase)

    def test_geometric_memory(self):
        """Test GeometricMemory class."""
        memory = GeometricMemory(max_history=100)

        # Record states
        for t in range(50):
            phases = np.random.uniform(0, 2*np.pi, 10)
            memory.record_state(phases, float(t))

        assert len(memory.phase_history) == 50
        assert len(memory.time_history) == 50

        # Compute Berry phase
        berry = memory.compute_total_berry_phase()
        assert np.isfinite(berry)

        # Get path integral memory
        path_mem = memory.get_path_integral_memory()
        assert "x" in path_mem
        assert "y" in path_mem
        assert "confidence" in path_mem

    def test_memory_max_history(self):
        """Test memory respects max_history limit."""
        memory = GeometricMemory(max_history=10)

        for t in range(50):
            memory.record_state(np.random.uniform(0, 2*np.pi, 5), float(t))

        assert len(memory.phase_history) == 10


# ============================================================================
# ENTROPIC STABILIZATION TESTS
# ============================================================================

class TestEntropicStabilization:
    """Tests for unified entropic stabilization system."""

    def test_negentropy_driver(self):
        """Test negentropy driver computation."""
        phases = np.random.uniform(0, 2*np.pi, 20)

        eta = compute_negentropy_driver(phases)

        assert np.isfinite(eta)
        assert eta >= 0

    def test_stabilization_coupling(self):
        """Test stabilization coupling computation."""
        # At z_c, negentropy is maximized
        K_eff = compute_stabilization_coupling(L4.Z_C, K0=L4.K, lambda_neg=0.5)

        # Should be larger than base K due to negentropy boost
        assert K_eff >= L4.K

    def test_stabilization_coupling_far_from_zc(self):
        """Test coupling far from critical point."""
        K_far = compute_stabilization_coupling(0.1, K0=L4.K, lambda_neg=0.5)
        K_near = compute_stabilization_coupling(L4.Z_C, K0=L4.K, lambda_neg=0.5)

        # Near z_c should have higher coupling
        assert K_near > K_far

    def test_state_creation(self):
        """Test EntropicStabilizationState creation."""
        phases = np.random.uniform(0, 2*np.pi, 20)

        state = EntropicStabilizationState(
            phases=phases,
            coherence=0.5,
            position=np.array([1.0, 2.0]),
            negentropy=0.7,
            effective_coupling=L4.K,
            topological_charge=0,
            berry_phase=0.0,
        )

        assert state.coherence == 0.5
        assert len(state.position) == 2

    def test_hybrid_dynamics_step(self):
        """Test hybrid dynamics step."""
        lattice = HexagonalLattice(rows=5, cols=5, seed=42)
        r0, _ = lattice.get_order_parameter()

        state = EntropicStabilizationState(
            phases=lattice.phases.copy(),
            coherence=r0,
            position=np.array([0.0, 0.0]),
            negentropy=0.5,
            effective_coupling=L4.K,
            topological_charge=0,
            berry_phase=0.0,
        )

        new_state = hybrid_dynamics_step(
            state, lattice, dt=0.1,
            K0=0.5, lambda_neg=1.0,
            frustration=0.1,
            pump_strength=0.05,
            velocity=np.array([0.1, 0.05]),
        )

        assert new_state.t == 0.1
        assert np.allclose(new_state.position, [0.01, 0.005])

    def test_rgb_output(self):
        """Test RGB output computation."""
        lattice = HexagonalLattice(rows=5, cols=5, seed=42)

        state = EntropicStabilizationState(
            phases=lattice.phases.copy(),
            coherence=0.5,
            position=np.array([1.0, 2.0]),
            negentropy=0.5,
            effective_coupling=L4.K,
            topological_charge=0,
            berry_phase=0.0,
        )

        rgb = compute_rgb_output(state, lattice)

        assert rgb.shape == (25, 3)
        assert rgb.dtype == np.uint8
        assert np.all(rgb >= 0)
        assert np.all(rgb <= 255)


# ============================================================================
# VALIDATION TESTS
# ============================================================================

class TestValidation:
    """Tests for validation functions."""

    def test_validate_hex_lattice_basic(self):
        """Test basic hex lattice validation."""
        lattice = HexagonalLattice(rows=7, cols=7, seed=42)

        result = validate_hex_lattice_system(lattice, verbose=False)

        assert isinstance(result, L4HexLatticeValidation)
        assert result.min_nodes_ok is True
        assert result.hex_connectivity_ok is True
        assert result.coupling_in_range is True
        assert result.hex_60_symmetry_ok is True

    def test_validate_with_state(self):
        """Test validation with state."""
        lattice = HexagonalLattice(rows=5, cols=5, seed=42)
        r, _ = lattice.get_order_parameter()

        state = EntropicStabilizationState(
            phases=lattice.phases.copy(),
            coherence=r,
            position=np.array([0.0, 0.0]),
            negentropy=0.5,
            effective_coupling=L4.K,
            topological_charge=0,
            berry_phase=0.0,
        )

        result = validate_hex_lattice_system(lattice, state, verbose=False)

        assert isinstance(result.coherence_at_lens, bool)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for full system."""

    def test_full_simulation_cycle(self):
        """Test complete simulation cycle."""
        # Create lattice
        lattice = HexagonalLattice(rows=7, cols=7, seed=42)

        # Create extended Kuramoto state
        ext_state = ExtendedKuramotoState(
            phases=lattice.phases.copy(),
            frequencies=lattice.frequencies.copy(),
            frustration_alpha=0.1,
            pump_strength=0.05,
            noise_amplitude=L4.GAP / 2,
        )

        # Run 100 steps
        for _ in range(100):
            ext_state = extended_kuramoto_step(
                ext_state, lattice, dt=0.1, K0=0.5, lambda_neg=1.0
            )

        # Analyze topological state
        topo = analyze_topological_state(lattice)

        # Compute Fisher Information
        fi = compute_spatial_fisher_information(lattice)

        # Validate
        result = validate_hex_lattice_system(lattice, verbose=False)

        # Assertions
        assert result.min_nodes_ok
        assert result.hex_connectivity_ok
        assert fi["I_total"] >= 0

    def test_navigation_with_memory(self):
        """Test navigation with geometric memory."""
        lattice = HexagonalLattice(rows=5, cols=5, seed=42)
        memory = GeometricMemory()

        state = EntropicStabilizationState(
            phases=lattice.phases.copy(),
            coherence=0.5,
            position=np.array([0.0, 0.0]),
            negentropy=0.5,
            effective_coupling=L4.K,
            topological_charge=0,
            berry_phase=0.0,
        )

        # Navigate in a loop
        velocities = [
            np.array([0.1, 0.0]),  # Right
            np.array([0.0, 0.1]),  # Up
            np.array([-0.1, 0.0]), # Left
            np.array([0.0, -0.1]), # Down
        ] * 25

        for v in velocities:
            state = hybrid_dynamics_step(
                state, lattice, dt=0.1,
                velocity=v,
            )
            memory.record_state(state.phases, state.t)

        # Should return near origin after loop
        assert np.linalg.norm(state.position) < 0.5

        # Berry phase should accumulate
        berry = memory.compute_total_berry_phase()
        assert np.isfinite(berry)

    def test_stochastic_resonance_optimization(self):
        """Test SR-optimized dynamics."""
        lattice = HexagonalLattice(rows=5, cols=5, seed=42)

        # Start at optimal SR noise
        D = L4.GAP / 2
        state = ExtendedKuramotoState(
            phases=lattice.phases.copy(),
            frequencies=lattice.frequencies.copy(),
            noise_amplitude=D,
        )

        sr_before = compute_stochastic_resonance(D)
        assert sr_before.is_resonant

        # Run dynamics
        for _ in range(100):
            state = extended_kuramoto_step(
                state, lattice, dt=0.1, K0=0.5
            )

        # System should remain functional
        r, _ = lattice.get_order_parameter()
        assert 0 < r < 1  # Not collapsed


# ============================================================================
# CONSTANTS TESTS
# ============================================================================

class TestConstants:
    """Tests for module constants."""

    def test_coordination_number(self):
        """Test hex coordination number."""
        assert HEX_COORDINATION_NUMBER == 6

    def test_critical_frustration(self):
        """Test Hopf-Turing bifurcation threshold."""
        assert ALPHA_CRITICAL == pytest.approx(math.pi / 2, rel=1e-10)

    def test_l4_constants_consistency(self):
        """Test L4 constants are consistent."""
        # L₄ = φ⁴ + φ⁻⁴ = 7
        assert L4.L4 == 7.0
        assert L4.PHI ** 4 + L4.TAU ** 4 == pytest.approx(7.0, rel=1e-10)

        # z_c = √3/2
        assert L4.Z_C == pytest.approx(math.sqrt(3) / 2, rel=1e-10)

        # K = √(1 - gap)
        assert L4.K == pytest.approx(math.sqrt(1 - L4.GAP), rel=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
