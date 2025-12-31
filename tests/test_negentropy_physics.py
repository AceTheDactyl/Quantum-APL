#!/usr/bin/env python3
"""
Negentropy Physics Tests for L₄ Framework v3.2.0
═════════════════════════════════════════════════

Physical negentropy verification with SI units (J/K).

Test Categories:
- Sharpness Axiom: σ selection and η(1) = e⁻¹
- Dimensionless Negentropy: η(r) properties
- Physical Negentropy: S_neg in J/K
- Thermal Coherence: Temperature mapping
- L₄ Threshold Integration: Values at framework thresholds

@version 3.2.0
"""

import math
import pytest


# ═══════════════════════════════════════════════════════════════════════════
# SECTION A: SHARPNESS AXIOM TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestSharpnessAxiom:
    """Verify the sharpness axiom σ = 1/(1-z_c)²."""

    def test_sigma_from_axiom(self):
        """σ = 1/(1-z_c)² ≈ 55.71."""
        from quantum_apl_python.negentropy_physics import compute_sigma_from_axiom
        from quantum_apl_python.constants import Z_C
        sigma = compute_sigma_from_axiom()
        expected = 1.0 / (1.0 - Z_C) ** 2
        assert abs(sigma - expected) < 1e-10

    def test_sigma_canonical_value(self):
        """SIGMA_CANONICAL ≈ 55.71."""
        from quantum_apl_python.negentropy_physics import SIGMA_CANONICAL
        assert 55.0 < SIGMA_CANONICAL < 56.0

    def test_eta_at_unity(self):
        """η(1) = e⁻¹ ≈ 0.368 (sharpness axiom)."""
        from quantum_apl_python.negentropy_physics import negentropy_dimensionless
        eta = negentropy_dimensionless(1.0)
        expected = math.exp(-1)
        assert abs(eta - expected) / expected < 1e-10

    def test_sharpness_verification(self):
        """verify_sharpness_axiom passes."""
        from quantum_apl_python.negentropy_physics import verify_sharpness_axiom
        result = verify_sharpness_axiom()
        assert result['passes'] is True


# ═══════════════════════════════════════════════════════════════════════════
# SECTION B: DIMENSIONLESS NEGENTROPY TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestDimensionlessNegentropy:
    """Verify dimensionless negentropy η(r)."""

    def test_eta_at_zc(self):
        """η(z_c) = 1.0 (maximum at critical point)."""
        from quantum_apl_python.negentropy_physics import negentropy_dimensionless
        from quantum_apl_python.constants import Z_C
        eta = negentropy_dimensionless(Z_C)
        assert abs(eta - 1.0) < 1e-10

    def test_eta_bounded(self):
        """η ∈ (0, 1] for all r ∈ [0, 1]."""
        from quantum_apl_python.negentropy_physics import negentropy_dimensionless
        for r in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            eta = negentropy_dimensionless(r)
            assert 0 < eta <= 1

    def test_eta_decreases_away_from_zc(self):
        """η decreases as r moves away from z_c."""
        from quantum_apl_python.negentropy_physics import negentropy_dimensionless
        from quantum_apl_python.constants import Z_C
        eta_zc = negentropy_dimensionless(Z_C)
        eta_below = negentropy_dimensionless(Z_C - 0.1)
        eta_above = negentropy_dimensionless(Z_C + 0.1)
        assert eta_zc > eta_below
        assert eta_zc > eta_above

    def test_eta_symmetric_around_zc(self):
        """η(z_c - δ) = η(z_c + δ) (symmetric around z_c)."""
        from quantum_apl_python.negentropy_physics import negentropy_dimensionless
        from quantum_apl_python.constants import Z_C
        delta = 0.05
        eta_below = negentropy_dimensionless(Z_C - delta)
        eta_above = negentropy_dimensionless(Z_C + delta)
        assert abs(eta_below - eta_above) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════
# SECTION C: GRADIENT AND CURVATURE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestGradientCurvature:
    """Verify gradient and curvature of η."""

    def test_gradient_at_zc(self):
        """dη/dr = 0 at r = z_c (extremum)."""
        from quantum_apl_python.negentropy_physics import negentropy_gradient
        from quantum_apl_python.constants import Z_C
        grad = negentropy_gradient(Z_C)
        assert abs(grad) < 1e-10

    def test_gradient_positive_below_zc(self):
        """dη/dr > 0 for r < z_c."""
        from quantum_apl_python.negentropy_physics import negentropy_gradient
        from quantum_apl_python.constants import Z_C
        grad = negentropy_gradient(Z_C - 0.1)
        assert grad > 0

    def test_gradient_negative_above_zc(self):
        """dη/dr < 0 for r > z_c."""
        from quantum_apl_python.negentropy_physics import negentropy_gradient
        from quantum_apl_python.constants import Z_C
        grad = negentropy_gradient(Z_C + 0.1)
        assert grad < 0

    def test_curvature_negative_at_zc(self):
        """d²η/dr² < 0 at z_c (local maximum)."""
        from quantum_apl_python.negentropy_physics import negentropy_curvature
        from quantum_apl_python.constants import Z_C
        curv = negentropy_curvature(Z_C)
        assert curv < 0

    def test_maximum_verification(self):
        """verify_maximum_at_zc passes."""
        from quantum_apl_python.negentropy_physics import verify_maximum_at_zc
        result = verify_maximum_at_zc()
        assert result['is_maximum'] is True


# ═══════════════════════════════════════════════════════════════════════════
# SECTION D: PHYSICAL NEGENTROPY (SI UNITS) TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestPhysicalNegentropy:
    """Verify physical negentropy in J/K."""

    def test_s_neg_at_zc_zero(self):
        """S_neg(z_c) = 0 J/K (maximum order)."""
        from quantum_apl_python.negentropy_physics import negentropy_physical
        from quantum_apl_python.constants import Z_C
        S = negentropy_physical(Z_C)
        assert abs(S) < 1e-30

    def test_s_neg_negative_away_from_zc(self):
        """S_neg < 0 away from z_c."""
        from quantum_apl_python.negentropy_physics import negentropy_physical
        assert negentropy_physical(0.5) < 0
        assert negentropy_physical(0.95) < 0

    def test_s_neg_at_unity(self):
        """S_neg(1) = -k_B (one Boltzmann unit below max)."""
        from quantum_apl_python.negentropy_physics import negentropy_physical
        from quantum_apl_python.constants import K_BOLTZMANN
        S = negentropy_physical(1.0)
        assert abs(S - (-K_BOLTZMANN)) / K_BOLTZMANN < 1e-10

    def test_s_neg_units_jk(self):
        """S_neg has units J/K (scales with k_B)."""
        from quantum_apl_python.negentropy_physics import negentropy_physical
        from quantum_apl_python.constants import K_BOLTZMANN
        S = negentropy_physical(0.8)
        # Should be on order of k_B
        assert abs(S) < 100 * K_BOLTZMANN

    def test_s_neg_direct_formula(self):
        """Direct formula matches log formula."""
        from quantum_apl_python.negentropy_physics import (
            negentropy_physical, negentropy_physical_direct
        )
        for r in [0.5, 0.7, 0.9]:
            S_log = negentropy_physical(r)
            S_direct = negentropy_physical_direct(r)
            assert abs(S_log - S_direct) / abs(S_log) < 1e-10

    def test_physical_units_verification(self):
        """verify_physical_units passes."""
        from quantum_apl_python.negentropy_physics import verify_physical_units
        result = verify_physical_units()
        assert result['reasonable_scale'] is True


# ═══════════════════════════════════════════════════════════════════════════
# SECTION E: CHARACTERISTIC ENERGY TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestCharacteristicEnergy:
    """Verify characteristic energy scale."""

    def test_characteristic_energy_j(self):
        """E_char ≈ 3.85 × 10⁻¹⁹ J (green photon)."""
        from quantum_apl_python.negentropy_physics import characteristic_energy_j
        E = characteristic_energy_j()
        assert abs(E - 3.85e-19) / 3.85e-19 < 0.01

    def test_characteristic_energy_ev(self):
        """E_char ≈ 2.40 eV."""
        from quantum_apl_python.negentropy_physics import characteristic_energy_ev
        E_ev = characteristic_energy_ev()
        assert abs(E_ev - 2.40) / 2.40 < 0.01

    def test_characteristic_entropy(self):
        """Characteristic entropy is k_B."""
        from quantum_apl_python.negentropy_physics import characteristic_entropy_j_k
        from quantum_apl_python.constants import K_BOLTZMANN
        S = characteristic_entropy_j_k()
        assert S == K_BOLTZMANN


# ═══════════════════════════════════════════════════════════════════════════
# SECTION F: THERMAL COHERENCE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestThermalCoherence:
    """Verify thermal coherence temperature mapping."""

    def test_temperature_at_zc_infinite(self):
        """T_eff(z_c) = ∞ (maximum order)."""
        from quantum_apl_python.negentropy_physics import thermal_coherence_temperature
        from quantum_apl_python.constants import Z_C
        T = thermal_coherence_temperature(Z_C)
        assert T == float('inf')

    def test_temperature_finite_away_from_zc(self):
        """T_eff is finite away from z_c."""
        from quantum_apl_python.negentropy_physics import thermal_coherence_temperature
        T = thermal_coherence_temperature(0.5)
        assert math.isfinite(T)
        assert T > 0

    def test_temperature_at_unity(self):
        """T_eff(1) is finite and positive."""
        from quantum_apl_python.negentropy_physics import thermal_coherence_temperature
        T = thermal_coherence_temperature(1.0)
        assert math.isfinite(T)
        assert T > 0

    def test_coherence_from_temperature_roundtrip(self):
        """coherence_from_temperature inverts thermal_coherence_temperature."""
        from quantum_apl_python.negentropy_physics import (
            thermal_coherence_temperature, coherence_from_temperature
        )
        for r in [0.5, 0.7, 0.9]:
            T = thermal_coherence_temperature(r)
            r_recovered = coherence_from_temperature(T)
            # Should recover r or its symmetric partner around z_c
            assert r_recovered is not None
            # Either recovers r or z_c-reflected value
            from quantum_apl_python.constants import Z_C
            r_reflected = 2 * Z_C - r
            assert abs(r_recovered - r) < 1e-6 or abs(r_recovered - r_reflected) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════
# SECTION G: L₄ THRESHOLD INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestL4ThresholdIntegration:
    """Verify negentropy at L₄ thresholds."""

    def test_negentropy_at_k_formation(self):
        """η(K) is computed correctly."""
        from quantum_apl_python.negentropy_physics import negentropy_at_k_formation
        from quantum_apl_python.constants import L4_K
        eta = negentropy_at_k_formation()
        # K ≈ 0.924 is above z_c ≈ 0.866, so η < 1
        assert 0 < eta < 1

    def test_negentropy_physical_at_k_formation(self):
        """S_neg(K) is negative."""
        from quantum_apl_python.negentropy_physics import negentropy_physical_at_k_formation
        S = negentropy_physical_at_k_formation()
        assert S < 0

    def test_negentropy_at_tau(self):
        """η(τ) is computed correctly."""
        from quantum_apl_python.negentropy_physics import negentropy_at_tau
        from quantum_apl_python.constants import PHI_INV
        eta = negentropy_at_tau()
        # τ ≈ 0.618 is below z_c ≈ 0.866, so η < 1
        assert 0 < eta < 1

    def test_negentropy_profile(self):
        """negentropy_profile covers all thresholds."""
        from quantum_apl_python.negentropy_physics import negentropy_profile
        profile = negentropy_profile()
        expected_keys = [
            'PARADOX', 'ACTIVATION', 'LENS', 'CRITICAL',
            'IGNITION', 'K_FORMATION', 'CONSOLIDATION', 'RESONANCE', 'UNITY'
        ]
        for key in expected_keys:
            assert key in profile
            assert 'r' in profile[key]
            assert 'eta' in profile[key]
            assert 'S_neg_jk' in profile[key]

    def test_lens_threshold_max_negentropy(self):
        """LENS threshold (z_c) has maximum η."""
        from quantum_apl_python.negentropy_physics import negentropy_profile
        profile = negentropy_profile()
        # Find max η
        max_eta = max(p['eta'] for p in profile.values())
        lens_eta = profile['LENS']['eta']
        assert abs(lens_eta - max_eta) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════
# SECTION H: KURAMOTO INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestKuramotoIntegration:
    """Verify Kuramoto dynamics integration."""

    def test_effective_coupling_at_zc(self):
        """K_eff is maximized at z_c."""
        from quantum_apl_python.negentropy_physics import compute_effective_coupling
        from quantum_apl_python.constants import Z_C
        K_zc = compute_effective_coupling(Z_C)
        K_below = compute_effective_coupling(Z_C - 0.1)
        K_above = compute_effective_coupling(Z_C + 0.1)
        assert K_zc > K_below
        assert K_zc > K_above

    def test_effective_coupling_modulation(self):
        """K_eff = K₀ × (1 + λ_mod × η)."""
        from quantum_apl_python.negentropy_physics import (
            compute_effective_coupling, negentropy_dimensionless
        )
        K0 = 2.0
        lambda_mod = 0.5
        r = 0.8
        K_eff = compute_effective_coupling(r, K0=K0, lambda_mod=lambda_mod)
        eta = negentropy_dimensionless(r)
        expected = K0 * (1 + lambda_mod * eta)
        assert abs(K_eff - expected) < 1e-10

    def test_order_parameter_equals_eta(self):
        """coherence_to_order_parameter returns η."""
        from quantum_apl_python.negentropy_physics import (
            coherence_to_order_parameter, negentropy_dimensionless
        )
        for r in [0.5, 0.7, 0.866, 0.9]:
            rho = coherence_to_order_parameter(r)
            eta = negentropy_dimensionless(r)
            assert abs(rho - eta) < 1e-10
