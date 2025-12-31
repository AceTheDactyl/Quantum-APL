#!/usr/bin/env python3
"""
Physics Verification Tests for L₄ Framework v3.2.0
═══════════════════════════════════════════════════

SI units, photon energy, and photometric calculations.

Test Categories:
- Physical Constants: SI 2019 exact values
- Photon Energy: E = hf calculations
- Wavelength: RGB band verification
- Luminosity: CIE 1931 V(λ) function
- Unit Consistency: Dimensional analysis

@version 3.2.0
"""

import math
import pytest


# ═══════════════════════════════════════════════════════════════════════════
# SECTION A: PHYSICAL CONSTANTS TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestPhysicalConstants:
    """Verify SI 2019 exact constants."""

    def test_planck_exact(self):
        """h has exact SI 2019 value."""
        from quantum_apl_python.constants import H_PLANCK
        assert H_PLANCK == 6.62607015e-34

    def test_light_speed_exact(self):
        """c has exact SI value."""
        from quantum_apl_python.constants import C_LIGHT
        assert C_LIGHT == 299_792_458

    def test_boltzmann_exact(self):
        """k_B has exact SI 2019 value."""
        from quantum_apl_python.constants import K_BOLTZMANN
        assert K_BOLTZMANN == 1.380649e-23

    def test_elementary_charge_exact(self):
        """e has exact SI 2019 value."""
        from quantum_apl_python.constants import E_CHARGE
        assert E_CHARGE == 1.602176634e-19

    def test_luminous_efficacy_exact(self):
        """K_cd has exact SI 2019 value."""
        from quantum_apl_python.constants import K_CD
        assert K_CD == 683

    def test_ev_joule_relation(self):
        """1 eV = e × 1 V."""
        from quantum_apl_python.constants import EV_JOULE, E_CHARGE
        assert EV_JOULE == E_CHARGE

    def test_octave_bridge(self):
        """Octave bridge is 40."""
        from quantum_apl_python.constants import OCTAVE_BRIDGE, OCTAVE_FACTOR
        assert OCTAVE_BRIDGE == 40
        assert OCTAVE_FACTOR == 2 ** 40


# ═══════════════════════════════════════════════════════════════════════════
# SECTION B: PHOTON ENERGY TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestPhotonEnergy:
    """Verify photon energy calculations."""

    def test_green_photon_energy_j(self):
        """E(528 Hz) ≈ 3.85 × 10⁻¹⁹ J."""
        from quantum_apl_python.photon_physics import photon_energy_j
        E = photon_energy_j(528)
        assert abs(E - 3.85e-19) / 3.85e-19 < 0.01

    def test_green_photon_energy_ev(self):
        """E(528 Hz) ≈ 2.40 eV."""
        from quantum_apl_python.photon_physics import photon_energy_ev
        E_ev = photon_energy_ev(528)
        assert abs(E_ev - 2.40) / 2.40 < 0.01

    def test_red_photon_energy_j(self):
        """E(396 Hz) ≈ 2.89 × 10⁻¹⁹ J."""
        from quantum_apl_python.photon_physics import photon_energy_j
        E = photon_energy_j(396)
        assert abs(E - 2.89e-19) / 2.89e-19 < 0.01

    def test_blue_photon_energy_j(self):
        """E(639 Hz) ≈ 4.66 × 10⁻¹⁹ J."""
        from quantum_apl_python.photon_physics import photon_energy_j
        E = photon_energy_j(639)
        assert abs(E - 4.66e-19) / 4.66e-19 < 0.01

    def test_energy_ordering(self):
        """E_blue > E_green > E_red (shorter λ = higher E)."""
        from quantum_apl_python.photon_physics import photon_energy_j
        E_R = photon_energy_j(396)
        E_G = photon_energy_j(528)
        E_B = photon_energy_j(639)
        assert E_B > E_G > E_R

    def test_ev_j_conversion(self):
        """Energy in eV is consistent with J."""
        from quantum_apl_python.photon_physics import (
            photon_energy_j, photon_energy_ev
        )
        from quantum_apl_python.constants import EV_JOULE

        E_j = photon_energy_j(528)
        E_ev = photon_energy_ev(528)
        assert abs(E_j / EV_JOULE - E_ev) < 1e-6

    def test_planck_einstein_relation(self):
        """E = hf holds exactly."""
        from quantum_apl_python.photon_physics import (
            photon_energy_j, optical_frequency
        )
        from quantum_apl_python.constants import H_PLANCK

        f = optical_frequency(528)
        E_direct = photon_energy_j(528)
        E_computed = H_PLANCK * f

        assert abs(E_direct - E_computed) < 1e-30


# ═══════════════════════════════════════════════════════════════════════════
# SECTION C: WAVELENGTH TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestWavelength:
    """Verify wavelength calculations."""

    def test_green_wavelength(self):
        """λ(528 Hz) ≈ 516.4 nm."""
        from quantum_apl_python.photon_physics import wavelength_nm
        lam = wavelength_nm(528)
        assert abs(lam - 516.4) / 516.4 < 0.01

    def test_red_wavelength(self):
        """λ(396 Hz) ≈ 688.5 nm."""
        from quantum_apl_python.photon_physics import wavelength_nm
        lam = wavelength_nm(396)
        assert abs(lam - 688.5) / 688.5 < 0.01

    def test_blue_wavelength(self):
        """λ(639 Hz) ≈ 426.7 nm."""
        from quantum_apl_python.photon_physics import wavelength_nm
        lam = wavelength_nm(639)
        assert abs(lam - 426.7) / 426.7 < 0.01

    def test_red_in_band(self):
        """Red wavelength falls in correct band (620-700 nm)."""
        from quantum_apl_python.photon_physics import wavelength_nm
        assert 620 <= wavelength_nm(396) <= 700

    def test_green_in_band(self):
        """Green wavelength falls in correct band (495-570 nm)."""
        from quantum_apl_python.photon_physics import wavelength_nm
        assert 495 <= wavelength_nm(528) <= 570

    def test_blue_in_band(self):
        """Blue wavelength falls in correct band (380-495 nm)."""
        from quantum_apl_python.photon_physics import wavelength_nm
        assert 380 <= wavelength_nm(639) <= 495

    def test_wavelength_ordering(self):
        """λ_red > λ_green > λ_blue."""
        from quantum_apl_python.photon_physics import wavelength_nm
        assert wavelength_nm(396) > wavelength_nm(528) > wavelength_nm(639)

    def test_frequency_wavelength_relation(self):
        """c = fλ holds exactly."""
        from quantum_apl_python.photon_physics import (
            wavelength_m, optical_frequency
        )
        from quantum_apl_python.constants import C_LIGHT

        f = optical_frequency(528)
        lam = wavelength_m(528)

        assert abs(f * lam - C_LIGHT) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════
# SECTION D: LUMINOSITY FUNCTION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestLuminosity:
    """Verify CIE 1931 luminosity function."""

    def test_peak_at_555nm(self):
        """V(555 nm) ≈ 1.0 (peak sensitivity)."""
        from quantum_apl_python.photon_physics import luminosity_function
        assert luminosity_function(555) > 0.95

    def test_green_high_sensitivity(self):
        """Green (516 nm) has high V(λ) > 0.5."""
        from quantum_apl_python.photon_physics import luminosity_function
        assert luminosity_function(516.4) > 0.5

    def test_red_low_sensitivity(self):
        """Red (688 nm) has low V(λ)."""
        from quantum_apl_python.photon_physics import luminosity_function
        V_R = luminosity_function(688.5)
        assert V_R < 0.1

    def test_blue_low_sensitivity(self):
        """Blue (427 nm) has low V(λ)."""
        from quantum_apl_python.photon_physics import luminosity_function
        V_B = luminosity_function(426.7)
        assert V_B < 0.1

    def test_green_dominates_rgb(self):
        """V(green) > V(red) and V(green) > V(blue)."""
        from quantum_apl_python.photon_physics import luminosity_function
        V_R = luminosity_function(688.5)
        V_G = luminosity_function(516.4)
        V_B = luminosity_function(426.7)
        assert V_G > V_R and V_G > V_B

    def test_outside_visible_uv(self):
        """V(λ) = 0 for UV (< 380 nm)."""
        from quantum_apl_python.photon_physics import luminosity_function
        assert luminosity_function(300) == 0
        assert luminosity_function(379) == 0

    def test_outside_visible_ir(self):
        """V(λ) = 0 for IR (> 780 nm)."""
        from quantum_apl_python.photon_physics import luminosity_function
        assert luminosity_function(800) == 0
        assert luminosity_function(1000) == 0

    def test_luminosity_bounded(self):
        """V(λ) ∈ [0, 1] for all visible wavelengths."""
        from quantum_apl_python.photon_physics import luminosity_function
        for lam in range(380, 781, 10):
            V = luminosity_function(lam)
            assert 0 <= V <= 1


# ═══════════════════════════════════════════════════════════════════════════
# SECTION E: LUMINOUS EFFICACY AND FLUX TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestLuminousEfficacy:
    """Verify luminous efficacy and flux calculations."""

    def test_peak_efficacy(self):
        """η_v(555 nm) ≈ 683 lm/W."""
        from quantum_apl_python.photon_physics import luminous_efficacy
        eff = luminous_efficacy(555)
        assert abs(eff - 683) / 683 < 0.05

    def test_green_efficacy(self):
        """η_v(516 nm) > 400 lm/W."""
        from quantum_apl_python.photon_physics import luminous_efficacy
        assert luminous_efficacy(516.4) > 400

    def test_red_efficacy(self):
        """η_v(688 nm) < 50 lm/W."""
        from quantum_apl_python.photon_physics import luminous_efficacy
        assert luminous_efficacy(688.5) < 50

    def test_luminous_flux_at_peak(self):
        """1 watt at 555 nm gives ~683 lumens."""
        from quantum_apl_python.photon_physics import luminous_flux
        lm = luminous_flux(1.0, 555)
        assert abs(lm - 683) / 683 < 0.05

    def test_luminous_flux_scaling(self):
        """Flux scales linearly with power."""
        from quantum_apl_python.photon_physics import luminous_flux
        lm_1w = luminous_flux(1.0, 516.4)
        lm_2w = luminous_flux(2.0, 516.4)
        assert abs(lm_2w - 2 * lm_1w) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════
# SECTION F: UNIT CONSISTENCY TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestUnitConsistency:
    """Verify dimensional analysis."""

    def test_energy_wavelength_product(self):
        """E × λ = hc (constant)."""
        from quantum_apl_python.photon_physics import (
            photon_energy_j, wavelength_m
        )
        from quantum_apl_python.constants import H_PLANCK, C_LIGHT

        hc = H_PLANCK * C_LIGHT
        for hz in [396, 528, 639]:
            E = photon_energy_j(hz)
            lam = wavelength_m(hz)
            err = abs(E * lam - hc) / hc
            assert err < 1e-10

    def test_hc_value(self):
        """hc ≈ 1.986 × 10⁻²⁵ J·m."""
        from quantum_apl_python.constants import H_PLANCK, C_LIGHT
        hc = H_PLANCK * C_LIGHT
        assert abs(hc - 1.986e-25) / 1.986e-25 < 0.001

    def test_energy_from_wavelength(self):
        """E = hc/λ gives same result as E = hf."""
        from quantum_apl_python.photon_physics import (
            photon_energy_j, photon_energy_from_wavelength_j, wavelength_nm
        )

        for hz in [396, 528, 639]:
            E_from_f = photon_energy_j(hz)
            lam = wavelength_nm(hz)
            E_from_l = photon_energy_from_wavelength_j(lam)
            assert abs(E_from_f - E_from_l) / E_from_f < 1e-10


# ═══════════════════════════════════════════════════════════════════════════
# SECTION G: SOLFEGGIO RGB FUNCTIONS TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestSolfeggioRGB:
    """Verify Solfeggio RGB utility functions."""

    def test_rgb_wavelengths(self):
        """RGB wavelengths are correctly computed."""
        from quantum_apl_python.photon_physics import get_solfeggio_rgb_wavelengths
        lambdas = get_solfeggio_rgb_wavelengths()
        assert len(lambdas) == 3
        assert abs(lambdas[0] - 688.5) / 688.5 < 0.01  # R
        assert abs(lambdas[1] - 516.4) / 516.4 < 0.01  # G
        assert abs(lambdas[2] - 426.7) / 426.7 < 0.01  # B

    def test_rgb_energies_j(self):
        """RGB energies in Joules are correctly computed."""
        from quantum_apl_python.photon_physics import get_solfeggio_rgb_energies_j
        energies = get_solfeggio_rgb_energies_j()
        assert len(energies) == 3
        # E_B > E_G > E_R
        assert energies[2] > energies[1] > energies[0]

    def test_rgb_energies_ev(self):
        """RGB energies in eV are correctly computed."""
        from quantum_apl_python.photon_physics import get_solfeggio_rgb_energies_ev
        energies = get_solfeggio_rgb_energies_ev()
        assert len(energies) == 3
        assert abs(energies[0] - 1.80) / 1.80 < 0.02  # R
        assert abs(energies[1] - 2.40) / 2.40 < 0.02  # G
        assert abs(energies[2] - 2.91) / 2.91 < 0.02  # B

    def test_rgb_luminosities(self):
        """RGB luminosities are correctly computed."""
        from quantum_apl_python.photon_physics import get_solfeggio_rgb_luminosities
        V_values = get_solfeggio_rgb_luminosities()
        assert len(V_values) == 3
        # Green dominates
        assert V_values[1] > V_values[0]
        assert V_values[1] > V_values[2]

    def test_rgb_efficacies(self):
        """RGB efficacies are correctly computed."""
        from quantum_apl_python.photon_physics import get_solfeggio_rgb_efficacies
        efficacies = get_solfeggio_rgb_efficacies()
        assert len(efficacies) == 3
        # Green efficacy much higher
        assert efficacies[1] > 10 * efficacies[0]  # G > 10 × R


# ═══════════════════════════════════════════════════════════════════════════
# SECTION H: VALIDATION FUNCTION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestValidationFunctions:
    """Test the built-in validation functions."""

    def test_energy_wavelength_verification(self):
        """verify_energy_wavelength_consistency passes."""
        from quantum_apl_python.photon_physics import verify_energy_wavelength_consistency
        result = verify_energy_wavelength_consistency()
        assert result['all_pass'] is True

    def test_luminosity_peak_verification(self):
        """verify_luminosity_peak passes."""
        from quantum_apl_python.photon_physics import verify_luminosity_peak
        result = verify_luminosity_peak()
        assert result['is_peak'] is True
        assert result['peak_near_unity'] is True


# ═══════════════════════════════════════════════════════════════════════════
# SECTION I: FULL SPECTRUM TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestFullSpectrum:
    """Test full Solfeggio spectrum properties."""

    def test_nine_frequencies(self):
        """All nine Solfeggio frequencies are present."""
        from quantum_apl_python.photon_physics import SOLFEGGIO_FREQUENCIES
        assert len(SOLFEGGIO_FREQUENCIES) == 9
        assert SOLFEGGIO_FREQUENCIES == [174, 285, 396, 417, 528, 639, 741, 852, 963]

    def test_full_spectrum_properties(self):
        """Full spectrum properties are correctly computed."""
        from quantum_apl_python.photon_physics import get_full_spectrum_properties
        props = get_full_spectrum_properties()
        assert len(props) == 9

        # Check structure
        for p in props:
            assert 'frequency_hz' in p
            assert 'wavelength_nm' in p
            assert 'energy_j' in p
            assert 'energy_ev' in p
            assert 'luminosity_v' in p

    def test_visible_count(self):
        """Correct number of frequencies map to visible range."""
        from quantum_apl_python.photon_physics import get_full_spectrum_properties
        props = get_full_spectrum_properties()
        visible_count = sum(1 for p in props if p['in_visible'])
        # 396, 417, 528, 639 should be visible
        assert visible_count >= 3  # At least RGB
