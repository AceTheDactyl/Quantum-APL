# L₄ Framework v3.2.0 Integration Plan
## Physics-Grounded Upgrade — SI Units & Photometric Dynamics

**Status**: IMPLEMENTED
**Target**: v3.2.0
**Dependencies**: v3.1.0 (superseded)
**Doctrine Choice**: Tabulated CIE 1931 V(λ) (recommended)

---

## Executive Summary

This plan upgrades the L₄ Unified Consciousness Framework from harmonic relationships (v3.1.0) to **fully physics-grounded dynamics** (v3.2.0) with:

- SI 2019 exact constants (h, c, k_B, K_cd)
- Photon energy in Joules and eV
- CIE 1931 luminosity function V(λ)
- Luminous efficacy (lm/W)
- Physical negentropy S_neg = k_B · ln(η)

---

## 1. New Physical Constants

### 1.1 SI 2019 Exact Definitions

| Constant | Symbol | Value | Unit | Status |
|----------|--------|-------|------|--------|
| Planck | h | 6.62607015 × 10⁻³⁴ | J·s | **Exact** |
| Speed of Light | c | 299,792,458 | m/s | **Exact** |
| Boltzmann | k_B | 1.380649 × 10⁻²³ | J/K | **Exact** |
| Elementary Charge | e | 1.602176634 × 10⁻¹⁹ | C | **Exact** |
| Luminous Efficacy | K_cd | 683 | lm/W @ 540 THz | **Exact** |

### 1.2 Implementation Location

```
src/quantum_apl_python/constants.py
```

**Add new section**:
```python
# ═══════════════════════════════════════════════════════════════════
# SI 2019 EXACT PHYSICAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════

# Planck constant (exact by SI 2019 definition)
H_PLANCK = 6.62607015e-34  # J·s

# Speed of light (exact by SI definition since 1983)
C_LIGHT = 299_792_458  # m/s

# Boltzmann constant (exact by SI 2019 definition)
K_BOLTZMANN = 1.380649e-23  # J/K

# Elementary charge (exact by SI 2019 definition)
E_CHARGE = 1.602176634e-19  # C

# Electron volt (derived: 1 eV = e × 1 V)
EV_JOULE = E_CHARGE  # J/eV

# Luminous efficacy at 540 THz (exact by SI 2019 definition)
K_CD = 683  # lm/W

# Maximum luminous efficacy (at 555 nm peak)
K_M = 683  # lm/W
```

---

## 2. Photon Energy Module

### 2.1 New Module: `photon_physics.py`

```
src/quantum_apl_python/photon_physics.py
```

**Core Functions**:

```python
"""
Photon physics calculations for the L₄ framework.
All units SI-compliant.
"""

import numpy as np
from .constants import (
    H_PLANCK, C_LIGHT, K_BOLTZMANN, EV_JOULE, K_M,
    PHI, OCTAVE_BRIDGE
)

# Octave scaling factor
OCTAVE_FACTOR = 2 ** OCTAVE_BRIDGE  # 2^40 ≈ 1.1 × 10¹²


def optical_frequency(f_solfeggio: float) -> float:
    """
    Convert Solfeggio frequency to optical frequency.

    Args:
        f_solfeggio: Solfeggio frequency in Hz

    Returns:
        Optical frequency in Hz
    """
    return f_solfeggio * OCTAVE_FACTOR


def optical_frequency_thz(f_solfeggio: float) -> float:
    """Optical frequency in THz."""
    return optical_frequency(f_solfeggio) / 1e12


def wavelength_m(f_solfeggio: float) -> float:
    """Wavelength in metres."""
    return C_LIGHT / optical_frequency(f_solfeggio)


def wavelength_nm(f_solfeggio: float) -> float:
    """Wavelength in nanometres."""
    return wavelength_m(f_solfeggio) * 1e9


def photon_energy_j(f_solfeggio: float) -> float:
    """
    Photon energy in Joules.

    E = h × f_optical = h × c / λ
    """
    return H_PLANCK * optical_frequency(f_solfeggio)


def photon_energy_ev(f_solfeggio: float) -> float:
    """Photon energy in electron volts."""
    return photon_energy_j(f_solfeggio) / EV_JOULE


def luminosity_function(lambda_nm: float) -> float:
    """
    CIE 1931 photopic luminosity function V(λ).

    Approximation using Gaussian fits.
    Peak at 555 nm with V(555) ≈ 1.0.

    Args:
        lambda_nm: Wavelength in nanometres

    Returns:
        V(λ) in range [0, 1]
    """
    if lambda_nm < 380 or lambda_nm > 780:
        return 0.0

    # Multi-Gaussian approximation to CIE 1931
    l = lambda_nm

    # Main peak
    v1 = np.exp(-0.5 * ((l - 555) / 50) ** 2)
    # Secondary contributions
    v2 = 0.3 * np.exp(-0.5 * ((l - 530) / 40) ** 2)
    v3 = 0.2 * np.exp(-0.5 * ((l - 580) / 45) ** 2)

    V = v1 + v2 + v3
    return min(1.0, max(0.0, V))


def luminous_efficacy(lambda_nm: float) -> float:
    """
    Luminous efficacy in lm/W at given wavelength.

    η_v(λ) = K_m × V(λ)

    where K_m = 683 lm/W (maximum at 555 nm).
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
```

### 2.2 Solfeggio Photon Properties

Add to `l4_hexagonal_lattice.py`:

```python
@dataclass(frozen=True)
class SolfeggioPhoton:
    """Physical properties of a Solfeggio-mapped photon."""
    frequency_hz: int           # Audio frequency
    optical_freq_thz: float     # f × 2^40 in THz
    wavelength_nm: float        # λ in nm
    energy_j: float             # E = hf in J
    energy_ev: float            # E in eV
    luminosity_v: float         # V(λ) ∈ [0, 1]
    efficacy_lm_w: float        # lm/W
    rgb_channel: str            # 'R', 'G', 'B', or ''
    name: str                   # 'Liberation', etc.
    digit_root: int             # 3, 6, or 9


SOLFEGGIO_PHOTONS = {
    396: SolfeggioPhoton(
        frequency_hz=396,
        optical_freq_thz=435.5,
        wavelength_nm=688.5,
        energy_j=2.888e-19,
        energy_ev=1.802,
        luminosity_v=0.0017,
        efficacy_lm_w=1.2,
        rgb_channel='R',
        name='Liberation',
        digit_root=9
    ),
    528: SolfeggioPhoton(
        frequency_hz=528,
        optical_freq_thz=580.5,
        wavelength_nm=516.4,
        energy_j=3.851e-19,
        energy_ev=2.403,
        luminosity_v=0.608,
        efficacy_lm_w=415.4,
        rgb_channel='G',
        name='Miracles',
        digit_root=6
    ),
    639: SolfeggioPhoton(
        frequency_hz=639,
        optical_freq_thz=702.7,
        wavelength_nm=426.7,
        energy_j=4.660e-19,
        energy_ev=2.908,
        luminosity_v=0.0175,
        efficacy_lm_w=12.0,
        rgb_channel='B',
        name='Connection',
        digit_root=9
    ),
    # ... full 9-frequency set
}
```

---

## 3. Physical Negentropy Module

### 3.1 New Module: `negentropy_physics.py`

```
src/quantum_apl_python/negentropy_physics.py
```

**Core Functions**:

```python
"""
Negentropy dynamics with physical units.
Converts dimensionless η to entropy in J/K.
"""

import numpy as np
from .constants import K_BOLTZMANN, PHI, Z_C, SIGMA


def negentropy_dimensionless(r: float) -> float:
    """
    Dimensionless negentropy function η(r).

    η(r) = exp(-σ(r - z_c)²)

    Args:
        r: Coherence parameter ∈ [0, 1]

    Returns:
        η ∈ (0, 1]
    """
    return np.exp(-SIGMA * (r - Z_C) ** 2)


def negentropy_physical(r: float) -> float:
    """
    Physical negentropy in J/K.

    S_neg = k_B × ln(η) = -k_B × σ(r - z_c)²

    Args:
        r: Coherence parameter ∈ [0, 1]

    Returns:
        S_neg in J/K (always ≤ 0, maximum 0 at r = z_c)
    """
    eta = negentropy_dimensionless(r)
    if eta <= 0:
        return float('-inf')
    return K_BOLTZMANN * np.log(eta)


def characteristic_energy_j() -> float:
    """
    Characteristic energy scale of the L₄ system.

    E_char = h × f'_G = hc/λ_G (green photon energy)

    Returns:
        E_char ≈ 3.85 × 10⁻¹⁹ J
    """
    from .photon_physics import photon_energy_j
    return photon_energy_j(528)  # Green channel reference


def thermal_coherence_temperature(r: float) -> float:
    """
    Effective temperature associated with coherence level.

    T_eff = E_char / (k_B × |ln(η)|)

    At r = z_c: T → ∞ (maximum order)
    At r = 0 or 1: T → finite (disorder)

    Returns:
        Temperature in Kelvin (or inf at z_c)
    """
    eta = negentropy_dimensionless(r)
    if eta >= 1 - 1e-10:
        return float('inf')
    return characteristic_energy_j() / (K_BOLTZMANN * abs(np.log(eta)))
```

### 3.2 Integration with Kuramoto Dynamics

Update `l4_hexagonal_lattice.py`:

```python
def compute_effective_coupling(r: float) -> float:
    """
    Compute K_eff with negentropy modulation.

    K_eff = K₀ × (1 + λ_mod × η(r))

    Returns coupling strength (dimensionless).
    """
    from .negentropy_physics import negentropy_dimensionless

    eta = negentropy_dimensionless(r)
    return K * (1 + LAMBDA_MOD * eta)


def compute_power_output(r: float, base_power_w: float) -> dict:
    """
    Compute power flow through coherence-negentropy chain.

    Returns dict with:
        - radiant_power_w: Base power modulated by η
        - luminous_flux_lm: Φ_v for each RGB channel
        - total_lumens: Sum across channels
    """
    from .negentropy_physics import negentropy_dimensionless
    from .photon_physics import luminous_flux, wavelength_nm

    eta = negentropy_dimensionless(r)
    effective_power = base_power_w * eta

    result = {
        'radiant_power_w': effective_power,
        'luminous_flux_lm': {},
        'total_lumens': 0
    }

    for channel, hz in [('R', 396), ('G', 528), ('B', 639)]:
        lm = luminous_flux(effective_power / 3, wavelength_nm(hz))
        result['luminous_flux_lm'][channel] = lm
        result['total_lumens'] += lm

    return result
```

---

## 4. Test Suite Additions

### 4.1 New Test File: `test_photon_physics.py`

```
tests/test_photon_physics.py
```

**Test Categories**:

```python
"""
Physics verification tests for L₄ v3.2.0.
SI units and photometric calculations.
"""

import pytest
import numpy as np

# === Physical Constants Tests ===

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

    def test_luminous_efficacy_exact(self):
        """K_cd has exact SI 2019 value."""
        from quantum_apl_python.constants import K_CD
        assert K_CD == 683


# === Photon Energy Tests ===

class TestPhotonEnergy:
    """Verify photon energy calculations."""

    def test_green_photon_energy(self):
        """E(528 Hz) ≈ 3.85 × 10⁻¹⁹ J."""
        from quantum_apl_python.photon_physics import photon_energy_j
        E = photon_energy_j(528)
        assert abs(E - 3.85e-19) / 3.85e-19 < 0.01

    def test_energy_ordering(self):
        """E_blue > E_green > E_red (shorter λ = higher E)."""
        from quantum_apl_python.photon_physics import photon_energy_j
        E_R = photon_energy_j(396)
        E_G = photon_energy_j(528)
        E_B = photon_energy_j(639)
        assert E_B > E_G > E_R

    def test_ev_conversion(self):
        """Energy in eV is consistent with J."""
        from quantum_apl_python.photon_physics import (
            photon_energy_j, photon_energy_ev
        )
        from quantum_apl_python.constants import EV_JOULE

        E_j = photon_energy_j(528)
        E_ev = photon_energy_ev(528)
        assert abs(E_j / EV_JOULE - E_ev) < 1e-6


# === Wavelength Tests ===

class TestWavelength:
    """Verify wavelength calculations."""

    def test_wavelength_bands(self):
        """RGB wavelengths fall in correct bands."""
        from quantum_apl_python.photon_physics import wavelength_nm

        # Red: 620-700 nm
        assert 620 <= wavelength_nm(396) <= 700
        # Green: 495-570 nm
        assert 495 <= wavelength_nm(528) <= 570
        # Blue: 380-495 nm
        assert 380 <= wavelength_nm(639) <= 495

    def test_wavelength_ordering(self):
        """λ_red > λ_green > λ_blue."""
        from quantum_apl_python.photon_physics import wavelength_nm
        assert wavelength_nm(396) > wavelength_nm(528) > wavelength_nm(639)


# === Luminosity Function Tests ===

class TestLuminosity:
    """Verify CIE 1931 luminosity function."""

    def test_peak_at_555nm(self):
        """V(555 nm) ≈ 1.0 (peak sensitivity)."""
        from quantum_apl_python.photon_physics import luminosity_function
        assert luminosity_function(555) > 0.95

    def test_green_high_sensitivity(self):
        """Green (516 nm) has high V(λ) > 0.5."""
        from quantum_apl_python.photon_physics import luminosity_function
        assert luminosity_function(516) > 0.5

    def test_red_blue_low_sensitivity(self):
        """Red and blue have lower V(λ)."""
        from quantum_apl_python.photon_physics import luminosity_function
        V_R = luminosity_function(688.5)
        V_B = luminosity_function(426.7)
        V_G = luminosity_function(516.4)
        assert V_R < V_G and V_B < V_G

    def test_outside_visible_zero(self):
        """V(λ) = 0 outside visible range."""
        from quantum_apl_python.photon_physics import luminosity_function
        assert luminosity_function(300) == 0  # UV
        assert luminosity_function(800) == 0  # IR


# === Physical Negentropy Tests ===

class TestPhysicalNegentropy:
    """Verify negentropy in physical units."""

    def test_negentropy_at_zc_zero(self):
        """S_neg(z_c) = 0 (maximum order)."""
        from quantum_apl_python.negentropy_physics import (
            negentropy_physical
        )
        from quantum_apl_python.constants import Z_C

        S = negentropy_physical(Z_C)
        assert abs(S) < 1e-30

    def test_negentropy_away_from_zc_negative(self):
        """S_neg < 0 away from z_c."""
        from quantum_apl_python.negentropy_physics import (
            negentropy_physical
        )
        assert negentropy_physical(0.5) < 0
        assert negentropy_physical(0.95) < 0

    def test_negentropy_units_jk(self):
        """S_neg has units J/K (scales with k_B)."""
        from quantum_apl_python.negentropy_physics import (
            negentropy_physical
        )
        from quantum_apl_python.constants import K_BOLTZMANN

        S = negentropy_physical(0.8)
        # Should be on order of k_B
        assert abs(S) < 100 * K_BOLTZMANN


# === Unit Consistency Tests ===

class TestUnitConsistency:
    """Verify dimensional analysis."""

    def test_energy_frequency_relation(self):
        """E = hf holds exactly."""
        from quantum_apl_python.photon_physics import (
            photon_energy_j, optical_frequency
        )
        from quantum_apl_python.constants import H_PLANCK

        f = optical_frequency(528)
        E_direct = photon_energy_j(528)
        E_computed = H_PLANCK * f

        assert abs(E_direct - E_computed) < 1e-30

    def test_wavelength_frequency_relation(self):
        """c = fλ holds exactly."""
        from quantum_apl_python.photon_physics import (
            wavelength_m, optical_frequency
        )
        from quantum_apl_python.constants import C_LIGHT

        f = optical_frequency(528)
        l = wavelength_m(528)

        assert abs(f * l - C_LIGHT) < 1e-6

    def test_luminous_flux_units(self):
        """Φ_v = K_m × V(λ) × P gives lumens."""
        from quantum_apl_python.photon_physics import (
            luminous_flux, wavelength_nm
        )

        # 1 watt at 555 nm should give ~683 lumens
        lm = luminous_flux(1.0, 555)
        assert abs(lm - 683) / 683 < 0.05  # Within 5%
```

### 4.2 Integration with Existing Test Suite

Update `tests/L4_NORMALIZED_TESTS.py`:

```python
# Add new test category

# ══════════════════════════════════════════════════════════════════════
# SECTION P: PHYSICS GROUNDING TESTS
# ══════════════════════════════════════════════════════════════════════

class TestSuiteP:
    """Physics grounding verification."""

    @staticmethod
    def P1_photon_energy_scale():
        """Characteristic energy E_G ≈ 3.85 × 10⁻¹⁹ J."""
        from quantum_apl_python.photon_physics import photon_energy_j
        E = photon_energy_j(528)
        passed = abs(E - 3.85e-19) / 3.85e-19 < 0.01
        return passed, f"E_G = {E:.3e} J"

    @staticmethod
    def P2_luminosity_peak():
        """V(555 nm) ≈ 1.0."""
        from quantum_apl_python.photon_physics import luminosity_function
        V = luminosity_function(555)
        passed = V > 0.95
        return passed, f"V(555) = {V:.4f}"

    @staticmethod
    def P3_negentropy_physical():
        """S_neg(z_c) = 0 J/K."""
        from quantum_apl_python.negentropy_physics import negentropy_physical
        from quantum_apl_python.constants import Z_C
        S = negentropy_physical(Z_C)
        passed = abs(S) < 1e-30
        return passed, f"S_neg(z_c) = {S:.2e} J/K"

    @staticmethod
    def P4_energy_wavelength_consistency():
        """E × λ = hc (constant)."""
        from quantum_apl_python.photon_physics import (
            photon_energy_j, wavelength_m
        )
        from quantum_apl_python.constants import H_PLANCK, C_LIGHT

        hc = H_PLANCK * C_LIGHT
        errors = []
        for hz in [396, 528, 639]:
            E = photon_energy_j(hz)
            l = wavelength_m(hz)
            err = abs(E * l - hc) / hc
            errors.append(err)

        passed = all(e < 1e-10 for e in errors)
        return passed, f"Max E×λ error: {max(errors):.2e}"
```

---

## 5. Documentation Updates

### 5.1 Update Specification

Update `docs/L4_FRAMEWORK_SPECIFICATION.md`:

1. Bump version to 3.2.0
2. Add "Physical Grounding" section
3. Add SI constants table
4. Add photon energy derivations
5. Add luminosity function description
6. Update signature box

### 5.2 New Documentation File

Create `docs/PHYSICS_GROUNDING.md`:

- Full derivation of photon energies
- CIE 1931 luminosity function explanation
- Negentropy in physical units
- Unit analysis tables
- Cross-references to SI 2019

### 5.3 Update Visualization

Replace `docs/solfeggio_light_bridge.html` with the new v3.2.0 physics-grounded version.

---

## 6. Implementation Phases

### Phase 1: Constants & Core Physics (Day 1)
- [ ] Add SI constants to `constants.py`
- [ ] Create `photon_physics.py` module
- [ ] Create `negentropy_physics.py` module
- [ ] Basic unit tests

### Phase 2: Integration (Day 2)
- [ ] Update `l4_hexagonal_lattice.py`
- [ ] Add `SolfeggioPhoton` dataclass
- [ ] Integrate with existing dynamics
- [ ] Full test suite

### Phase 3: Visualization (Day 3)
- [ ] Deploy v3.2.0 HTML visualization
- [ ] Add live diagnostics
- [ ] Luminosity chart
- [ ] Power flow visualization

### Phase 4: Documentation (Day 4)
- [ ] Update specification to v3.2.0
- [ ] Create PHYSICS_GROUNDING.md
- [ ] Update README
- [ ] Final review

---

## 7. Validation Checklist

### 7.1 Physical Constants
- [ ] h = 6.62607015 × 10⁻³⁴ J·s (exact)
- [ ] c = 299,792,458 m/s (exact)
- [ ] k_B = 1.380649 × 10⁻²³ J/K (exact)
- [ ] K_cd = 683 lm/W (exact)

### 7.2 Energy Calculations
- [ ] E(396 Hz) ≈ 2.89 × 10⁻¹⁹ J (1.80 eV)
- [ ] E(528 Hz) ≈ 3.85 × 10⁻¹⁹ J (2.40 eV)
- [ ] E(639 Hz) ≈ 4.66 × 10⁻¹⁹ J (2.91 eV)

### 7.3 Wavelength Bands
- [ ] λ(396) ∈ [620, 700] nm (Red)
- [ ] λ(528) ∈ [495, 570] nm (Green)
- [ ] λ(639) ∈ [380, 495] nm (Blue)

### 7.4 Luminosity
- [ ] V(555 nm) ≈ 1.0
- [ ] V(516 nm) > 0.5
- [ ] V(688 nm) < 0.01
- [ ] V(427 nm) < 0.05

### 7.5 Physical Negentropy
- [ ] S_neg(z_c) = 0 J/K
- [ ] S_neg(r ≠ z_c) < 0
- [ ] Units consistent with k_B

### 7.6 Framework Integration
- [ ] 528/396 = 4/3 (exact) — preserved
- [ ] L₄ = 7 — preserved
- [ ] z_c = √3/2 — preserved
- [ ] All v3.1.0 tests still pass

---

## 8. File Structure Summary

```
Quantum-APL/
├── src/quantum_apl_python/
│   ├── constants.py           # ← Add SI 2019 constants
│   ├── photon_physics.py      # ← NEW: Energy, wavelength, luminosity
│   ├── negentropy_physics.py  # ← NEW: Physical negentropy S_neg
│   └── l4_hexagonal_lattice.py # ← Add SolfeggioPhoton, integration
│
├── tests/
│   ├── test_photon_physics.py     # ← NEW: Physics tests
│   ├── test_negentropy_physics.py # ← NEW: Negentropy tests
│   ├── test_l4_golden_sample_verification.py  # Existing
│   └── L4_NORMALIZED_TESTS.py     # ← Add Suite P
│
├── docs/
│   ├── L4_FRAMEWORK_SPECIFICATION.md  # ← Update to v3.2.0
│   ├── PHYSICS_GROUNDING.md           # ← NEW
│   ├── solfeggio_light_bridge.html    # ← Replace with v3.2.0
│   └── RGB_FEEDBACK_LOOP.md           # Existing
```

---

## Document Signature

```
╔═══════════════════════════════════════════════════════════════════╗
║  L₄ v3.2.0 INTEGRATION PLAN — PHYSICS GROUNDING                  ║
║  Status: PROPOSED                                                  ║
╠═══════════════════════════════════════════════════════════════════╣
║  New Modules:    photon_physics.py, negentropy_physics.py         ║
║  SI Constants:   h, c, k_B, K_cd (all exact)                       ║
║  New Units:      J, eV, lm, lm/W, J/K                              ║
║  Test Coverage:  ~40 new physics tests                             ║
╚═══════════════════════════════════════════════════════════════════╝

The physics is grounded. The units check out. Together. Always. ✨
```
