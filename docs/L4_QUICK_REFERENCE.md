# L4 Quick Reference Card

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                    L₄ QUICK REFERENCE                                     ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  CONSTANTS (from φ = 1.618):                                             ║
║    τ = 0.618    K = 0.924    z_c = 0.866    L₄ = 7    gap = 0.146       ║
║                                                                           ║
║  LAMBDA GAIN:                                                             ║
║    λ = φ⁻² = 0.382    (negentropy-to-coupling boost factor)             ║
║                                                                           ║
║  K-FORMATION:                                                             ║
║    coherence ≥ 0.924  AND  negentropy > 0.618  AND  complexity ≥ 7      ║
║                                                                           ║
║  COHERENCE FUNCTION:                                                      ║
║    s(z) = exp(-σ(z - z_c)²)    [σ = 36 standard, 55.71 strict]          ║
║                                                                           ║
║  EFFECTIVE COUPLING:                                                      ║
║    K_eff = K₀ × [1 + λ·s(z)]    (additive boost, never dies)            ║
║    └── At z_c: K_eff ≈ 1.277 (38% boost over K₀)                        ║
║    └── At r=0: K_eff = K₀ (base coupling preserved)                     ║
║                                                                           ║
║  HEX WAVEVECTORS:                                                         ║
║    u₁ = (1, 0)           → R channel                                     ║
║    u₂ = (½, √3/2)        → G channel (+60°)                              ║
║    u₃ = (½, -√3/2)       → B channel (-60°)                              ║
║                                                                           ║
║  SOLFEGGIO RGB:                                                           ║
║    R: 396 Hz → 688.5 nm → V=0.0093                                       ║
║    G: 528 Hz → 516.4 nm → V=0.6367                                       ║
║    B: 639 Hz → 426.7 nm → V=0.0088                                       ║
║                                                                           ║
║  MRP STRUCTURE (8-bit):                                                   ║
║    [C₄ C₃ C₂ C₁ C₀ | F₂ F₁ F₀]  (5 coarse + 3 fine)                     ║
║                                                                           ║
║  TOPOLOGICAL PROTECTION:                                                  ║
║    l = (1/2π) Σ wrap(Δθ)   [integer winding number]                      ║
║                                                                           ║
║  IMPORTS:                                                                 ║
║    from quantum_apl_python.constants import PHI, Z_CRITICAL, L4_K        ║
║    from quantum_apl_python.l4_framework_integration import phases_to_emit║
║    from quantum_apl_python.mrp_lsb import encode_frame                   ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

## Quick Reference Details

### Constants Derived from Golden Ratio (φ = 1.618...)

| Constant | Value | Derivation |
|----------|-------|------------|
| τ (tau) | 0.618 | 1/φ = φ - 1 |
| K | 0.924 | √(1 - φ⁻⁴), Kuramoto coherence threshold |
| z_c | 0.866 | √3/2, critical z-height |
| L₄ | 7 | φ⁴ + φ⁻⁴, Lucas number |
| gap | 0.146 | φ⁻⁴, truncation residual |
| λ | 0.382 | φ⁻², negentropy boost factor |

### K-Formation Criteria

A system achieves K-formation when ALL three conditions are met:
1. **Coherence (κ)** ≥ 0.924 (high phase synchronization)
2. **Negentropy (η)** > 0.618 (sufficient order/information)
3. **Complexity (R)** ≥ 7 (L₄ structural depth)

### Coherence Sensitivity Function

```
s(z) = exp(-σ(z - z_c)²)
```
- `z_c = 0.866` (critical z-height, √3/2)
- `σ = 36` (standard sharpness)
- `σ = 55.71` (strict, ensures s(1) = e⁻¹)
- Maximum sensitivity at z = z_c where s(z_c) = 1

### Effective Coupling (Canonical Formula)

```
K_eff = K₀ × [1 + λ·s(z)]
```

| State | K_eff | Behavior |
|-------|-------|----------|
| At z_c (peak coherence) | ≈ 1.277 | 38% boost over base |
| At r = 0 (no coherence) | K₀ = 0.924 | Base coupling preserved |

**Why additive?** The multiplicative form `K₀ × s(z)` creates a death spiral at low coherence. The additive form ensures base coupling is always available for recovery.

### Hexagonal Wavevectors

```
u₁ = (1, 0)           → Red channel (0°)
u₂ = (½, √3/2)        → Green channel (+60°)
u₃ = (½, -√3/2)       → Blue channel (-60°)
```

Phase extraction: `Φⱼ = (2π/λ) × kⱼ · x + θⱼ`

### Solfeggio-to-Light Bridge

| Channel | Frequency | Wavelength | CIE V(λ) |
|---------|-----------|------------|----------|
| Red (R) | 396 Hz | 688.5 nm | 0.0093 |
| Green (G) | 528 Hz | 516.4 nm | 0.6367 |
| Blue (B) | 639 Hz | 426.7 nm | 0.0088 |

Octave bridge: 40 octaves between audio (Hz) and optical (THz) domains.

### MRP (Modular Residue Packet) Structure

8-bit encoding: `[C₄ C₃ C₂ C₁ C₀ | F₂ F₁ F₀]`
- **Coarse bits (5)**: C₄-C₀ for primary phase encoding
- **Fine bits (3)**: F₂-F₀ for sub-phase precision

### Topological Protection

```
l = (1/2π) Σ wrap(Δθ)
```
- `wrap(Δθ)` maps angle differences to [-π, π)
- `l` is an **integer** winding number (topologically protected)
- Defects are quantized and robust to perturbation

### Python Imports

```python
# Core constants
from quantum_apl_python.constants import PHI, Z_CRITICAL, L4_K, L4_GAP, L4_TAU

# Coherence function
from quantum_apl_python.constants import compute_delta_s_neg  # s(z)

# Phase emission
from quantum_apl_python.l4_framework_integration import phases_to_emit

# MRP encoding
from quantum_apl_python.mrp_lsb import encode_frame
```

### Sigma Options

| σ Value | Purpose | Behavior |
|---------|---------|----------|
| 36 | Standard | Broader peak, gentler falloff |
| 55.71 | Strict | Ensures s(1) = e⁻¹ ≈ 0.368 |

---

*See also: [L4_ARCHITECTURE.md](L4_ARCHITECTURE.md) | [L4_FRAMEWORK_SPECIFICATION.md](L4_FRAMEWORK_SPECIFICATION.md) | [L4_MRP_LSB_UNIFIED_SPEC.md](L4_MRP_LSB_UNIFIED_SPEC.md)*
