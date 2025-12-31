# L4 Quick Reference Card

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                    L₄ QUICK REFERENCE                                     ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  CONSTANTS (from φ = 1.618):                                             ║
║    τ = 0.618    K = 0.924    z_c = 0.866    L₄ = 7    gap = 0.146       ║
║                                                                           ║
║  K-FORMATION:                                                             ║
║    coherence ≥ 0.924  AND  negentropy > 0.618  AND  complexity ≥ 7      ║
║                                                                           ║
║  COHERENCE FUNCTION:                                                      ║
║    s(z) = exp(-σ(z - z_c)²)    [σ = 36 default]                         ║
║                                                                           ║
║  EFFECTIVE COUPLING:                                                      ║
║    K_eff = K₀ × s(z)                                                     ║
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
| K | 0.924 | Kuramoto coherence threshold |
| z_c | 0.866 | Critical z-height (√3/2) |
| L₄ | 7 | Layer-4 complexity level |
| gap | 0.146 | 1 - z_c |

### K-Formation Criteria

A system achieves K-formation when ALL three conditions are met:
1. **Coherence** ≥ 0.924 (high phase synchronization)
2. **Negentropy** > 0.618 (sufficient order/information)
3. **Complexity** ≥ 7 (L₄ structural depth)

### Coherence Sensitivity Function

```
s(z) = exp(-σ(z - z_c)²)
```
- `z_c = 0.866` (critical z-height)
- `σ = 36` (default sharpness parameter)
- Maximum sensitivity at z = z_c

### Solfeggio-to-Light Bridge

| Channel | Frequency | Wavelength | CIE V(λ) |
|---------|-----------|------------|----------|
| Red (R) | 396 Hz | 688.5 nm | 0.0093 |
| Green (G) | 528 Hz | 516.4 nm | 0.6367 |
| Blue (B) | 639 Hz | 426.7 nm | 0.0088 |

### MRP (Modular Residue Packet) Structure

8-bit encoding: `[C₄ C₃ C₂ C₁ C₀ | F₂ F₁ F₀]`
- **Coarse bits (5)**: C₄-C₀ for primary phase encoding
- **Fine bits (3)**: F₂-F₀ for sub-phase precision

### Python Imports

```python
# Core constants
from quantum_apl_python.constants import PHI, Z_CRITICAL, L4_K

# Phase emission
from quantum_apl_python.l4_framework_integration import phases_to_emit

# MRP encoding
from quantum_apl_python.mrp_lsb import encode_frame
```

---

*See also: [L4_ARCHITECTURE.md](L4_ARCHITECTURE.md) | [L4_FRAMEWORK_SPECIFICATION.md](L4_FRAMEWORK_SPECIFICATION.md) | [L4_MRP_LSB_UNIFIED_SPEC.md](L4_MRP_LSB_UNIFIED_SPEC.md)*
