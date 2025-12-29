# φ⁻¹ (Golden Ratio Inverse) — Definition, Identities, and Role

This note consolidates the definition of φ⁻¹, its exact identities, and how it is used in the helix system as the K‑formation coherence threshold, distinct from the geometric lens `z_c = √3/2`.

**L₄-Helix Context:** φ⁻¹ = L4_PARADOX (threshold 1) in the 9-threshold system. The gap = φ⁻⁴ provides the fundamental normalization unit, connecting φ⁻¹ to the Lucas-4 foundation (L₄ = φ⁴ + φ⁻⁴ = 7).

## Definitions
- φ = (1 + √5) / 2 ≈ 1.6180339887
- φ⁻¹ = 1/φ = φ − 1 = (√5 − 1) / 2 ≈ 0.6180339887

## Identities (exact)
- φ² = φ + 1
- (φ⁻¹)² + φ⁻¹ − 1 = 0 (minimal polynomial)
- φ · φ⁻¹ = 1
- Fixed points:
  - φ = 1 + 1/φ
  - φ⁻¹ = 1 / (1 + φ⁻¹)
- Continued fractions:
  - φ = [1; 1, 1, 1, …]
  - φ⁻¹ = [0; 1, 1, 1, …]

## Role in the System
- K‑formation coherence threshold: η > φ⁻¹ signals sufficient coherence for emergence checks.
- Centralization:
  - JS: `PHI`, `PHI_INV` in `src/constants.js`
  - Python: `PHI`, `PHI_INV` in `src/quantum_apl_python/constants.py`
- Integrity tests:
  - φ · φ⁻¹ ≈ 1 checked in constants tests.

## Separation from z_c
- `z_c = √3/2 ≈ 0.8660254` (THE LENS) is the geometric/information threshold for integrated regime and negative‑entropy geometry stability.
- φ⁻¹ (≈ 0.618) is a lower coherence threshold used in K‑formation criteria.
- They serve different purposes and are both anchored analytically.

## L₄-Helix Gap Normalization

φ⁻¹ is the foundation of the L₄-Helix 9-threshold system through the gap:

```
gap = φ⁻⁴ ≈ 0.1459 (truncation residual)
L₄ = φ⁴ + φ⁻⁴ = 7 (Lucas-4 identity)
```

The gap normalizes all 9 thresholds. Key derived constants:
- K² = 1 − gap = 1 − φ⁻⁴ (L4_ACTIVATION ≈ 0.854)
- K = √(1 − gap) (L4_K_FORMATION ≈ 0.924)
- τ = φ⁻¹ (L4_PARADOX ≈ 0.618)

See `docs/PHYSICS_GROUNDING.md` for nuclear spin physics derivation.

## References
- docs/CONSTANTS_ARCHITECTURE.md (inventory and invariants)
- docs/Z_CRITICAL_LENS.md (lens authority)
- src/constants.js, src/quantum_apl_python/constants.py

