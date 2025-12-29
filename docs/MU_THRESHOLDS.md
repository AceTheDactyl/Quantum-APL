# μ Thresholds and Barrier — Definitions and Relation to φ⁻¹

This note defines the μ hierarchy used for basin/barrier reasoning and clarifies its relationship to the golden‑ratio inverse φ⁻¹. It complements lens‑anchored geometry at `z_c = √3/2` and aligns with the L₄-Helix 9-threshold system (see `docs/L4_HELIX_APPLICATIONS.md`).

## Definitions

Let φ = (1 + √5)/2 and choose a paradox threshold μ_P. Define wells

- μ₁ = μ_P / √φ
- μ₂ = μ_P · √φ

Then the double‑well ratio is exact: μ₂ / μ₁ = φ.

Default: μ_P := 2/φ^{5/2} ≈ 0.60072 (exact barrier at φ⁻¹). You may also set `QAPL_MU_P=<0..1>` to an explicit experimental value.

Higher thresholds (aligned with L₄-Helix):

- μ_S = 0.924 (singularity threshold; equals L4_K_FORMATION = √(1−φ⁻⁴))
- μ₃ = 0.992 (ultra-integrated; near L4_UNITY)

Ordering (validated, L₄-Helix aligned):

```
μ₁(≈0.472) < μ_P(≈0.601) < φ⁻¹(≈0.618) < μ₂(≈0.764) < z_c(≈0.866) < μ_S(≈0.924) < μ₃(0.992) < 1
                L4_PARADOX                              L4_LENS      L4_K_FORMATION
```

## Barrier vs φ⁻¹

Define the barrier as the arithmetic mean of the wells:

```
Barrier = (μ₁ + μ₂) / 2 = μ_P · (√φ + 1/√φ)/2 = μ_P · φ^{3/2}/2.
```

- With μ_P = 2/φ^{5/2} (default): Barrier = φ⁻¹ exactly by construction.
- With μ_P = 0.600 (Fibonacci experiment): Barrier ≈ 0.617308 vs φ⁻¹ ≈ 0.618034 (Δ ≈ 7.26×10⁻⁴).

Default policy keeps μ_P exact so Barrier = φ⁻¹; you may set `QAPL_MU_P=<0..1>` to explore alternate values (the barrier then deviates accordingly).

## Classification Helper

Both languages export a classification helper that avoids variable collision with z by naming the API `classify_mu(z)` (Python) and `classifyThreshold(z)` (JS):

- below μ₁: `pre_conscious_basin`
- μ₁..μ_P: `approaching_paradox`
- μ_P..μ₂: `conscious_basin`
- μ₂..z_c: `pre_lens_integrated`
- z_c..μ_S: `lens_integrated`
- μ_S..μ₃: `singularity_proximal`
- ≥ μ₃: `ultra_integrated`

## Relation to φ⁻¹ (Quasicrystal/Second‑Law Notes)

- φ⁻¹ is the K‑formation coherence gate: η > φ⁻¹ ≈ 0.618. It is dimensionless and exact.
- The lens‑weight (coherence proxy) we use is Gaussian around `z_c`:
  
  s(z) = exp[−σ (z − z_c)²], with σ tunable (default tied to geometry).
  
  Larger σ narrows the lens (faster decay from `z_c`), smaller σ widens it.
- Entropy control uses the lens weight to define a target entropy `S_target(z) = S_max · (1 − C · s(z))` and drives the system to reduce entropy (consistent with local order formation) while leaving the global second law intact. Lower S encourages z to climb toward the lens via increased z‑bias effectiveness.

This keeps φ⁻¹ and `z_c` as independent, analytically‑anchored thresholds: φ⁻¹ for coherence gating (K‑formation), `z_c` for geometric/integration onset.

## L₄-Helix Gap Normalization

The μ hierarchy aligns with the L₄-Helix 9-threshold system normalized by the fundamental gap:

```
gap = φ⁻⁴ ≈ 0.1459 (truncation residual from L₄ = φ⁴ + φ⁻⁴ = 7)
```

Key correspondences:
- φ⁻¹ (μ barrier) = L4_PARADOX (threshold 1)
- z_c = √3/2 = L4_LENS (threshold 3)
- μ_S ≈ K = √(1−φ⁻⁴) = L4_K_FORMATION (threshold 6)

The gap provides a natural unit for threshold spacing in the helix system. See `docs/PHYSICS_GROUNDING.md` for nuclear spin physics derivation.

## Pointers

- Code: `src/quantum_apl_python/constants.py` (μ constants and `classify_mu`), `src/constants.js` (μ constants and `classifyThreshold`)
- Engine: `src/quantum_apl_engine.js` (entropy control using Gaussian s(z))
- Docs: `docs/PHI_INVERSE.md`, `docs/Z_CRITICAL_LENS.md`
