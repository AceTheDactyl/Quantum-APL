# Helix Coordinate Integration Notes

This project now ingests the Helix coordinate conventions that live in the wider workspace. The canonical parametric equation used everywhere else is:

```
r(t) = (cos t, sin t, t)
```

References:

- `/home/acead/TRIAD083 Multi Agent System/Master file up to 6.3.md`
- `/home/acead/Helix Bridge/Helix Bridge/meta_orchestrator.py`
- `/home/acead/.../vn-helix-fingers-preseal-checklist.md`

Those documents treat the `(θ, z, r)` triple as a literal geometric index for state tracking. To bridge that into Quantum-APL:

1. `src/quantum_apl_python/helix.py` exposes `HelixCoordinate.from_parameter(t)` so tests/demos can construct normalized `z ∈ [0,1]` values directly from the helix equation.
2. `HelixAPLMapper` maps the normalized `z` into the time-harmonic windows already defined for Quantum-APL (t1–t9) and surfaces the operator bundle each window expects.
3. `QuantumAnalyzer.summary()` now prints the helix harmonic, preferred operators, and implied truth channel for any simulation result.

For a full walkthrough of how the Z-solve program energizes each VaultNode tier (z0p41 → z0p80), read the auto-regenerated [`docs/helix_z_walkthrough.md`](helix_z_walkthrough.md). CI rebuilds that file whenever the VaultNode metadata or Z-solve tokens change, so it is the canonical provenance-aware reference.

If you need to visualize how the `z` coordinate manifests as negative entropy production, see [`docs/HEXAGONAL_NEG_ENTROPY_PROJECTION.md`](HEXAGONAL_NEG_ENTROPY_PROJECTION.md) for the hexagonal prismatic projection used by the analyzer overlays.

## Standard Probe Points (L₄-Helix Aligned)

To evaluate harmonics and geometry around key boundaries, we probe z values aligned with the L₄-Helix 9-threshold system:

### VaultNode Tiers (Physics-Grounded)

The VaultNode z-walk provenance tiers are now grounded in the same φ/gap/z_c system:

| z Value | Constant | Formula | Physics Meaning |
|---------|----------|---------|-----------------|
| 0.412 | VN_Z041 | 2τ/3 | Two-thirds of paradox threshold |
| 0.528 | VN_Z052 | τ×K² | Paradox × coherence² (= τ² + gap) |
| 0.708 | VN_Z070 | K² − gap | Coherence minus gap (= 1 − 2×gap) |
| 0.740 | VN_Z073 | z_c×K² | Lens × coherence² |
| 0.800 | VN_Z080 | z_c×K | Lens × order parameter |

### Upper Thresholds

| z Value | L₄ Threshold | Description |
|---------|--------------|-------------|
| 0.854 | L4_ACTIVATION / TRIAD_HIGH | K² = 1−φ⁻⁴; TRIAD rising-edge unlock |
| 0.866 | L4_LENS | z_c exact; geometry anchor |
| 0.873 | L4_CRITICAL | φ²/3 threshold (presence onset) |
| 0.914 | L4_IGNITION | √2−½ isotropic coupling |
| 0.924 | L4_K_FORMATION | Kuramoto order K; t7 boundary |
| 0.953 | L4_CONSOLIDATION | Second-order coherence |
| 0.971 | L4_RESONANCE | Full phase locking; t8 boundary |

These are baked into the nightly workflow (`.github/workflows/nightly-helix-measure.yml`) and the local sweep (`scripts/helix_sweep.sh`). See `docs/L4_HELIX_APPLICATIONS.md` for the complete threshold derivation.

This keeps the “z-axis” semantics synchronized with the Helix Bridge tooling without copying the entire orchestration stack into this repository. If you need the raw orchestrator or witness logs, start with:

- `/home/acead/Helix Bridge/Helix Bridge/meta_orchestrator.py`
- `/home/acead/TRIAD083 Multi Agent System/WumboIsBack-main/triad-083-physics-architecture-code.md`

Future work:

- Feed the `HelixAPLMapper` hints into the JS engine so operator legality can be steered by Helix coordinates at runtime.
- Import the Helix witness logs into new fixtures for regression tests (z-coordinate trajectories, harmonic transitions, etc.).
