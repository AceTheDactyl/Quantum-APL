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

This keeps the “z-axis” semantics synchronized with the Helix Bridge tooling without copying the entire orchestration stack into this repository. If you need the raw orchestrator or witness logs, start with:

- `/home/acead/Helix Bridge/Helix Bridge/meta_orchestrator.py`
- `/home/acead/TRIAD083 Multi Agent System/WumboIsBack-main/triad-083-physics-architecture-code.md`

Future work:

- Feed the `HelixAPLMapper` hints into the JS engine so operator legality can be steered by Helix coordinates at runtime.
- Import the Helix witness logs into new fixtures for regression tests (z-coordinate trajectories, harmonic transitions, etc.).
