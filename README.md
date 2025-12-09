# Alpha-Physical Language (APL) — Seven Sentences Test Pack

A minimal operator grammar for describing physical system behaviors across geometry, waves, chemistry, and biology.

## Overview

**APL (Alpha-Physical Language)** is an experimental framework that uses compact "sentences" to predict physical regimes across diverse domains. This repository contains a test pack designed to allow independent teams to validate whether APL's core operator language has genuine physical content.

The test pack translates 7 compact APL sentences into **falsifiable, cross-domain hypotheses** that can be probed with standard models:
- Navier–Stokes (fluid dynamics)
- Wave equations (electromagnetics, acoustics)
- Reaction–diffusion (chemistry)
- Phase-field models (interfaces, materials)
- Polymer growth / aggregation

## Core Concept

APL describes physical systems using:

### Fields ("Spirals")
- **Φ** — Structure field (geometry, lattice, boundaries)
- **e** — Energy field (waves, thermodynamics, flows)
- **π** — Emergence field (information, chemistry, biology)

### Universal Operations
- `()` — Boundary / containment
- `×` — Fusion / convergence / joining
- `^` — Amplify / gain
- `%` — Decohere / noise / reset
- `+` — Group / aggregation / routing
- `–` — Separate / splitting / fission

### Operator States (UMOL: Universal Modulation Operator Law)
- **u** (𝒰) — Expansion / forward projection
- **d** (𝒟) — Collapse / backward integration
- **m** (CLT) — Modulation / coherence lock

An APL sentence has the form:
```
[Direction][Op] | [Machine] | [Domain] → [Regime/Behavior]
```

For example: `u^|Oscillator|wave` reads as "Forward amplification in an oscillatory machine in a wave domain."

## The Seven Test Sentences

Each sentence is a **testable hypothesis** predicting that specific operator-machine-domain combinations statistically favor particular physical regimes:

| # | Sentence | Predicted Regime | Domain |
|---|----------|------------------|--------|
| **A3** | `u^|Oscillator|wave` | Closed vortex / recirculation | Wave dynamics |
| **A7** | `u%|Reactor|wave` | Turbulent decoherence | Flow/wave systems |
| **A1** | `d()|Conductor|geometry` | Isotropic lattice / sphere | Geometry/interfaces |
| **A4** | `m×|Encoder|chemistry` | Helical encoding | Chemistry/polymers |
| **A5** | `u×|Catalyst|chemistry` | Branching networks | Chemistry/growth |
| **A6** | `u+|Reactor|wave` | Focusing jet / beam | Fluid/plasma/wave |
| **A8** | `m()|Filter|wave` & `d×|Catalyst|chemistry` | Adaptive filter / selectivity | Wave & chemistry |

### Interpretation Rule

For all sentences:
```
LHS → RHS
```
means:

> If a system is built to match the **left-hand side** (LHS) structure and driving, then the **right-hand side** (RHS) regime should appear **more often, more strongly, or at lower thresholds** than in controls that break the LHS structure, with all else as equal as possible.

**Evidence FOR APL:** Clear, reproducible overrepresentation of the RHS regime under LHS conditions vs. controls.

**Evidence AGAINST APL:** No such bias, or controls produce the RHS regime equally or more often.

## Repository Structure

```
APL/
├── README.md                           # This file
├── apl-operators-manual.tex            # Complete operator reference (LaTeX)
├── apl-seven-sentences-test-pack.tex   # Complete test protocol (LaTeX)
├── COMPILE_INSTRUCTIONS.md             # LaTeX compilation guide
└── docs/                               # Documentation and compiled outputs
    ├── index.html                      # HTML version of operator's manual
    ├── apl-operators-manual.pdf        # PDF version (auto-compiled)
    └── apl-seven-sentences-test-pack.pdf  # Test pack PDF
```

## Testing Strategy

For each sentence, the recommended approach:

1. **Choose a standard model** appropriate to the domain
   - Geometry: phase-field, Cahn–Hilliard, curvature flow
   - Flows/waves: Navier–Stokes, lattice Boltzmann, wave equation
   - Chemistry: reaction–diffusion, polymerization, DLA, kinetic Monte Carlo

2. **Implement the LHS conditions**
   - `u^`: Add gain/amplification at resonant modes
   - `u%`: Add explicit stochastic/decohering forcing
   - `d()`: Allow boundaries to relax/collapse under isotropic energy
   - `m()`: Modulate boundaries in response to passing modes
   - `u×`/`d×`: Implement forward-biased or collapse–fusion catalysts
   - `u+`: Add grouping/convergent geometry or fields

3. **Design matched controls**
   - Remove or invert the key operator
   - Keep everything else comparable

4. **Define regime metrics** (A1–A8)
   - A1: Sphericity, surface/volume ratio, packing isotropy
   - A3: Vortex count/lifetime, closed streamline fraction
   - A4: Helical order parameters, information capacity
   - A5: Fractal dimension, branching degree
   - A6: Jet opening angle, centerline coherence
   - A7: Spectral width, RMS fluctuations, Lyapunov exponents
   - A8: Adaptive sharpening, retuning capability

5. **Sweep parameters** and compare
   - Drive strength, noise amplitude, surface tension, catalytic bias, etc.
   - Run multiple realizations
   - Check whether LHS conditions robustly bias metrics toward target regime

## Preliminary Results

The test pack includes two toy numerical checks:

- **A1 (Isotropic cluster):** 2D point collapse under isotropic central force → circular, angle-isotropic cluster ✓
- **A5 (Branching networks):** 2D diffusion-limited aggregation → fractal branching structure (D ≈ 1.2) ✓

These are minimal sandbox experiments consistent with APL predictions. Full testing requires domain-appropriate models across all seven sentences.

## Requirements

**To use this test pack, you need:**
- Basic familiarity with physics, chemistry, and data analysis
- NO prior knowledge of CET or other APL meta-structures
- Access to standard simulation tools (Python/NumPy, MATLAB, COMSOL, OpenFOAM, etc.)

**Assumptions:**
- Independent testing by teams without invested stake in APL
- Falsifiable approach: both positive and negative results are valuable

## Documentation

### APL Operator's Manual

A comprehensive reference guide for APL operators, syntax, and usage patterns is available in multiple formats:

- **HTML Version:** Open `docs/index.html` in your browser for an interactive, responsive manual
- **PDF Version:** Automatically compiled from LaTeX source via GitHub Actions
- **GitHub Pages:** Available online if GitHub Pages is enabled for this repository

The manual includes:
- Complete operator reference with symbols and meanings
- Field definitions (the three "spirals": Φ, e, π)
- Operator state modulation (UMOL)
- Machine contexts and domains
- Syntax rules and sentence structure
- Usage patterns and examples
- Quick reference tables

### Helix Coordinate Mapping

If you are using the Helix coordinate system from the VaultNode tooling, read `docs/HELIX_COORDINATES.md`. It explains how the parametric helix equation (r(t)=(cos t, sin t, t)) maps into the Quantum-APL z-axis and how the new `HelixAPLMapper` surfaces the recommended harmonics/operators directly from the normalized z value.

#### TRIAD Gate vs. Critical Lens

- Geometry and analytics continue to treat the critical point as THE LENS at `z_c = √3/2 ≈ 0.8660254`.
- The runtime can unlock a TRIAD gate for t6 at `z = 0.83` after three distinct z≥0.85 passes (with hysteresis at 0.82), modeled as three “single helix” completions:
  - Auto‑unlock (in‑session): the bridge increments completions on each rising edge z≥0.85 and flips the engine to use the 0.83 t6 gate after the third pass.
  - Environment knobs (optional): `QAPL_TRIAD_COMPLETIONS` (≥3 unlocks) and `QAPL_TRIAD_UNLOCK=1/true` to force unlock.
  - This affects only the t6 boundary in the helix advisor. Hex‑prism geometry continues to use `z_c`.

### Alpha Programming Language Bridge

The upstream Alpha Programming Language assets live in `/home/acead/Aces-Brain-Thpughts/APL`. This repository now ingests that operator grammar via `src/quantum_apl_python/alpha_language.py`. The analyzer synthesizes the Seven Sentence test pack tokens from helix-driven operator windows and prints the matched sentence/regime next to every simulation summary. See `docs/ALPHA_SYNTAX_BRIDGE.md` for the full crosswalk produced after sweeping the workspace for helix/Z references.

### System Architecture Overview

For a complete end-to-end diagram of the integrated system (Python API → JavaScript engines → classical stacks → measurement flow), see `docs/SYSTEM_ARCHITECTURE.md`. It reproduces the final delivery schematic showing each layer, the Z-axis map, truth states, performance metrics, and file organization.

## Constants

- Lens specification: `docs/Z_CRITICAL_LENS.md` — authoritative definition of the critical lens `z_c = √3/2 ≈ 0.8660254`, methodology of use across engine/geometry/analyzer/bridge, and validation notes.
- Constants research: `docs/CONSTANTS_RESEARCH.md` — survey of constants across the helix path (harmonic zoning, geometry projection, pump/engine parameters, selection heuristics) and maintenance policy.
- Single sources of truth:
  - Python: `src/quantum_apl_python/constants.py`
  - JavaScript: `src/constants.js`
  - Do not inline numeric thresholds; always import from these modules.

### Quick Flags (env)

Control lens width, geometry width, analyzer overlays, and blending at the source. These are read by both JS and Python modules.

- `QAPL_LENS_SIGMA` — coherence width σ for s(z) around z_c (default `36.0`)
- `QAPL_GEOM_SIGMA` — geometry width σ for ΔS_neg (default: falls back to `QAPL_LENS_SIGMA`)
- `QAPL_ANALYZER_OVERLAYS=1` — draw μ markers and s(z) overlay in analyzer plots
- `QAPL_BLEND_PI=1` — enable Π/loc cross‑fade above the lens: `w_pi = s(z)`, `w_loc = 1 − w_pi`
- `QAPL_MU_P=<0..1>` — optional μ_P override (default exact `2/φ^{5/2}` → barrier = φ⁻¹)

Analyzer summary prints φ⁻¹ and z_c at full precision and reports the μ barrier line:

```
φ⁻¹ = 0.6180339887498948
z_c = 0.8660254037844386
μ barrier: φ⁻¹ exact @ 0.6180339887498948
```

## Testing

### JavaScript Tests

```bash
npm install                                # Install dev dependencies (ajv)
node tests/test_apl_measurements.js        # APL measurement tests
node tests/test_schema_validation.js       # JSON schema validation (Ajv)
node tests/test_seeded_selection.js        # Reproducible RNG tests
```

### Schema Validation

JSON schemas for geometry sidecars and APL bundles live in `schemas/`:
- `schemas/geometry-sidecar.schema.json` — 63-vertex hex prism geometry
- `schemas/apl-bundle.schema.json` — APL token array validation

Run validation tests: `node tests/test_schema_validation.js`

### Reproducible Simulations

Set `QAPL_RANDOM_SEED` to enable deterministic sampling:

```bash
QAPL_RANDOM_SEED=12345 qapl-run --steps 3 --mode measured --output out.json
```

See `docs/REPRODUCIBLE_RESEARCH.md` for details.

## Z Pump Profiles, Entropy Control, and CLI Shortcuts

The APL-aligned z pump raises z using physically meaningful operator sequences (u^, ×, and Π lock) and classical feedback. You can control its behavior via profiles and CLI sugar.

### Profiles

- gentle: lower coupling (gain 0.08, sigma 0.16), cadence: × every 3, Π lock every 9, blend 0.5·Ω + 0.5·target
- balanced (default): gain 0.12, sigma 0.12, cadence: × every 2, Π lock every 6, blend 0.3·Ω + 0.7·target
- aggressive: higher coupling (gain 0.18, sigma 0.10), cadence: × every 1, Π lock every 4, blend 0.2·Ω + 0.8·target

All profiles preserve hex‑prism geometry using the lens `z_c = √3/2` and obey APL operator semantics (u^ → coherent e excitation; × → Φ fusion; Π lock → integrated regime).

### Entropy Control (optional)

You can enable a lightweight entropy control law that nudges the engine toward a target entropy tied to lens‑anchored coherence:

u_S = k_s · (S_target(z) − S) / S_max,    S_target(z) = S_max · (1 − C · ΔS_neg(z))

- S(ρ) is von Neumann entropy; S_max = log₂(dimTotal)
- ΔS_neg(z) is centered at `z_c` and decreases with |z − z_c|
- Default: disabled. Enable via engine config or bridge:

```js
const engine = new QuantumAPL({ entropyCtrlEnabled: true, entropyCtrlGain: 0.2, entropyCtrlCoeff: 0.5 });
```

With control enabled, the effective z‑bias gain is adjusted each step; geometry remains lens‑anchored and TRIAD policy is unchanged.

### CLI Sugar

- Shortcut mode (maps `--steps` to cycles):
  - `qapl-run --z-pump 0.86 --steps 120`
- Explicit pump flags:
  - `qapl-run --mode z_pump --z-pump-target 0.86 --z-pump-cycles 120 --z-pump-profile aggressive`

Environment equivalents: `QAPL_PUMP_TARGET`, `QAPL_PUMP_CYCLES`, `QAPL_PUMP_PROFILE`.

- Default target: if a pump target is omitted, the bridge resolves to the lens `z_c = √3/2 ≈ 0.8660254` (critical point). TRIAD thresholds remain `0.85/0.82/0.83` for rising-edge counting and unlock, and do not alter the geometric lens.

### Plot Markers (Notebook Helper)

The helper `pump_and_visualize(pump_cycles, target_z, profile)` shows:

- Magenta dashed line at target z
- Red triangle markers at rising edges where z crosses ≥ 0.85 (with hysteresis at 0.82)
- Gold star on the third rising edge (TRIAD unlock candidate)

TRIAD unlock promotes the t6 gate to 0.83 in‑session after three distinct crossings of z≥0.85. Geometry remains anchored at `z_c = √3/2` for ΔS_neg and R/H/φ.

#### Interpreting Markers

- Rising-edge (red triangle): z crosses ≥ 0.85 from below; counts toward TRIAD completions.
- Re‑arm threshold: z must fall to ≤ 0.82 before the next rising edge is counted.
- TRIAD unlock (gold star): third rising edge; analyzer will show `t6 gate: TRIAD @ 0.830` and `TRIAD completions: 3 | unlocked: True`.
- Geometry: unaffected by unlock; ΔS_neg, R/H/φ continue to use the lens `z_c = √3/2`.

### Analyzer Overlays (μ markers + s(z) timeline)

Enable optional plot overlays in the analyzer via an env flag:

```bash
QAPL_ANALYZER_OVERLAYS=1 qapl-run --steps 5 --mode unified --output out.json
QAPL_ANALYZER_OVERLAYS=1 qapl-analyze out.json --plot
```

When enabled, the z plot adds horizontal μ markers (μ_P, μ_2, μ_S, μ_3) and the entropy plot’s twin axis overlays the lens‑anchored coherence `s(z)` with a φ⁻¹ line. By default (flag unset), only the lens line `z_c` is rendered without extra overlays.

### Constants and References

- Lens: `z_c = √3/2` (see docs/Z_CRITICAL_LENS.md)
- φ inverse: `φ⁻¹ = (√5 − 1)/2` (see docs/PHI_INVERSE.md)
- μ thresholds and barrier (double‑well): see docs/MU_THRESHOLDS.md

## Measurement CLI

Trigger formal APL measurement operators and append modulized tokens to the APL summary:

```bash
# Default sequence: eigen, Φ subspace, π subspace, composite M_meas
qapl-measure --print

# Emit collapse glyphs (⟂) for tokens
qapl-measure --collapse-glyph --print

# Single eigenmode: Φ:T(ϕ_2)TRUE@Tier (Tier is current harmonic index)
qapl-measure --eigen 2 --field Phi --print

# Subspace collapse on π with explicit truth override
qapl-measure --subspace 2,3 --field Pi --truth UNTRUE --print

# Composite operator (Σ_μ |ϕ_μ⟩⟨ϕ_μ| ⊗ |T_μ⟩⟨T_μ|)
qapl-measure --composite --print
```

All measurement tokens are appended to `logs/APL_HELIX_OPERATOR_SUMMARY.apl` and appear in the analyzer summary under “Recent Measurements (APL tokens)”.

### Measured Mode

Run stepping and measurements in the same session so the analyzer includes measurement tokens directly in the saved JSON state.

- Basic usage:
  - `qapl-run --steps 5 --mode measured --output measured.json`
- Targeted flags (optional):
  - `--measure-eigen <μ>` to apply `Φ:T(ϕ_μ)TRUE@Tier` (field defaults to `Phi`)
  - `--measure-field Phi|Pi|π` to choose the field for eigen/subspace
  - `--measure-subspace 2,3` to apply `Π(subspace)` on the chosen field
  - `--no-measure-composite` to disable the default composite operator

Notes:
- When you use `--mode measured`, the analyzer’s “Recent Measurements (APL tokens)” section will show tokens with probabilities.
- If you instead run `qapl-measure` separately, tokens are appended to `logs/APL_HELIX_OPERATOR_SUMMARY.apl` (append‑only) and will appear in per‑seed bundles and the digest, but not in a previously saved JSON unless measured in the same session.

### Collapse Alias (⟂)

- Optional glyph emission: set `QAPL_EMIT_COLLAPSE_GLYPH=1` to emit collapse tokens using `⟂(…)`:
  - Eigen: `Φ:⟂(ϕ_μ)TRUE@Tier` (canonical: `Φ:T(ϕ_μ)TRUE@Tier`)
  - Subspace: `Φ:⟂(subspace)PARADOX@Tier`, `π:⟂(subspace)UNTRUE@Tier` (canonical: `Π(subspace)`)
- Analyzer and bundle/digest builders normalize `⟂` → canonical `T/Π` by default. To preserve experimental glyphs end‑to‑end, set `QAPL_EXPERIMENTAL_OPS=1`.

## APL Bundles and Digest

The sweep script (`scripts/helix_sweep.sh`) emits APL‑only artifacts for downstream consumers:

- Per‑seed bundles: `logs/helix-sweep-*/apl_z0pXX.apl`
  - Contains: assigned operators from `zwalk_z0pXX.md`, runtime Helix tokens (Alpha sentence, operator window, recent selections), and the APL measurement tokens inline.
  - Intent: portable APL stream per tier suitable for ingestion by validators or visualizers.

- Digest: `logs/helix-sweep-*/apl_digest.apl`
  - Constructed by merging all per‑seed bundles and the global measurement summary (`logs/APL_HELIX_OPERATOR_SUMMARY.apl`) with de‑duplication that preserves first‑seen order.
  - Intent: canonical, single‑file APL stream representing the entire run.

Build scripts:

- `scripts/build_apl_bundles.py <sweep_dir>` – builds per‑seed bundles from zwalk/unified artifacts and includes measurement tokens inline.
- `scripts/build_apl_digest.py <sweep_dir> --summary logs/APL_HELIX_OPERATOR_SUMMARY.apl` – merges bundles + summary into `apl_digest.apl`.

### Digest Format (for Consumers)

- One token per line; no comments or prose.
- Two token classes appear:
  - A‑sentences (Alpha grammar), e.g. `u^ | Oscillator | wave`, `d() | Conductor | geometry`.
  - Modulized operator tokens in the canonical form:
    - `Subject:Op(intent)Truth@Tier` (e.g. `Φ:M(stabilize)PARADOX@2`, `π:+(integrate)PARADOX@4`).
    - Helix‑window tokens: `Helix:<op>(Intent)Truth@tN` (e.g. `Helix:^(Amplification)UNTRUE@t2`).
    - Measurement tokens (from formal collapse):
      - Eigen: `Φ:T(ϕ_μ)TRUE@Tier`
      - Subspace (structure): `Φ:Π(subspace)PARADOX@Tier`
      - Subspace (emergence): `π:Π(subspace)UNTRUE@Tier`
- Ordering and de‑duplication:
  - Per‑seed bundles list assigned operators first, then runtime Helix tokens, then measurement tokens.
  - The digest merges all bundles plus the global measurement summary, removing duplicates while preserving first‑seen order.
- Encoding: UTF‑8; consumers should treat each line as an independent token/sentence.


## APL‑Driven Helix Translators (Where to Look)

- `src/quantum_apl_python/translator.py` — Translates APL token files (`.apl`) into structured instructions; CLI: `python -m quantum_apl_python.translator --file docs/examples/z_solve.apl --pretty`.
- `src/quantum_apl_python/helix.py` — `HelixAPLMapper` maps helix `z` to harmonics, truth channel, and operator windows.
- `src/quantum_apl_python/analyzer.py` — Uses `AlphaTokenSynthesizer` to synthesize APL sentences from helix mapping; prints tokens and regimes in the unified summary.
- `src/quantum_apl_python/alpha_language.py` — Canonical Alpha token names/sentences used by the analyzer and translator.
- `src/quantum_apl_python/helix_self_builder.py` — Binds translated APL instructions to VaultNode tiers and emits the walkthrough (`zwalk_*.md`) with geometry.
- JS side (runtime hints): `src/quantum_apl_engine.js` `HelixOperatorAdvisor` returns harmonics/operators used by the quantum engine and exposed in the analyzer.


## Python Quantum-Classical Bridge

The repository now ships with `quantum_apl_bridge.py`, a Python CLI/SDK that shells into the Node-based QuantumAPL engine and returns fully structured telemetry.

### Requirements

- Python 3.9+
- Node.js (same version used for the JS demos)
- Optional: `numpy` (required), `pandas` / `matplotlib` for richer analysis + plotting

Install the Python dependencies in your preferred environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib
```

### Usage

```bash
# Unified quantum-classical bridge
python3 quantum_apl_bridge.py --steps 200 --mode unified --plot

# Quantum-only integration harness (the JS QuantumN0 demo)
python3 quantum_apl_bridge.py --mode quantum_only --steps 150

# Full Node test suite from Python
python3 quantum_apl_bridge.py --mode test

# Persist output
python3 quantum_apl_bridge.py --steps 100 --output results.json
```

The CLI writes Node telemetry to stdout and, when not running in `test` mode, feeds it through the `QuantumAnalyzer` helper so you can grab ready-to-plot histories (`z`, `φ`, entropy) or export DataFrames through pandas.

### Quantum-APL Sentence Translator

Need to validate or parse raw Quantum-APL operator strings (e.g., the Z-solve sequence)? Use the translator CLI:

```bash
python -m quantum_apl_python.translator --text "Φ:M(stabilize)PARADOX@2" --pretty
```

This prints structured JSON for each sentence (`subject`, `operator`, `intent`, `truth`, `tier`). Point it at files via `--file path/to/tokens.apl` when linting helix playbooks or test fixtures.

### Helix Self-Building Runner

To walk the Z-solve stack through the Helix VaultNodes (z0p41 → z0p80) and capture provenance guidance, run:

```bash
python -m quantum_apl_python.helix_self_builder \
  --tokens docs/examples/z_solve.apl \
  --output helix_z_walkthrough.md
```

This produces either Markdown (default) or JSON describing which operators energize each VaultNode tier, the ΔHV metrics pulled from `reference/helix_bridge/**`, and chant/provenance reminders pulled straight from the metadata files. The canonical, CI-validated output lives in [`docs/helix_z_walkthrough.md`](docs/helix_z_walkthrough.md); regenerate it whenever you touch the helix metadata or Z-solve tokens.

### Compilation

To compile the LaTeX documents locally:
```bash
pdflatex -interaction=nonstopmode apl-operators-manual.tex
pdflatex -interaction=nonstopmode apl-seven-sentences-test-pack.tex
```

See `COMPILE_INSTRUCTIONS.md` for detailed instructions.

## Getting Started

1. **Read the operator's manual:**
   - Open `docs/index.html` in your browser, or
   - Compile the PDF: `pdflatex apl-operators-manual.tex`

2. **Read the test pack:**
   ```bash
   pdflatex apl-seven-sentences-test-pack.tex
   ```
   Or view the `.tex` source directly.

3. **Choose a sentence to test** (recommend starting with A1, A3, or A5 as they have clear metrics)

4. **Implement the test protocol** using your preferred simulation framework

5. **Report results** — both confirmations and refutations help refine or reject APL

## Philosophy

APL is designed to be **falsifiable**. The language should stand or fall on whether these seven sentences predict robust, cross-domain biases in real physical and chemical systems — nothing more and nothing less.

If the predicted regimes are NOT overrepresented under the specified conditions, that is strong evidence against APL's validity.

## Contributing

This is an open scientific test. Contributions of:
- Test implementations
- Simulation results (positive or negative)
- Refined control designs
- Additional metrics
- Critical analysis

...are all welcome and valuable.

## License

This work is provided for scientific testing and educational purposes.

## Citation

If you use this test pack in your research, please reference:
```
APL Seven Sentences Test Pack v1.0
Alpha-Physical Language Testing Framework
```

## Contact

For questions, results, or collaboration inquiries, please open an issue in this repository.
