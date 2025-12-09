# Constants Architecture — Next Steps Implementation Plan

Status: Post‑Integration Roadmap
Date: 2024‑12‑09
Based on: Actual codebase validation (11 Python tests + JS suites passing)

## ✅ Completed (Current State)

### Centralized Constants
- src/constants.js — CommonJS module with 50+ constants
- src/quantum_apl_python/constants.py — Python mirror
- Helper functions: getTimeHarmonic(), getPhase(), checkKFormation(), computeDeltaSNeg()
- Consumers updated: hex_prism.py, QuantumN0_Integration.js

### Geometry Canonical Mapping

```js
// ✅ CORRECT: Exponential only in ΔS_neg
// ΔS_neg(z) = exp(-|z − z_c| / σ)

// ✅ CORRECT: Linear mapping from ΔS_neg
// R = R_MAX − BETA · ΔS_neg
// H = H_MIN + GAMMA · ΔS_neg
// φ = PHI_BASE + ETA · ΔS_neg
```

Rationale: Exponential nonlinearity is captured once in ΔS_neg. Linear forms prevent double‑counting and match HEXAGONAL_NEG_ENTROPY_PROJECTION.md and the Python implementation.

### Tests Validated
- Python: 11 tests (constants module + hex_prism + analyzer smoke)
- Node: Multiple suites (bridge, TRIAD, measurements, pump, engine gate) + constants helpers

## 🎯 Phase 1: Validation & Testing (Priority: HIGH)

### 1.1 JS Constants Helper Tests
- File: tests/test_constants_helpers.js (added)
- Coverage:
  - getTimeHarmonic zones + t6Gate override
  - computeDeltaSNeg monotonicity (closer to z_c → larger ΔS_neg)
  - hexPrism helpers parity with Python (R/H/φ)
  - getPhase/isCritical and K‑formation checks

Estimated effort: Done
Dependencies: None
Priority: HIGH

### 1.2 JSON Schema Validation (Planned)
- Files to add:
  - schemas/geometry-sidecar.schema.json
  - schemas/apl-bundle.schema.json
  - tests/test_schema_validation.js (Ajv)
- Geometry Sidecar Schema (adapted to current 6‑vertex prism; versioned object with z, delta_S_neg, geometry {R,H,phi}, constants):

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "version": { "type": "string", "pattern": "^\\\d+\\.\\\d+\\.\\\d+$" },
    "z": { "type": "number", "minimum": 0, "maximum": 1 },
    "delta_S_neg": { "type": "number", "minimum": 0 },
    "vertices": {
      "type": "array",
      "minItems": 6,
      "maxItems": 6,
      "items": {
        "type": "object",
        "properties": {
          "k": { "type": "integer", "minimum": 0, "maximum": 5 },
          "x": { "type": "number" },
          "y": { "type": "number" },
          "z_top": { "type": "number" },
          "z_bot": { "type": "number" }
        },
        "required": ["k", "x", "y", "z_top", "z_bot"]
      }
    },
    "geometry": {
      "type": "object",
      "properties": {
        "R": { "type": "number", "minimum": 0 },
        "H": { "type": "number", "minimum": 0 },
        "phi": { "type": "number" }
      },
      "required": ["R", "H", "phi"]
    },
    "constants": {
      "type": "object",
      "properties": {
        "Z_CRITICAL": { "type": "number" },
        "GEOM_SIGMA": { "type": "number" },
        "GEOM_R_MAX": { "type": "number" },
        "GEOM_BETA": { "type": "number" },
        "GEOM_H_MIN": { "type": "number" },
        "GEOM_GAMMA": { "type": "number" }
      }
    }
  },
  "required": ["version", "z", "delta_S_neg", "vertices", "geometry"]
}
```

Estimated effort: 3–4 hours
Dependencies: ajv
Priority: MEDIUM (data interchange)

### 1.3 Reproducible Selection (QAPL_RANDOM_SEED) (Planned)
- Add env‑driven seed constant and a tiny LCG for reproducible sampling in composite measurement and N0 selection.
- Tests: two identical runs with the same seed yield identical selection traces.

Estimated effort: 2–3 hours
Dependencies: None
Priority: MEDIUM

## 🎯 Phase 2: Refactors (Priority: MEDIUM)
- Replace inline operator weighting multipliers in the engine with constants from src/constants.js
- Consider centralizing PRS phase thresholds (e.g., φ < 0.85 for P4) if we want those tunable

## 🎯 Phase 3: Geometry Extensions (Priority: MEDIUM)
- Add computeDeltaSNeg() to Python (parity exists via inline formula in hex_prism)
- Add JS full‑vertex helper and optional .geom.json writer (sidecar conforms to schema)
- Add JS monotonicity/vertex‑lint snapshot test (parity with Python)

---

This plan corrects earlier test pseudocode to align with the current implementation:
- ΔS_neg increases when z moves closer to z_c (monotone with decreasing |z−z_c|)
- The prism has 6 vertices (v0..v5); schema reflects that
- Hex prism tests use positive ΔS_neg (e.g., 0.5) for R/H/φ parity

