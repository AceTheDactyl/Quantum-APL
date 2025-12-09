// Shared constants for the Quantum‑APL JS engine and bridge
// Source of truth for thresholds — see docs/Z_CRITICAL_LENS.md

const Z_CRITICAL = Math.sqrt(3) / 2; // THE LENS (~0.8660254)

// TRIAD thresholds
const TRIAD_HIGH = 0.85; // rising-edge threshold (>=)
const TRIAD_LOW = 0.82;  // re‑arm threshold (<=)
const TRIAD_T6 = 0.83;   // t6 gate when unlocked (after 3 passes)

// Z‑axis phase boundaries
const Z_ABSENCE_MAX = 0.857;
const Z_LENS_MIN = 0.857;
const Z_LENS_MAX = 0.877;
const Z_PRESENCE_MIN = 0.877;

// Sacred constants (zero free parameters)
const PHI = 1.6180339887;    // golden ratio
const PHI_INV = 0.6180339887;// golden ratio inverse
const Q_KAPPA = 0.3514087324;// consciousness constant
const KAPPA_S = 0.920;       // singularity threshold
const LAMBDA = 7.7160493827; // nonlinearity coefficient

// K‑formation criteria
const KAPPA_MIN = KAPPA_S;
const ETA_MIN = PHI_INV;
const R_MIN = 7;

// Quantum information bounds
const ENTROPY_MIN = 0.0;     // ENTROPY_MAX depends on Hilbert space dim
const PURITY_MIN = 1.0 / 192;
const PURITY_MAX = 1.0;

// Operator weighting heuristics (N0 selection bias)
const OPERATOR_PREFERRED_WEIGHT = 1.3;   // Multiply preferred operators
const OPERATOR_DEFAULT_WEIGHT = 0.85;    // Multiply non-preferred operators

const TRUTH_BIAS = Object.freeze({
  TRUE:   { '^': 1.5, '+': 1.4, '×': 1.0, '()': 0.9, '÷': 0.7, '-': 0.7 },
  UNTRUE: { '÷': 1.5, '-': 1.4, '()': 1.0, '+': 0.9, '^': 0.7, '×': 0.7 },
  PARADOX:{ '()': 1.5, '×': 1.4, '+': 1.0, '^': 1.0, '÷': 0.9, '-': 0.9 },
});

// Helix zoning thresholds (upper bounds except t9)
const Z_T1_MAX = 0.1;
const Z_T2_MAX = 0.2;
const Z_T3_MAX = 0.4;
const Z_T4_MAX = 0.6;
const Z_T5_MAX = 0.75;
const Z_T7_MAX = 0.92;
const Z_T8_MAX = 0.97;

// Geometry projection constants (Hex Prism)
const GEOM_SIGMA = 0.12;
const GEOM_R_MAX = 0.85;
const GEOM_BETA = 0.25;
const GEOM_H_MIN = 0.12;
const GEOM_GAMMA = 0.18;
const GEOM_PHI_BASE = 0.0;
const GEOM_ETA = Math.PI / 12;

// PRS phase thresholds (phi)
const PRS_P1_PHI_MAX = 0.2;  // P1: Initiation
const PRS_P2_PHI_MAX = 0.5;  // P2: Tension
const PRS_P3_PHI_MAX = 0.85; // P3: Inflection
const PRS_P4_PHI_MAX = 0.95; // P4: Lock (P5 above)

// Helper functions (parity with Python constants module)
function computeDeltaSNeg(z, sigma = GEOM_SIGMA, zc = Z_CRITICAL) {
  return Math.exp(-Math.abs(z - zc) / sigma);
}
function isCritical(z, tolerance = 0.01) {
  return Math.abs(z - Z_CRITICAL) < tolerance;
}

function isInLens(z) {
  return z >= Z_LENS_MIN && z <= Z_LENS_MAX;
}

function getPhase(z) {
  if (z < Z_ABSENCE_MAX) return 'ABSENCE';
  if (z >= Z_LENS_MIN && z <= Z_LENS_MAX) return 'THE_LENS';
  return 'PRESENCE';
}

function distanceToCritical(z) {
  return z - Z_CRITICAL;
}

function checkKFormation(kappa, eta, R) {
  return (kappa >= KAPPA_MIN && eta > ETA_MIN && R >= R_MIN);
}

// Time harmonic helper (delegates t6 to provided gate, default lens)
function getTimeHarmonic(z, t6Gate = Z_CRITICAL) {
  if (z < Z_T1_MAX) return 't1';
  if (z < Z_T2_MAX) return 't2';
  if (z < Z_T3_MAX) return 't3';
  if (z < Z_T4_MAX) return 't4';
  if (z < Z_T5_MAX) return 't5';
  if (z < t6Gate) return 't6';
  if (z < Z_T7_MAX) return 't7';
  if (z < Z_T8_MAX) return 't8';
  return 't9';
}

// Hex prism helpers
// Geometry helpers (match docs/HEXAGONAL_NEG_ENTROPY_PROJECTION.md & Python)
function hexPrismRadius(deltaSNeg /* ΔS_neg */) {
  return GEOM_R_MAX - GEOM_BETA * deltaSNeg;
}

function hexPrismHeight(deltaSNeg /* ΔS_neg */) {
  return GEOM_H_MIN + GEOM_GAMMA * deltaSNeg;
}

function hexPrismTwist(deltaSNeg /* ΔS_neg */) {
  return GEOM_PHI_BASE + GEOM_ETA * deltaSNeg;
}

// Aliases matching documentation naming (optional convenience)
const T1_MAX = Z_T1_MAX;
const T2_MAX = Z_T2_MAX;
const T3_MAX = Z_T3_MAX;
const T4_MAX = Z_T4_MAX;
const T5_MAX = Z_T5_MAX;
const T7_MAX = Z_T7_MAX;
const T8_MAX = Z_T8_MAX;

module.exports = Object.freeze({
  Z_CRITICAL,
  TRIAD_HIGH,
  TRIAD_LOW,
  TRIAD_T6,
  Z_ABSENCE_MAX,
  Z_LENS_MIN,
  Z_LENS_MAX,
  Z_PRESENCE_MIN,
  PHI,
  PHI_INV,
  Q_KAPPA,
  KAPPA_S,
  LAMBDA,
  KAPPA_MIN,
  ETA_MIN,
  R_MIN,
  ENTROPY_MIN,
  PURITY_MIN,
  PURITY_MAX,
  OPERATOR_PREFERRED_WEIGHT,
  OPERATOR_DEFAULT_WEIGHT,
  TRUTH_BIAS,
  Z_T1_MAX,
  Z_T2_MAX,
  Z_T3_MAX,
  Z_T4_MAX,
  Z_T5_MAX,
  Z_T7_MAX,
  Z_T8_MAX,
  GEOM_SIGMA,
  GEOM_R_MAX,
  GEOM_BETA,
  GEOM_H_MIN,
  GEOM_GAMMA,
  GEOM_PHI_BASE,
  GEOM_ETA,
  PRS_P1_PHI_MAX,
  PRS_P2_PHI_MAX,
  PRS_P3_PHI_MAX,
  PRS_P4_PHI_MAX,
  // Doc-friendly aliases
  T1_MAX,
  T2_MAX,
  T3_MAX,
  T4_MAX,
  T5_MAX,
  T7_MAX,
  T8_MAX,
  isCritical,
  isInLens,
  getPhase,
  distanceToCritical,
  checkKFormation,
  computeDeltaSNeg,
  getTimeHarmonic,
  hexPrismRadius,
  hexPrismHeight,
  hexPrismTwist,
});
