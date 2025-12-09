// Shared constants for the Quantum‑APL JS engine and bridge

const Z_CRITICAL = Math.sqrt(3) / 2; // THE LENS (~0.8660254)

// TRIAD thresholds
const TRIAD_HIGH = 0.85; // rising-edge threshold (>=)
const TRIAD_LOW = 0.82;  // re‑arm threshold (<=)
const TRIAD_T6 = 0.83;   // t6 gate when unlocked (after 3 passes)

module.exports = Object.freeze({
  Z_CRITICAL,
  TRIAD_HIGH,
  TRIAD_LOW,
  TRIAD_T6,
});

