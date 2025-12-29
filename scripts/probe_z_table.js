#!/usr/bin/env node
/*
Probe z values and print a Markdown table with:
- s(z) = ΔS_neg(z) using LENS_SIGMA
- φGateScale(s)
- K-formation PASS/FAIL based on η=s vs φ⁻¹
- time harmonic (t1..t9)
- truth channel
- recommended operators (by harmonic)

Usage:
  PROBE_ZS="0.41 0.52 0.70 0.80 0.854 0.866 0.873 0.914 0.924 0.971" node scripts/probe_z_table.js
*/

const path = require('path');
const CONST = require(path.join(__dirname, '..', 'src', 'constants'));
const { QuantumAPL } = require(path.join(__dirname, '..', 'src', 'quantum_apl_engine'));

function parseList(envStr) {
  if (!envStr) return null;
  return envStr
    .split(/[\s,]+/)
    .map(s => s.trim())
    .filter(Boolean)
    .map(Number)
    .filter(n => Number.isFinite(n));
}

// L₄-Helix 9-threshold aligned probe points:
// VaultNode tiers + ACTIVATION, THE LENS, CRITICAL, IGNITION, K-FORMATION, RESONANCE
const defaultZs = [0.41, 0.52, 0.70, 0.73, 0.80, CONST.L4_ACTIVATION, CONST.Z_CRITICAL,
                   CONST.L4_CRITICAL, CONST.L4_IGNITION, CONST.L4_K_FORMATION, CONST.L4_RESONANCE];
const zs = parseList(process.env.PROBE_ZS) || defaultZs;

const engine = new QuantumAPL();

function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}

function row(z) {
  const zc = clamp01(z);
  const s = CONST.computeDeltaSNeg(zc, CONST.LENS_SIGMA);
  const gate = CONST.phiGateScale(s);
  const pass = s > CONST.PHI_INV ? 'PASS' : 'FAIL';
  const hints = engine.helixAdvisor.describe(zc);
  const harmonic = hints.harmonic;
  const truth = hints.truthChannel;
  const ops = Array.isArray(hints.operators) ? hints.operators.join(', ') : '';
  return `| ${zc.toFixed(6)} | ${s.toFixed(3)} | ${gate.toFixed(3)} | ${pass} | ${harmonic} | ${truth} | ${ops} |`;
}

console.log('| z | s(z) | φGateScale(s) | K-formation | harmonic | truth | operators |');
console.log('|---|------|----------------|------------|----------|-------|-----------|');
zs.forEach(z => console.log(row(z)));

