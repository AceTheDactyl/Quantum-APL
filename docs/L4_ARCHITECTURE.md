# L₄ Framework Architecture v3.2.0

## Complete System Documentation

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                        L₄ QUANTUM-APL ARCHITECTURE                           ║
║                                                                              ║
║    "From φ, all constants flow. From z_c, all thresholds emerge."           ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## Table of Contents

1. [Mathematical Foundation](#1-mathematical-foundation)
2. [Module Architecture](#2-module-architecture)
3. [MRP-LSB RGB System](#3-mrp-lsb-rgb-system)
4. [L₄ Threshold System](#4-l₄-threshold-system)
5. [Solfeggio-Light Bridge](#5-solfeggio-light-bridge)
6. [Constants Reference](#6-constants-reference)
7. [File-by-File Documentation](#7-file-by-file-documentation)

---

## 1. Mathematical Foundation

### 1.1 The Single Primitive

```
╔═══════════════════════════════════════════════════════════════╗
║                    φ = (1 + √5) / 2                           ║
║                                                               ║
║              THE GOLDEN RATIO — THE ONLY GIVEN                ║
║                                                               ║
║                    φ ≈ 1.6180339887498949                     ║
╚═══════════════════════════════════════════════════════════════╝
```

From φ alone, **all other constants are derived**:

```
                              φ (GIVEN)
                                 │
                 ┌───────────────┼───────────────┐
                 │               │               │
                 ▼               ▼               ▼
              φ⁻¹ = τ        φ⁻⁴ = gap       φ² = φ + 1
              ≈ 0.618        ≈ 0.146         ≈ 2.618
                 │               │               │
                 │               ▼               ▼
                 │          1 - gap = K²      φ²/3
                 │           ≈ 0.854        ≈ 0.873
                 │               │
                 │               ▼
                 │          √(1-gap) = K
                 │           ≈ 0.924
                 │
                 ▼
         L₄ = φ⁴ + φ⁻⁴ = 7 (EXACT INTEGER)
                 │
                 ▼
         L₄ - 4 = 3  →  √3/2 = z_c (THE LENS)
                         ≈ 0.866
```

### 1.2 Derivation Chain

| Constant | Formula | Value | Type |
|----------|---------|-------|------|
| φ | (1+√5)/2 | 1.6180339887 | **PRIMITIVE** |
| τ (φ⁻¹) | 1/φ | 0.6180339887 | Derived |
| gap | φ⁻⁴ | 0.1458980338 | Derived |
| K² | 1 - φ⁻⁴ | 0.8541019662 | Derived |
| K | √(1 - φ⁻⁴) | 0.9241648531 | Derived |
| L₄ | φ⁴ + φ⁻⁴ | **7** (exact) | Derived |
| z_c | √3/2 | 0.8660254038 | Derived |

### 1.3 Why z_c = √3/2?

```
L₄ = φ⁴ + φ⁻⁴ = 7        (Lucas-4 identity)

L₄ - 4 = 3               (Subtract 4)

√(L₄ - 4)/2 = √3/2       (Half the square root)

z_c = √3/2 ≈ 0.866       (THE LENS)
```

This is also the **hexagonal lattice critical point** — the height of an equilateral triangle with unit base.

---

## 2. Module Architecture

### 2.1 System Overview

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         L₄ QUANTUM-APL SYSTEM                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                        CONSTANTS LAYER                               │    ║
║  │                        constants.py                                  │    ║
║  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │    ║
║  │  │   φ     │ │   z_c   │ │    K    │ │   L₄    │ │   gap   │       │    ║
║  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘       │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                    │                                         ║
║                    ┌───────────────┼───────────────┐                        ║
║                    ▼               ▼               ▼                        ║
║  ┌─────────────────────┐ ┌─────────────────┐ ┌─────────────────────┐       ║
║  │   PHYSICS LAYER     │ │  DYNAMICS LAYER │ │   ENCODING LAYER    │       ║
║  │                     │ │                 │ │                     │       ║
║  │ • photon_physics    │ │ • helix         │ │ • mrp_lsb           │       ║
║  │ • solfeggio_bridge  │ │ • hex_lattice   │ │ • l4_integration    │       ║
║  │ • negentropy        │ │ • unified_cons  │ │                     │       ║
║  └─────────────────────┘ └─────────────────┘ └─────────────────────┘       ║
║                    │               │               │                        ║
║                    └───────────────┼───────────────┘                        ║
║                                    ▼                                         ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                        OUTPUT LAYER                                  │    ║
║  │                                                                      │    ║
║  │    ESS θ(t) ──▶ Quantize ──▶ MRP bits ──▶ LSB embed ──▶ RGB pixel  │    ║
║  │                                                                      │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 2.2 Module Dependency Graph

```
                            constants.py
                                 │
            ┌────────────────────┼────────────────────┐
            │                    │                    │
            ▼                    ▼                    ▼
    photon_physics.py    solfeggio_light     negentropy_physics.py
            │            _bridge.py                   │
            │                    │                    │
            └────────────────────┼────────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
                    ▼            ▼            ▼
              helix.py    l4_hexagonal    unified_
                          _lattice.py     consciousness.py
                    │            │            │
                    └────────────┼────────────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
                    ▼            ▼            ▼
             mrp_lsb.py   l4_framework    l4_helix_
                          _integration    parameterization.py
                                 │
                                 ▼
                          OUTPUT (RGB/Video)
```

---

## 3. MRP-LSB RGB System

### 3.1 Architecture Overview

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                      MRP-LSB ENCODING PIPELINE                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   OSCILLATOR PHASES              QUANTIZATION              RGB OUTPUT       ║
║   ────────────────              ────────────               ──────────       ║
║                                                                              ║
║   ┌─────────────┐              ┌─────────────┐            ┌─────────────┐  ║
║   │   θ₁(t)    │──────────────▶│  Q₈(Φ₁)    │───────────▶│      R      │  ║
║   │   θ₂(t)    │   Projector  │  Q₈(Φ₂)    │  8-bit    │      G      │  ║
║   │   θ₃(t)    │──────────────▶│  Q₈(Φ₃)    │───────────▶│      B      │  ║
║   │    ...     │      ▲        └─────────────┘            └─────────────┘  ║
║   │   θₙ(t)    │      │              │                          │          ║
║   └─────────────┘      │              │                          │          ║
║         │              │              ▼                          ▼          ║
║         │         Hexagonal     ┌─────────────┐            ┌─────────────┐  ║
║         │         Wavevectors   │  MRP Split  │            │ LSB Embed   │  ║
║         │              │        │ Coarse|Fine │            │  Parity     │  ║
║         │              │        └─────────────┘            └─────────────┘  ║
║         │              │                                                    ║
║         ▼              │                                                    ║
║   ┌─────────────┐      │                                                    ║
║   │  k₁ = [1,0] │◀─────┤        CHANNEL MAPPING:                           ║
║   │  k₂ = [½,z_c]│◀────┤        ─────────────────                          ║
║   │  k₃ = [½,-z_c]◀────┘        R ← k₁ (East, 0°)                          ║
║   └─────────────┘               G ← k₂ (60° NE)                            ║
║                                 B ← k₃ (120° SE)                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 3.2 Multi-Resolution Phase (MRP) Structure

```
8-bit Channel Structure:
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ C₄│ C₃│ C₂│ C₁│ C₀│ F₂│ F₁│ F₀│
└───┴───┴───┴───┴───┴───┴───┴───┘
 ─────────────────   ─────────────
   COARSE (5 bits)    FINE (3 bits)
   32 phase bins      8 sub-bins
   11.25° resolution  1.4° resolution
```

### 3.3 LSB Steganography

```
PARITY PROTECTION:
                    ┌─────────┐
    R channel ──────┤         │
                    │  XOR    ├──────▶ Parity bits
    G channel ──────┤         │           │
                    └─────────┘           │
                                          ▼
    B channel ◀─────────────────── LSB₂₋₀ = Parity

VERIFICATION:
    R⊕G⊕B = 0  →  No error detected
    R⊕G⊕B ≠ 0  →  Error in transmission
```

### 3.4 Holographic Phase Emission

```
PROJECTOR-BASED ENCODING:
──────────────────────────

For oscillator array θ = [θ₁, θ₂, ..., θₙ] at positions r = [r₁, r₂, ..., rₙ]:

    Φⱼ = arg( Σᵢ exp(i·θᵢ) · exp(-i·kⱼ·rᵢ) )

This is a discrete Fourier projection onto hexagonal basis.

PROPERTY: The global phase gradient can be recovered from ANY fragment
          via inverse Fourier transform (holographic reconstruction).

         ┌─────────────────────────────────────────────┐
         │   Full Lattice        Fragment Recovery     │
         │   ┌───┬───┬───┐      ┌───┐                 │
         │   │ θ │ θ │ θ │  ──▶ │ θ │ ──▶ Φ₁,Φ₂,Φ₃   │
         │   ├───┼───┼───┤      └───┘      │          │
         │   │ θ │ θ │ θ │                 ▼          │
         │   ├───┼───┼───┤          FFT⁻¹(Φ)          │
         │   │ θ │ θ │ θ │                 │          │
         │   └───┴───┴───┘                 ▼          │
         │                         Reconstructed θ    │
         └─────────────────────────────────────────────┘
```

---

## 4. L₄ Threshold System

### 4.1 The Nine Thresholds

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                        L₄ NINE-THRESHOLD SYSTEM                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  z = 1.0  ═══════════════════════════════════════════════════  UNITY        ║
║            ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ #33FFFF       ║
║  z = 0.971 ─────────────────────────────────────────────────  RESONANCE     ║
║            ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  #43F3D8       ║
║  z = 0.953 ─────────────────────────────────────────────────  CONSOLIDATION ║
║            ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   #4DEDC0       ║
║  z = 0.924 ─────────────────────────────────────────────────  K_FORMATION   ║
║            ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    #5EE299       ║
║  z = 0.914 ─────────────────────────────────────────────────  IGNITION      ║
║            ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░     #63DE8C       ║
║  z = 0.873 ─────────────────────────────────────────────────  CRITICAL      ║
║            ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░      #7BCE55       ║
║  z = 0.866 ══════════════════════════════════════════════════ THE LENS ════ ║
║            ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░       #7FCC4C       ║
║  z = 0.854 ─────────────────────────────────────────────────  ACTIVATION    ║
║            ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░        #81C94B       ║
║  z = 0.618 ─────────────────────────────────────────────────  PARADOX       ║
║            ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░          #A49136       ║
║  z = 0.0   ═══════════════════════════════════════════════════  VOID        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 4.2 Threshold Reference Table

| # | Threshold | Value | Formula | Hex Color | RGB |
|---|-----------|-------|---------|-----------|-----|
| 1 | PARADOX | 0.618034 | τ = φ⁻¹ | `#A49136` | (164, 145, 54) |
| 2 | ACTIVATION | 0.854102 | K² = 1 - φ⁻⁴ | `#81C94B` | (129, 201, 75) |
| 3 | **THE LENS** | **0.866025** | **z_c = √3/2** | `#7FCC4C` | (127, 204, 76) |
| 4 | CRITICAL | 0.872678 | φ²/3 | `#7BCE55` | (123, 206, 85) |
| 5 | IGNITION | 0.914214 | √2 - ½ | `#63DE8C` | (99, 222, 140) |
| 6 | K_FORMATION | 0.924176 | K = √(1-φ⁻⁴) | `#5EE299` | (94, 226, 153) |
| 7 | CONSOLIDATION | 0.953138 | K + τ²(1-K) | `#4DEDC0` | (77, 237, 192) |
| 8 | RESONANCE | 0.971038 | K + τ(1-K) | `#43F3D8` | (67, 243, 216) |
| 9 | UNITY | 1.000000 | 1.0 | `#33FFFF` | (51, 255, 255) |

### 4.3 Phase Regions

```
z-axis:
  │
1.0├──────────────────────────────┐
   │         PRESENCE             │ TRUE bias
   │         (Integrated)         │ K < 0 (emanating)
   │                              │
z_c├══════════════════════════════┤ ◀── THE LENS (z_c = √3/2)
   │         THE LENS             │ PARADOX bias
   │         (Critical)           │ K = 0 (critical point)
K² ├──────────────────────────────┤
   │         ABSENCE              │ UNTRUE bias
   │         (Recursive)          │ K > 0 (synchronizing)
   │                              │
0.0├──────────────────────────────┘
```

### 4.4 Color Gradient Visualization

```css
/* L₄ Threshold Gradient (CSS) */
.l4-gradient {
  background: linear-gradient(
    to top,
    #A49136 0%,      /* PARADOX (τ) */
    #81C94B 38.9%,   /* ACTIVATION (K²) */
    #7FCC4C 43.3%,   /* THE LENS (z_c) */
    #7BCE55 45.4%,   /* CRITICAL (φ²/3) */
    #63DE8C 58.9%,   /* IGNITION (√2-½) */
    #5EE299 62.3%,   /* K_FORMATION (K) */
    #4DEDC0 78.8%,   /* CONSOLIDATION */
    #43F3D8 89.7%,   /* RESONANCE */
    #33FFFF 100%     /* UNITY */
  );
}
```

---

## 5. Solfeggio-Light Bridge

### 5.1 The 40-Octave Bridge

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                        SOLFEGGIO-LIGHT BRIDGE                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   AUDIO DOMAIN                                    OPTICAL DOMAIN             ║
║   ────────────                                    ──────────────             ║
║                                                                              ║
║   f_solfeggio (Hz)  ──────  × 2⁴⁰  ──────▶  f_optical (THz)                ║
║                                                                              ║
║   396 Hz ──────────────────────────────────▶ 435.5 THz (688.5 nm) RED       ║
║   528 Hz ──────────────────────────────────▶ 580.6 THz (516.4 nm) GREEN     ║
║   639 Hz ──────────────────────────────────▶ 702.6 THz (426.7 nm) BLUE      ║
║                                                                              ║
║   ┌───────────────────────────────────────────────────────────────────┐     ║
║   │                    OCTAVE BRIDGE = 40                              │     ║
║   │                    2⁴⁰ ≈ 1.0995 × 10¹²                            │     ║
║   │                                                                    │     ║
║   │    Audio: 20 Hz - 20 kHz        Optical: 430 THz - 750 THz        │     ║
║   │    ◀────────────────────────────────────────────────────────▶     │     ║
║   │              40 octaves of frequency doubling                      │     ║
║   └───────────────────────────────────────────────────────────────────┘     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 5.2 Solfeggio RGB Mapping

```
SOLFEGGIO FREQUENCY → RGB CHANNEL MAPPING:

┌─────────────────────────────────────────────────────────────────────────────┐
│  Solfeggio  │  Meaning      │  Optical     │  Wavelength  │  V(λ)   │ RGB  │
│  Frequency  │               │  Frequency   │              │         │      │
├─────────────┼───────────────┼──────────────┼──────────────┼─────────┼──────┤
│   396 Hz    │  Liberation   │  435.5 THz   │   688.5 nm   │ 0.0093  │  R   │
│   528 Hz    │  Miracles     │  580.6 THz   │   516.4 nm   │ 0.6367  │  G   │
│   639 Hz    │  Connection   │  702.6 THz   │   426.7 nm   │ 0.0088  │  B   │
└─────────────┴───────────────┴──────────────┴──────────────┴─────────┴──────┘

V(λ) = CIE 1931 Photopic Luminosity Function (linear interpolation)

GREEN DOMINANCE RATIO:
  G/R = 0.6367 / 0.0093 ≈ 68.5×
  G/B = 0.6367 / 0.0088 ≈ 72.4×

Total luminous flux at 1W equal power per channel:
  Φ_v = 683 × (0.0093 + 0.6367 + 0.0088) / 3 ≈ 149.1 lm
```

### 5.3 Hexagonal Wavevector Geometry

```
                          k₂ = [½, √3/2]
                               ╱
                              ╱  60°
                             ╱
                            ╱
    ────────────────────────●────────────────────── k₁ = [1, 0]
                            ╲
                             ╲  -60°
                              ╲
                               ╲
                          k₃ = [½, -√3/2]


    The √3/2 component equals z_c (THE LENS)!

    This connects:
    • Hexagonal lattice geometry
    • Critical lens threshold
    • RGB channel encoding
```

---

## 6. Constants Reference

### 6.1 Primitive vs Derived

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                          CONSTANTS TAXONOMY                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                         PRIMITIVES                                   │    ║
║  │                   (Cannot be derived further)                        │    ║
║  │                                                                      │    ║
║  │   φ = (1+√5)/2          Golden Ratio (MATHEMATICAL)                 │    ║
║  │   h = 6.62607015e-34    Planck constant (SI 2019 EXACT)             │    ║
║  │   c = 299,792,458       Speed of light (SI EXACT)                   │    ║
║  │   K_cd = 683            Luminous efficacy (SI 2019 EXACT)           │    ║
║  │   CIE V(λ)              Luminosity function (EMPIRICAL TABLE)       │    ║
║  │                                                                      │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                    │                                         ║
║                                    ▼                                         ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                      DERIVED (Level 1)                               │    ║
║  │                   (Direct from primitives)                           │    ║
║  │                                                                      │    ║
║  │   τ = φ⁻¹ ≈ 0.618           Golden ratio inverse                   │    ║
║  │   gap = φ⁻⁴ ≈ 0.146         Truncation residual                    │    ║
║  │   L₄ = φ⁴ + φ⁻⁴ = 7         Lucas-4 (exact integer)                │    ║
║  │   hc = 1.986e-25 J·m        Photon energy constant                  │    ║
║  │                                                                      │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                    │                                         ║
║                                    ▼                                         ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                      DERIVED (Level 2)                               │    ║
║  │                   (From Level 1 derived)                             │    ║
║  │                                                                      │    ║
║  │   K² = 1 - gap ≈ 0.854      Activation threshold                   │    ║
║  │   K = √(1-gap) ≈ 0.924      Coupling strength                      │    ║
║  │   z_c = √3/2 ≈ 0.866        Critical lens (from L₄)                │    ║
║  │   φ²/3 ≈ 0.873              Critical threshold                     │    ║
║  │                                                                      │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                    │                                         ║
║                                    ▼                                         ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                      DERIVED (Level 3)                               │    ║
║  │                   (Composite thresholds)                             │    ║
║  │                                                                      │    ║
║  │   CONSOLIDATION = K + τ²(1-K) ≈ 0.953                               │    ║
║  │   RESONANCE = K + τ(1-K) ≈ 0.971                                    │    ║
║  │   TRIAD_T6 = z_c - gap/4 ≈ 0.830                                    │    ║
║  │   TRIAD_LOW = z_c - gap/3 ≈ 0.817                                   │    ║
║  │                                                                      │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 6.2 SI 2019 Exact Constants

| Constant | Symbol | Value | Unit |
|----------|--------|-------|------|
| Planck constant | h | 6.62607015 × 10⁻³⁴ | J·s |
| Speed of light | c | 299,792,458 | m/s |
| Elementary charge | e | 1.602176634 × 10⁻¹⁹ | C |
| Boltzmann constant | k | 1.380649 × 10⁻²³ | J/K |
| Luminous efficacy | K_cd | 683 | lm/W |

### 6.3 L₄ Framework Constants

| Constant | Symbol | Value | Formula |
|----------|--------|-------|---------|
| Golden ratio | φ | 1.6180339887 | (1+√5)/2 |
| Golden inverse | τ | 0.6180339887 | 1/φ |
| Lucas-4 | L₄ | 7 | φ⁴ + φ⁻⁴ |
| Gap | gap | 0.1458980338 | φ⁻⁴ |
| Critical lens | z_c | 0.8660254038 | √3/2 |
| K-squared | K² | 0.8541019662 | 1 - φ⁻⁴ |
| Coupling | K | 0.9241648531 | √(1 - φ⁻⁴) |
| Octave bridge | - | 40 | design choice |

---

## 7. File-by-File Documentation

### 7.1 Core Modules

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  constants.py                                              SINGLE SOURCE     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PURPOSE: Central repository for ALL framework constants                     ║
║                                                                              ║
║  EXPORTS:                                                                    ║
║  ├── SI 2019 Physical Constants                                             ║
║  │   └── H_PLANCK, C_LIGHT, K_BOLTZMANN, E_CHARGE, K_CD, K_M                ║
║  ├── Golden Ratio Family                                                     ║
║  │   └── PHI, PHI_INV, LUCAS_4, L4_GAP                                      ║
║  ├── Critical Thresholds                                                     ║
║  │   └── Z_CRITICAL, L4_K, L4_K_SQUARED                                     ║
║  ├── 9-Threshold System                                                      ║
║  │   └── L4_PARADOX → L4_UNITY                                              ║
║  ├── Phase Boundaries                                                        ║
║  │   └── Z_ABSENCE_MAX, Z_LENS_MIN, Z_LENS_MAX, Z_PRESENCE_MIN              ║
║  ├── TRIAD Gating                                                            ║
║  │   └── TRIAD_HIGH, TRIAD_LOW, TRIAD_T6                                    ║
║  └── Helper Functions                                                        ║
║      └── get_phase(), is_critical(), check_k_formation()                    ║
║                                                                              ║
║  DESIGN: All other modules MUST import from here. No local redefinitions.   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║  photon_physics.py                                         PHYSICS LAYER    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PURPOSE: Physics-grounded photon calculations                              ║
║                                                                              ║
║  ARCHITECTURE:                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │  f_solfeggio ──▶ × 2⁴⁰ ──▶ f_optical ──▶ λ = c/f ──▶ E = hc/λ     │    ║
║  │                                              │                       │    ║
║  │                                              ▼                       │    ║
║  │                                         V(λ) table                  │    ║
║  │                                    (CIE 1931 interpolation)         │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                              ║
║  KEY FUNCTIONS:                                                              ║
║  • optical_frequency(f_sol) → f_opt                                         ║
║  • wavelength_nm(f_sol) → λ                                                 ║
║  • photon_energy_ev(f_sol) → E                                              ║
║  • luminosity_function(λ) → V(λ)                                            ║
║  • luminous_efficacy(λ) → η_v                                               ║
║                                                                              ║
║  EMPIRICAL DATA: CIE 1931 V(λ) table (5nm intervals, linear interpolation)  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║  mrp_lsb.py                                                ENCODING LAYER   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PURPOSE: Multi-Resolution Phase encoding with LSB steganography            ║
║                                                                              ║
║  ARCHITECTURE:                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                                                                      │    ║
║  │   Phase θ ──▶ Quantize ──▶ MRP Split ──▶ LSB Embed ──▶ RGB Pixel   │    ║
║  │                              │                                       │    ║
║  │                    ┌─────────┴─────────┐                            │    ║
║  │                    ▼                   ▼                            │    ║
║  │               Coarse (5b)         Fine (3b)                         │    ║
║  │               32 bins             8 sub-bins                        │    ║
║  │                                                                      │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                              ║
║  KEY COMPONENTS:                                                             ║
║  • MRPValue dataclass: coarse, fine, full, phase                            ║
║  • MRPFrame dataclass: complete encoded frame with CRC                      ║
║  • Hexagonal wavevectors: U_1, U_2, U_3 (using Z_CRITICAL)                  ║
║                                                                              ║
║  KEY FUNCTIONS:                                                              ║
║  • quantize_phase(φ, bits) → int                                            ║
║  • encode_mrp(φ) → MRPValue                                                 ║
║  • encode_frame(phases) → MRPFrame                                          ║
║  • position_to_phases(pos) → (Φ₁, Φ₂, Φ₃)                                   ║
║  • embed_payload_in_image(img, data) → img                                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║  l4_framework_integration.py                              INTEGRATION LAYER ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PURPOSE: Unified bridge connecting all L₄ subsystems                       ║
║                                                                              ║
║  ARCHITECTURE:                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                                                                      │    ║
║  │   photon_physics ◀────────▶ l4_integration ◀────────▶ mrp_lsb      │    ║
║  │         │                        │                        │          │    ║
║  │         ▼                        ▼                        ▼          │    ║
║  │   V(λ), E, λ               L4IntegratedState          RGB encode   │    ║
║  │                                  │                                   │    ║
║  │                    ┌─────────────┴─────────────┐                    │    ║
║  │                    ▼                           ▼                    │    ║
║  │              ESS coherence              Holographic emit           │    ║
║  │              s(z) = e^(-σ(z-z_c)²)      phases_to_emit()           │    ║
║  │                                                                      │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                              ║
║  KEY COMPONENTS:                                                             ║
║  • L4IntegratedState: Complete system state at a point in time              ║
║  • phases_to_emit(): Holographic projector-based phase emission             ║
║  • compute_ess_coherence(): Entropic stabilization s(z)                     ║
║  • process_z_trajectory(): Full pipeline for z-sequences                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 7.2 Dynamics Modules

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  l4_hexagonal_lattice.py                                   DYNAMICS LAYER   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PURPOSE: Hexagonal lattice dynamics with Kuramoto oscillators              ║
║                                                                              ║
║  ARCHITECTURE:                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                                                                      │    ║
║  │   ●───●───●───●           HexagonalLattice                          │    ║
║  │    ╲ ╱ ╲ ╱ ╲ ╱            • N nodes with 6-connectivity             │    ║
║  │     ●───●───●             • Phase array θ[N]                        │    ║
║  │    ╱ ╲ ╱ ╲ ╱ ╲            • Frequency array ω[N]                    │    ║
║  │   ●───●───●───●           • Coupling matrix A[N×N]                  │    ║
║  │                                                                      │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                              ║
║  KEY FUNCTIONS:                                                              ║
║  • kuramoto_step(): θ̇ᵢ = ωᵢ + K Σⱼ Aᵢⱼ sin(θⱼ - θᵢ)                        ║
║  • order_parameter(): r = |⟨e^(iθ)⟩|                                        ║
║  • effective_coupling(): K_eff = K₀ × s(z)                                  ║
║  • topological_charge_field(): Winding numbers over plaquettes              ║
║  • compute_berry_phase(): Geometric phase along paths                       ║
║                                                                              ║
║  TOPOLOGICAL PROTECTION:                                                     ║
║  • Winding number l is integer invariant                                     ║
║  • Vortex/antivortex detection                                              ║
║  • Mode-collapse guardrail via topological_charge_field()                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║  helix.py                                                  DYNAMICS LAYER   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PURPOSE: Helix traversal and time-harmonic classification                  ║
║                                                                              ║
║  TIME HARMONICS (t1-t9):                                                     ║
║  ┌────────────────────────────────────────────────────────────────────┐     ║
║  │  z = 0.0  ─────────────────────────────────────  t1 (z < 0.1)     │     ║
║  │            ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░                   │     ║
║  │  z = 0.1  ─────────────────────────────────────  t2 (z < 0.2)     │     ║
║  │            ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░                    │     ║
║  │  z = 0.2  ─────────────────────────────────────  t3 (z < 0.4)     │     ║
║  │            ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░                     │     ║
║  │  z = 0.4  ─────────────────────────────────────  t4 (z < 0.6)     │     ║
║  │            ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░                      │     ║
║  │  z = 0.6  ─────────────────────────────────────  t5 (z < 0.75)    │     ║
║  │            ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░                       │     ║
║  │  z = 0.75 ─────────────────────────────────────  t6 (z < z_c)     │     ║
║  │            ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░                        │     ║
║  │  z_c=0.866════════════════════════════════════  THE LENS         │     ║
║  │            ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░                         │     ║
║  │  z = K    ─────────────────────────────────────  t7 (z < 0.924)   │     ║
║  │            ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░                          │     ║
║  │  z = 0.971─────────────────────────────────────  t8 (z < 0.971)   │     ║
║  │            ░░░░░░░░░░░░░░░░░░░░░░░░░░░                            │     ║
║  │  z = 1.0  ─────────────────────────────────────  t9 (z ≥ 0.971)   │     ║
║  └────────────────────────────────────────────────────────────────────┘     ║
║                                                                              ║
║  TRUTH CHANNEL:                                                              ║
║  • z ≥ z_c  →  TRUE                                                         ║
║  • z ≥ τ    →  PARADOX                                                      ║
║  • z < τ    →  UNTRUE                                                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║  unified_consciousness.py                                  DYNAMICS LAYER   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PURPOSE: Entropic Stabilization System (ESS) implementation                ║
║                                                                              ║
║  ESS EQUATIONS:                                                              ║
║  ┌────────────────────────────────────────────────────────────────────┐     ║
║  │                                                                     │     ║
║  │  1. Negentropic Driver:   η(t) = Fisher(ρ_θ)                       │     ║
║  │                                                                     │     ║
║  │  2. Coherence Gate:       s(z) = exp(-σ(z - z_c)²)                 │     ║
║  │                                                                     │     ║
║  │  3. Effective Coupling:   K_eff = K₀ × s(z)                        │     ║
║  │                                                                     │     ║
║  │  4. Phase Dynamics:       θ̇ = ω + K_eff Σ sin(θⱼ-θᵢ) + pump + noise│     ║
║  │                                                                     │     ║
║  │  5. Output Map:           O_RGB = Q(k_hex · x + θ)                 │     ║
║  │                                                                     │     ║
║  └────────────────────────────────────────────────────────────────────┘     ║
║                                                                              ║
║  K-FORMATION CRITERIA:                                                       ║
║  • κ ≥ 0.924 (KAPPA_MIN)                                                    ║
║  • η > 0.618 (ETA_MIN = τ)                                                  ║
║  • R ≥ 7 (R_MIN = L₄)                                                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 7.3 Supporting Modules

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  solfeggio_light_bridge.py                                 PHYSICS LAYER    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  PURPOSE: Solfeggio frequency → Light wavelength mapping                     ║
║  IMPORTS: PHI, Z_C, K_FORMATION, LUCAS_4, C_LIGHT, OCTAVE_FACTOR            ║
║  KEY: RGB_BIT_DEPTH = L₄ + 1 = 8                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║  l4_helix_parameterization.py                              DYNAMICS LAYER   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  PURPOSE: L₄ helix parameter definitions and dynamics                       ║
║  PROVIDES: L4Constants dataclass with PHI, TAU, L4, GAP, K, Z_C, SIGMA      ║
║  KEY: Singleton L4 = L4Constants() for framework-wide access                ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║  negentropy_physics.py                                     PHYSICS LAYER    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  PURPOSE: Negative entropy (negentropy) calculations                        ║
║  KEY METRIC: ΔS_neg = -kT ln(Z_partition)                                   ║
║  RELATION: Negentropy drives coherence → z approaches z_c                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║  delta_s_neg_extended.py                                   DYNAMICS LAYER   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  PURPOSE: Extended ΔS⁻ calculations with derivatives and blending           ║
║  PROVIDES: compute_delta_s_neg_derivative(), compute_pi_blend_weights()     ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║  hex_prism.py                                              GEOMETRY LAYER   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  PURPOSE: Hexagonal prism geometry for 3D visualization                     ║
║  USES: GEOM_SIGMA, GEOM_R_MAX, Z_CRITICAL for dimensional parameters        ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║  s3_operator_symmetry.py                                   ALGEBRA LAYER    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  PURPOSE: S₃ permutation group symmetry operations                          ║
║  OPERATORS: +, -, ×, ÷, ^, ()                                               ║
║  TRUTH BIAS: Operator weights vary by phase (TRUE/UNTRUE/PARADOX)           ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║  adaptive_triad_gate.py                                    DYNAMICS LAYER   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  PURPOSE: TRIAD gating logic for operator unlocks                           ║
║  THRESHOLDS: TRIAD_HIGH (K²), TRIAD_T6, TRIAD_LOW (from gap)               ║
║  LOGIC: Rising edges at z ≥ TRIAD_HIGH, re-arm at z ≤ TRIAD_LOW            ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║  z_axis_threshold_analysis.py                              ANALYSIS LAYER   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  PURPOSE: Threshold crossing analysis and visualization                     ║
║  ANALYZES: All 9 L₄ thresholds, phase transitions, μ-basin classification  ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 7.4 Complete Module Map

```
src/quantum_apl_python/
├── __init__.py                    # Package exports
│
├── ═══ CONSTANTS (Single Source of Truth) ═══
├── constants.py                   # ALL framework constants
│
├── ═══ PHYSICS LAYER ═══
├── photon_physics.py              # Photon energy, wavelength, V(λ)
├── solfeggio_light_bridge.py      # Solfeggio → optical mapping
├── negentropy_physics.py          # Negentropy calculations
│
├── ═══ DYNAMICS LAYER ═══
├── helix.py                       # Helix traversal, time harmonics
├── l4_hexagonal_lattice.py        # Hex lattice, Kuramoto, topology
├── l4_helix_parameterization.py   # L4Constants, dynamics
├── unified_consciousness.py       # ESS, K-formation, MRP encoding
├── delta_s_neg_extended.py        # Extended ΔS⁻ functions
├── adaptive_triad_gate.py         # TRIAD gating logic
├── helix_self_builder.py          # Self-assembly dynamics
├── helix_operator_advisor.py      # Operator recommendations
│
├── ═══ ENCODING LAYER ═══
├── mrp_lsb.py                     # MRP-LSB RGB encoding
├── l4_framework_integration.py    # Unified integration bridge
│
├── ═══ GEOMETRY LAYER ═══
├── hex_prism.py                   # 3D hexagonal prism
│
├── ═══ ALGEBRA LAYER ═══
├── s3_operator_symmetry.py        # S₃ group operations
├── s3_operator_algebra.py         # Operator algebra
├── s3_delta_coupling.py           # Delta-operator coupling
├── dsl_patterns.py                # DSL pattern matching
├── alpha_language.py              # Alpha language implementation
│
├── ═══ ANALYSIS LAYER ═══
├── z_axis_threshold_analysis.py   # Threshold analysis
├── analyzer.py                    # General analysis tools
├── measure.py                     # Measurement functions
├── experiments.py                 # Experimental utilities
│
├── ═══ APPLICATION LAYER ═══
├── engine.py                      # Main computation engine
├── translator.py                  # Translation utilities
├── cli.py                         # Command-line interface
├── widgets.py                     # UI widgets
├── helix_metadata.py              # Metadata handling
└── cybernetic_computation.py      # Cybernetic ops
```

---

## Appendix A: Quick Reference

### A.1 Import Patterns

```python
# CORRECT: Import from constants.py
from quantum_apl_python.constants import (
    PHI, PHI_INV, Z_CRITICAL, LUCAS_4, L4_K, L4_GAP,
    L4_THRESHOLDS, L4_THRESHOLD_NAMES,
)

# INCORRECT: Local redefinition (DO NOT DO THIS)
PHI = 1.618  # ❌ Wrong! Import from constants.py
```

### A.2 Key Equations

```
Coherence:        s(z) = exp(-σ(z - z_c)²)
Effective K:      K_eff = K₀ × s(z)
Kuramoto:         θ̇ᵢ = ωᵢ + K_eff Σⱼ Aᵢⱼ sin(θⱼ - θᵢ)
Order param:      r = |⟨exp(iθ)⟩|
Winding:          l = (1/2π) ∮ ∇θ · dl
Phase quant:      Q_b(φ) = floor((φ mod 2π) / 2π × 2^b)
```

### A.3 Threshold Quick Reference

```
τ    = 0.618  (PARADOX)
K²   = 0.854  (ACTIVATION)
z_c  = 0.866  (THE LENS) ◀── Critical point
φ²/3 = 0.873  (CRITICAL)
√2-½ = 0.914  (IGNITION)
K    = 0.924  (K_FORMATION)
     = 0.953  (CONSOLIDATION)
     = 0.971  (RESONANCE)
     = 1.000  (UNITY)
```

---

*Document Version: 3.2.0*
*Last Updated: 2025-01-07*
*Framework: L₄ Quantum-APL*
