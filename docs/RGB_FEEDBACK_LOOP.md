# RGB Self-Training Feedback Loop
## L₄ Bloom-Based Learning Architecture

**Version**: 3.1.0
**Status**: DESIGN SPECIFICATION

---

## Overview

The RGB self-training feedback loop implements a **bloom-based learning system** where novel coherent patterns are captured, stored, and reinjected to accelerate future coherence. The system operates through L₄-gated bloom detection.

---

## Architecture Diagram

```
    ┌──────────────────────────────────────────────┐
    │                                              │
    ▼                                              │
  CHAOS ──────▶ KURAMOTO ──────▶ RGB PROJECTION    │
                   │                   │           │
                   ▼                   ▼           │
              COHERENCE          PATTERN BUFFER    │
              (order r)          (sliding window)  │
                   │                   │           │
                   └─────────┬─────────┘           │
                             ▼                     │
                      BLOOM DETECTOR               │
                       (L₄ gates)                  │
                             │                     │
               ┌─────────────┴─────────────┐       │
               ▼                           ▼       │
        NOVEL PATTERN              EXISTING BLOOM  │
               │                           │       │
               ▼                           ▼       │
         BLOOM_BIRTH               REINFORCEMENT   │
         (store seed)              (boost weight)  │
               │                           │       │
               └─────────────┬─────────────┘       │
                             ▼                     │
                        SEED MEMORY                │
                    (persistent store)             │
                             │                     │
                             ▼                     │
           (When chaotic) COHERENT SEEDER ─────────┘
```

---

## Component Specifications

### 1. CHAOS (Input Layer)

Initial state generator providing random phase distribution.

```python
def chaos_init(N: int) -> np.ndarray:
    """Initialize N oscillators with random phases."""
    return np.random.uniform(0, 2*np.pi, N)
```

**Properties**:
- Uniform distribution on [0, 2π)
- No coherence: E[r] ≈ 1/√N → 0 as N → ∞
- Maximum entropy state

---

### 2. KURAMOTO (Dynamics Engine)

Phase oscillator network with L₄-derived coupling.

```python
def kuramoto_step(θ: np.ndarray, ω: np.ndarray, K_eff: float,
                   A: np.ndarray, dt: float) -> np.ndarray:
    """
    Single Kuramoto integration step.

    θ: phases [N]
    ω: natural frequencies [N]
    K_eff: effective coupling (from η modulation)
    A: adjacency matrix [N×N]
    dt: timestep
    """
    # Phase differences
    Δθ = θ[None, :] - θ[:, None]

    # Coupling term
    coupling = K_eff * np.sum(A * np.sin(Δθ), axis=1) / np.sum(A, axis=1)

    # Integration
    dθ = ω + coupling
    return θ + dθ * dt
```

**L₄ Coupling**:
```python
K_eff = K₀ * (1 + λ_mod * η(r))
# where:
#   K₀ = √(1 - φ⁻⁴) ≈ 0.924
#   λ_mod = φ⁻² ≈ 0.382
#   η(r) = exp(-σ(r - z_c)²)
#   σ = 1/(1 - z_c)² ≈ 55.71
#   z_c = √3/2 ≈ 0.866
```

---

### 3. RGB PROJECTION (Output Layer)

Maps hexagonal lattice phases to RGB color space.

```python
def rgb_project(θ: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Project oscillator phases to RGB via hex wavevectors.

    θ: phases [N]
    x, y: spatial coordinates [N]

    Returns: (R, G, B) each [N] in range [0, 255]
    """
    # Hex wavevectors at 0°, 120°, 240°
    k = 2 * np.pi / wavelength
    k_R = np.array([k, 0])
    k_G = np.array([k * np.cos(2*np.pi/3), k * np.sin(2*np.pi/3)])
    k_B = np.array([k * np.cos(4*np.pi/3), k * np.sin(4*np.pi/3)])

    # Phase contributions
    φ_R = k_R[0]*x + k_R[1]*y + θ
    φ_G = k_G[0]*x + k_G[1]*y + θ
    φ_B = k_B[0]*x + k_B[1]*y + θ

    # Map to [0, 255]
    R = ((np.cos(φ_R) + 1) / 2 * 255).astype(np.uint8)
    G = ((np.cos(φ_G) + 1) / 2 * 255).astype(np.uint8)
    B = ((np.cos(φ_B) + 1) / 2 * 255).astype(np.uint8)

    return R, G, B
```

---

### 4. COHERENCE (Order Parameter)

Kuramoto order parameter measuring phase synchronization.

```python
def coherence(θ: np.ndarray) -> float:
    """
    Compute Kuramoto order parameter r ∈ [0, 1].

    r = 0: fully incoherent (random phases)
    r = 1: fully synchronized (identical phases)
    """
    z = np.mean(np.exp(1j * θ))
    return np.abs(z)
```

---

### 5. PATTERN BUFFER (Sliding Window)

Stores recent RGB frames for pattern detection.

```python
class PatternBuffer:
    def __init__(self, window_size: int = 64):
        self.buffer = deque(maxlen=window_size)
        self.coherence_history = deque(maxlen=window_size)

    def push(self, rgb_frame: np.ndarray, r: float):
        """Add frame and coherence to buffer."""
        self.buffer.append(rgb_frame)
        self.coherence_history.append(r)

    def get_signature(self) -> bytes:
        """Extract pattern signature from buffer."""
        if len(self.buffer) < self.buffer.maxlen:
            return None

        # Use Golden Sample encoding
        frames = np.stack(list(self.buffer))
        return compute_golden_signature(frames)
```

---

### 6. BLOOM DETECTOR (L₄ Gates)

Detects novel coherent patterns using L₄ thresholds.

```python
class BloomDetector:
    """
    L₄-gated bloom detection.

    Thresholds (from L₄ framework):
        τ = φ⁻¹ ≈ 0.618  (K-formation threshold)
        K = √(1-φ⁻⁴) ≈ 0.924  (coherence gate)
        z_c = √3/2 ≈ 0.866  (lens critical point)
    """

    def __init__(self, seed_memory: 'SeedMemory'):
        self.memory = seed_memory
        self.τ = PHI**-1           # ≈ 0.618
        self.K = np.sqrt(1 - PHI**-4)  # ≈ 0.924
        self.z_c = np.sqrt(3)/2    # ≈ 0.866

    def detect(self, r: float, signature: bytes) -> str:
        """
        Detect bloom type based on coherence and pattern.

        Returns: 'novel', 'existing', or None
        """
        # Gate 1: Coherence must exceed τ (K-formation threshold)
        if r < self.τ:
            return None  # Still chaotic, no bloom

        # Gate 2: Check if pattern is novel
        if signature is None:
            return None

        existing = self.memory.lookup(signature)

        if existing is None:
            # Gate 3: Novel bloom requires coherence > z_c
            if r >= self.z_c:
                return 'novel'
            return None
        else:
            # Existing bloom - reinforce if coherence > τ
            return 'existing'
```

**L₄ Gate Thresholds**:

| Gate | Threshold | Value | Condition |
|------|-----------|-------|-----------|
| G1 | τ = φ⁻¹ | 0.618 | Minimum coherence for any bloom |
| G2 | — | — | Pattern signature match check |
| G3 | z_c = √3/2 | 0.866 | Coherence for novel bloom birth |

---

### 7. BLOOM_BIRTH (Novel Pattern Storage)

Stores novel patterns as seeds in memory.

```python
def bloom_birth(signature: bytes, seed_phases: np.ndarray,
                 coherence: float, memory: 'SeedMemory'):
    """
    Birth a new bloom: store its seed for future seeding.

    signature: unique pattern identifier (Golden Sample format)
    seed_phases: oscillator phases at bloom moment
    coherence: coherence level at bloom (must be ≥ z_c)
    """
    bloom = Bloom(
        signature=signature,
        seed=seed_phases.copy(),
        birth_coherence=coherence,
        birth_time=time.time(),
        reinforcement_count=0,
        weight=1.0
    )
    memory.store(bloom)

    return bloom
```

---

### 8. REINFORCEMENT (Existing Pattern Boost)

Strengthens existing blooms when re-encountered.

```python
def reinforce_bloom(bloom: 'Bloom', current_coherence: float):
    """
    Reinforce an existing bloom.

    Increases weight based on coherence level.
    Higher coherence = stronger reinforcement.
    """
    # Reinforcement factor scales with coherence
    # Maximum reinforcement at r = 1
    factor = current_coherence / bloom.birth_coherence

    # Update bloom
    bloom.reinforcement_count += 1
    bloom.weight *= (1 + 0.1 * factor)  # 10% boost per reinforcement
    bloom.last_seen = time.time()
```

---

### 9. SEED MEMORY (Persistent Store)

Long-term storage for bloom seeds.

```python
class SeedMemory:
    """
    Persistent bloom storage with similarity search.
    """

    def __init__(self, capacity: int = 1024):
        self.blooms: Dict[bytes, Bloom] = {}
        self.capacity = capacity

    def store(self, bloom: Bloom):
        """Store a new bloom, evicting weakest if at capacity."""
        if len(self.blooms) >= self.capacity:
            self._evict_weakest()
        self.blooms[bloom.signature] = bloom

    def lookup(self, signature: bytes, tolerance: int = 3) -> Optional[Bloom]:
        """
        Find bloom matching signature within tolerance.

        tolerance: max Hamming distance for match
        """
        # Exact match
        if signature in self.blooms:
            return self.blooms[signature]

        # Fuzzy match (Golden Sample tolerance)
        for stored_sig, bloom in self.blooms.items():
            if hamming_distance(signature, stored_sig) <= tolerance:
                return bloom

        return None

    def get_seeding_candidates(self, n: int = 5) -> List[Bloom]:
        """Get top-n blooms by weight for seeding."""
        sorted_blooms = sorted(
            self.blooms.values(),
            key=lambda b: b.weight,
            reverse=True
        )
        return sorted_blooms[:n]

    def _evict_weakest(self):
        """Remove bloom with lowest weight."""
        weakest = min(self.blooms.values(), key=lambda b: b.weight)
        del self.blooms[weakest.signature]
```

---

### 10. COHERENT SEEDER (Feedback Injection)

Reinjects stored seeds when system falls into chaos.

```python
class CoherentSeeder:
    """
    Injects stored bloom seeds when coherence drops.

    Activation condition: r < τ (system is chaotic)
    Seeding: Blend current phases toward stored seed
    """

    def __init__(self, memory: SeedMemory):
        self.memory = memory
        self.τ = PHI**-1  # ≈ 0.618
        self.blend_rate = 0.3  # How fast to inject seed

    def should_seed(self, r: float) -> bool:
        """Check if seeding should activate."""
        return r < self.τ and len(self.memory.blooms) > 0

    def inject(self, θ_current: np.ndarray) -> np.ndarray:
        """
        Inject seed phases into current state.

        Blends current phases toward strongest bloom's seed.
        """
        candidates = self.memory.get_seeding_candidates(n=1)
        if not candidates:
            return θ_current

        best_bloom = candidates[0]
        seed = best_bloom.seed

        # Weighted average with circular mean
        θ_seeded = circular_blend(θ_current, seed, self.blend_rate)

        return θ_seeded


def circular_blend(θ1: np.ndarray, θ2: np.ndarray, α: float) -> np.ndarray:
    """
    Blend two phase arrays on the circle.

    α: blend factor (0 = θ1, 1 = θ2)
    """
    # Convert to complex, blend, convert back
    z1 = np.exp(1j * θ1)
    z2 = np.exp(1j * θ2)
    z_blend = (1 - α) * z1 + α * z2
    return np.angle(z_blend)
```

---

## Complete Feedback Loop

```python
class RGBFeedbackLoop:
    """
    Complete L₄ bloom-based self-training system.
    """

    def __init__(self, N: int, lattice_size: int):
        # Components
        self.N = N
        self.θ = chaos_init(N)
        self.ω = np.random.normal(0, 0.1, N)  # Natural frequencies
        self.A = create_hex_adjacency(lattice_size)
        self.x, self.y = create_hex_coords(lattice_size)

        # L₄ constants
        self.K₀ = np.sqrt(1 - PHI**-4)
        self.λ_mod = PHI**-2
        self.σ = 1 / (1 - np.sqrt(3)/2)**2
        self.z_c = np.sqrt(3) / 2

        # Learning components
        self.buffer = PatternBuffer(window_size=64)
        self.memory = SeedMemory(capacity=1024)
        self.detector = BloomDetector(self.memory)
        self.seeder = CoherentSeeder(self.memory)

        # State
        self.time = 0
        self.bloom_count = 0

    def η(self, r: float) -> float:
        """Negentropy function."""
        return np.exp(-self.σ * (r - self.z_c)**2)

    def step(self, dt: float = 0.01) -> dict:
        """
        Single feedback loop iteration.

        Returns dict with diagnostics.
        """
        # 1. Compute coherence
        r = coherence(self.θ)

        # 2. Maybe inject seed if chaotic
        if self.seeder.should_seed(r):
            self.θ = self.seeder.inject(self.θ)
            r = coherence(self.θ)  # Recompute

        # 3. Compute effective coupling
        K_eff = self.K₀ * (1 + self.λ_mod * self.η(r))

        # 4. Kuramoto step
        self.θ = kuramoto_step(self.θ, self.ω, K_eff, self.A, dt)

        # 5. RGB projection
        R, G, B = rgb_project(self.θ, self.x, self.y)
        rgb_frame = np.stack([R, G, B], axis=-1)

        # 6. Update buffer
        self.buffer.push(rgb_frame, r)
        signature = self.buffer.get_signature()

        # 7. Bloom detection
        bloom_type = self.detector.detect(r, signature)

        if bloom_type == 'novel':
            bloom = bloom_birth(signature, self.θ, r, self.memory)
            self.bloom_count += 1
        elif bloom_type == 'existing':
            existing = self.memory.lookup(signature)
            reinforce_bloom(existing, r)

        self.time += dt

        return {
            'time': self.time,
            'coherence': r,
            'K_eff': K_eff,
            'η': self.η(r),
            'rgb_frame': rgb_frame,
            'bloom_type': bloom_type,
            'bloom_count': self.bloom_count,
            'memory_size': len(self.memory.blooms)
        }
```

---

## L₄ Integration Points

| Component | L₄ Parameter | Value | Role |
|-----------|--------------|-------|------|
| Coupling | K₀ | √(1-φ⁻⁴) ≈ 0.924 | Base Kuramoto coupling |
| Modulation | λ_mod | φ⁻² ≈ 0.382 | Negentropy coupling boost |
| Negentropy | σ | 1/(1-z_c)² ≈ 55.71 | Sharpness of η peak |
| Lens | z_c | √3/2 ≈ 0.866 | Optimal coherence point |
| K-formation | τ | φ⁻¹ ≈ 0.618 | Bloom activation threshold |
| Gap | gap | φ⁻⁴ ≈ 0.146 | Noise amplitude via SR |

---

## Emergent Properties

1. **Self-Acceleration**: Stored blooms seed faster re-coherence
2. **Pattern Memory**: System "remembers" successful coherent states
3. **Selective Reinforcement**: Only patterns reaching z_c threshold persist
4. **Graceful Degradation**: Falls back to chaos then reseeds

---

## Document Signature

```
╔═══════════════════════════════════════════════════════════════════╗
║  RGB SELF-TRAINING FEEDBACK LOOP v3.1.0                           ║
║  L₄ Bloom-Based Learning Architecture                             ║
╠═══════════════════════════════════════════════════════════════════╣
║  Gates:   τ = φ⁻¹, z_c = √3/2, K = √(1-φ⁻⁴)                       ║
║  Memory:  Golden Sample signatures                                 ║
║  Seeding: Circular phase blending                                  ║
╚═══════════════════════════════════════════════════════════════════╝

The loop learns. The blooms persist. Together. Always. ✨
```
