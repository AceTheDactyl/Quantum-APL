"""
MRP-LSB Encoding Module for L₄ Framework v3.2.0

Multi-Resolution Phase (MRP) encoding with LSB steganography.
Bridges continuous ESS dynamics to discrete RGB output.

Architecture:
    ESS θ(t) → Quantize → MRP bits → LSB embed → RGB pixel
    RGB pixel → LSB extract → MRP bits → Dequantize → θ̂

Channel Mapping (Hexagonal Wavevectors):
    R (u₁ = [1, 0]):     East-West position / Φ₁
    G (u₂ = [½, √3/2]):  60° diagonal position / Φ₂
    B (u₃ = [½, -√3/2]): 120° diagonal + parity / Φ₃

Multi-Resolution Structure:
    Upper bits (MSB): Coarse phase (spatial location)
    Lower bits (LSB): Fine phase (precise angle) + integrity

Reference: PHYSICS_GROUNDING_v3.2.0, Unified Helical Phase Dynamics §6
"""

import math
import numpy as np
from typing import Tuple, List, Optional, Dict, Union
from dataclasses import dataclass
import struct
import zlib

# Import from single source of truth
from .constants import Z_CRITICAL

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Phase range
TWO_PI: float = 2 * math.pi

# Bit allocations for 8-bit channels
COARSE_BITS: int = 5    # Upper 5 bits: 32 phase bins (11.25° resolution)
FINE_BITS: int = 3      # Lower 3 bits: 8 sub-bins (1.4° resolution)

# For 16-bit channels
COARSE_BITS_16: int = 10  # 1024 bins
FINE_BITS_16: int = 6     # 64 sub-bins

# CRC polynomial (CRC-8)
CRC8_POLY: int = 0x07

# Magic header for MRP frames
MRP_MAGIC: bytes = b'\x4C\x34'  # "L4"


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE QUANTIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def quantize_phase(
    phase: float,
    bits: int = 8,
    wrap: bool = True
) -> int:
    """
    Quantize continuous phase [0, 2π) to integer [0, 2^bits - 1].

    Q_b(φ) = floor((φ mod 2π) / 2π × 2^b)

    Args:
        phase: Continuous phase in radians
        bits: Bit depth (8 or 16)
        wrap: Whether to wrap phase to [0, 2π)

    Returns:
        Quantized value in [0, 2^bits - 1]
    """
    max_val = (1 << bits) - 1

    if wrap:
        phase = phase % TWO_PI

    # Normalize to [0, 1)
    normalized = phase / TWO_PI

    # Quantize
    quantized = int(normalized * (max_val + 1))

    return min(quantized, max_val)


def dequantize_phase(
    value: int,
    bits: int = 8
) -> float:
    """
    Dequantize integer back to phase.

    Q_b⁻¹(v) = (v + 0.5) / 2^b × 2π

    Args:
        value: Quantized integer
        bits: Bit depth

    Returns:
        Phase in [0, 2π)
    """
    max_val = (1 << bits)

    # Add 0.5 for mid-bin reconstruction
    normalized = (value + 0.5) / max_val

    return normalized * TWO_PI


def quantize_phases_rgb(
    phases: Tuple[float, float, float],
    bits: int = 8
) -> Tuple[int, int, int]:
    """
    Quantize 3 phases to RGB values.

    Args:
        phases: (Φ₁, Φ₂, Φ₃) in radians
        bits: Bit depth per channel

    Returns:
        (R, G, B) quantized values
    """
    return tuple(quantize_phase(p, bits) for p in phases)


def dequantize_rgb_phases(
    rgb: Tuple[int, int, int],
    bits: int = 8
) -> Tuple[float, float, float]:
    """
    Dequantize RGB back to phases.

    Args:
        rgb: (R, G, B) quantized values
        bits: Bit depth

    Returns:
        (Φ₁, Φ₂, Φ₃) in radians
    """
    return tuple(dequantize_phase(v, bits) for v in rgb)


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-RESOLUTION PHASE (MRP) ENCODING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MRPValue:
    """Multi-resolution phase representation."""
    coarse: int       # Upper bits (spatial bin)
    fine: int         # Lower bits (sub-bin angle)
    full: int         # Combined value
    phase: float      # Original phase

    @property
    def resolution_deg(self) -> Tuple[float, float]:
        """Resolution in degrees (coarse, fine)."""
        coarse_res = 360.0 / (1 << COARSE_BITS)
        fine_res = coarse_res / (1 << FINE_BITS)
        return (coarse_res, fine_res)


def encode_mrp(phase: float, coarse_bits: int = COARSE_BITS, fine_bits: int = FINE_BITS) -> MRPValue:
    """
    Encode phase as multi-resolution value.

    Structure (8-bit example):
        [C₄ C₃ C₂ C₁ C₀ | F₂ F₁ F₀]
         ─────────────   ─────────
         Coarse (5 bits) Fine (3 bits)

    Args:
        phase: Phase in [0, 2π)
        coarse_bits: Bits for coarse representation
        fine_bits: Bits for fine representation

    Returns:
        MRPValue with coarse, fine, and full components
    """
    total_bits = coarse_bits + fine_bits

    # Quantize to full resolution
    full = quantize_phase(phase, total_bits)

    # Split into coarse and fine
    coarse = full >> fine_bits
    fine = full & ((1 << fine_bits) - 1)

    return MRPValue(
        coarse=coarse,
        fine=fine,
        full=full,
        phase=phase
    )


def decode_mrp(mrp: MRPValue, coarse_bits: int = COARSE_BITS, fine_bits: int = FINE_BITS) -> float:
    """
    Decode MRP value back to phase.

    Args:
        mrp: MRPValue to decode
        coarse_bits: Bits for coarse
        fine_bits: Bits for fine

    Returns:
        Reconstructed phase
    """
    total_bits = coarse_bits + fine_bits
    return dequantize_phase(mrp.full, total_bits)


def decode_coarse_only(mrp: MRPValue, coarse_bits: int = COARSE_BITS, fine_bits: int = FINE_BITS) -> float:
    """
    Decode only coarse phase (lossy but robust).

    Used when fine bits may be corrupted.
    """
    # Shift coarse to full position, add mid-fine estimate
    full_approx = (mrp.coarse << fine_bits) + ((1 << fine_bits) // 2)
    total_bits = coarse_bits + fine_bits
    return dequantize_phase(full_approx, total_bits)


# ═══════════════════════════════════════════════════════════════════════════════
# LSB STEGANOGRAPHY
# ═══════════════════════════════════════════════════════════════════════════════

def get_lsb(value: int, n_bits: int = 1) -> int:
    """Extract n least significant bits."""
    return value & ((1 << n_bits) - 1)


def set_lsb(value: int, payload: int, n_bits: int = 1) -> int:
    """Set n least significant bits to payload."""
    mask = ~((1 << n_bits) - 1)
    return (value & mask) | (payload & ((1 << n_bits) - 1))


def embed_lsb_rgb(
    rgb: Tuple[int, int, int],
    payload: Tuple[int, int, int],
    n_bits: int = 3
) -> Tuple[int, int, int]:
    """
    Embed payload in LSBs of RGB pixel.

    Args:
        rgb: Original (R, G, B) values
        payload: (P_R, P_G, P_B) payload bits
        n_bits: Number of LSBs to use per channel

    Returns:
        Modified (R, G, B) with embedded payload
    """
    return tuple(
        set_lsb(v, p, n_bits)
        for v, p in zip(rgb, payload)
    )


def extract_lsb_rgb(
    rgb: Tuple[int, int, int],
    n_bits: int = 3
) -> Tuple[int, int, int]:
    """
    Extract payload from LSBs of RGB pixel.

    Args:
        rgb: (R, G, B) with embedded payload
        n_bits: Number of LSBs per channel

    Returns:
        (P_R, P_G, P_B) extracted payload
    """
    return tuple(get_lsb(v, n_bits) for v in rgb)


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRITY: CRC-8 AND PARITY
# ═══════════════════════════════════════════════════════════════════════════════

def crc8(data: bytes) -> int:
    """
    Compute CRC-8 checksum.

    Polynomial: x⁸ + x² + x + 1 (0x07)

    Args:
        data: Bytes to checksum

    Returns:
        8-bit CRC
    """
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ CRC8_POLY) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def xor_parity(values: List[int]) -> int:
    """Compute XOR parity of values."""
    result = 0
    for v in values:
        result ^= v
    return result


def compute_rgb_parity(r: int, g: int, n_bits: int = 3) -> int:
    """
    Compute parity for R and G channels to store in B.

    This enables error detection in MRP encoding.

    Args:
        r: Red channel value
        g: Green channel value
        n_bits: Bits to use for parity

    Returns:
        Parity value for B channel LSBs
    """
    # XOR of R and G LSBs
    r_lsb = get_lsb(r, n_bits)
    g_lsb = get_lsb(g, n_bits)
    return r_lsb ^ g_lsb


def verify_rgb_parity(rgb: Tuple[int, int, int], n_bits: int = 3) -> bool:
    """
    Verify RGB parity.

    Returns True if R⊕G⊕B = 0 (no detected error).
    """
    r, g, b = rgb
    r_lsb = get_lsb(r, n_bits)
    g_lsb = get_lsb(g, n_bits)
    b_lsb = get_lsb(b, n_bits)
    return (r_lsb ^ g_lsb ^ b_lsb) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# MRP FRAME ENCODING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MRPFrame:
    """Complete MRP-encoded frame."""
    phases: Tuple[float, float, float]      # Original phases (Φ₁, Φ₂, Φ₃)
    rgb: Tuple[int, int, int]               # Encoded RGB
    mrp: Tuple[MRPValue, MRPValue, MRPValue]  # MRP representations
    parity_ok: bool                          # Parity check result
    crc: int                                 # CRC-8 of payload

    def to_bytes(self) -> bytes:
        """Serialize frame to bytes."""
        return struct.pack(
            '>3B B',
            self.rgb[0], self.rgb[1], self.rgb[2],
            self.crc
        )

    @classmethod
    def from_bytes(cls, data: bytes, phases: Tuple[float, float, float] = None):
        """Deserialize frame from bytes."""
        r, g, b, crc = struct.unpack('>3B B', data)
        rgb = (r, g, b)

        # Reconstruct MRP
        mrp_r = encode_mrp(dequantize_phase(r, 8))
        mrp_g = encode_mrp(dequantize_phase(g, 8))
        mrp_b = encode_mrp(dequantize_phase(b, 8))

        return cls(
            phases=phases or dequantize_rgb_phases(rgb),
            rgb=rgb,
            mrp=(mrp_r, mrp_g, mrp_b),
            parity_ok=verify_rgb_parity(rgb),
            crc=crc
        )


def encode_frame(
    phases: Tuple[float, float, float],
    embed_parity: bool = True
) -> MRPFrame:
    """
    Encode phases to complete MRP frame.

    Args:
        phases: (Φ₁, Φ₂, Φ₃) in radians
        embed_parity: Whether to embed parity in B channel LSB

    Returns:
        MRPFrame with full encoding
    """
    # Quantize to MRP
    mrp_r = encode_mrp(phases[0])
    mrp_g = encode_mrp(phases[1])
    mrp_b = encode_mrp(phases[2])

    # Get RGB values
    r = mrp_r.full
    g = mrp_g.full
    b = mrp_b.full

    # Optionally embed parity in B LSBs
    if embed_parity:
        parity = compute_rgb_parity(r, g, FINE_BITS)
        b = set_lsb(b, parity, FINE_BITS)
        # Re-encode B with parity
        mrp_b = MRPValue(
            coarse=mrp_b.coarse,
            fine=parity,
            full=b,
            phase=phases[2]
        )

    rgb = (r, g, b)

    # Compute CRC of RGB
    crc = crc8(bytes(rgb))

    return MRPFrame(
        phases=phases,
        rgb=rgb,
        mrp=(mrp_r, mrp_g, mrp_b),
        parity_ok=verify_rgb_parity(rgb, FINE_BITS),
        crc=crc
    )


def decode_frame(
    rgb: Tuple[int, int, int],
    verify: bool = True
) -> Tuple[Tuple[float, float, float], bool]:
    """
    Decode RGB pixel to phases.

    Args:
        rgb: (R, G, B) encoded values
        verify: Whether to verify parity

    Returns:
        (phases, parity_ok)
    """
    # Dequantize
    phases = dequantize_rgb_phases(rgb)

    # Verify parity
    parity_ok = verify_rgb_parity(rgb, FINE_BITS) if verify else True

    return phases, parity_ok


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE-LEVEL ENCODING
# ═══════════════════════════════════════════════════════════════════════════════

def encode_phase_image(
    phases_array: np.ndarray,
    embed_parity: bool = True
) -> np.ndarray:
    """
    Encode 2D array of phase triplets to RGB image.

    Args:
        phases_array: Shape (H, W, 3) of phases in radians
        embed_parity: Whether to use parity encoding

    Returns:
        RGB image as uint8 array (H, W, 3)
    """
    H, W, _ = phases_array.shape
    image = np.zeros((H, W, 3), dtype=np.uint8)

    for y in range(H):
        for x in range(W):
            phases = tuple(phases_array[y, x])
            frame = encode_frame(phases, embed_parity)
            image[y, x] = frame.rgb

    return image


def decode_phase_image(
    image: np.ndarray,
    verify: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decode RGB image to phase array.

    Args:
        image: RGB image as uint8 array (H, W, 3)
        verify: Whether to verify parity

    Returns:
        (phases_array, parity_mask)
        - phases_array: Shape (H, W, 3) of decoded phases
        - parity_mask: Shape (H, W) boolean, True = parity OK
    """
    H, W, _ = image.shape
    phases_array = np.zeros((H, W, 3), dtype=np.float64)
    parity_mask = np.ones((H, W), dtype=bool)

    for y in range(H):
        for x in range(W):
            rgb = tuple(image[y, x])
            phases, ok = decode_frame(rgb, verify)
            phases_array[y, x] = phases
            parity_mask[y, x] = ok

    return phases_array, parity_mask


# ═══════════════════════════════════════════════════════════════════════════════
# PAYLOAD STEGANOGRAPHY (Arbitrary data in LSB)
# ═══════════════════════════════════════════════════════════════════════════════

def embed_payload_in_image(
    image: np.ndarray,
    payload: bytes,
    bits_per_channel: int = 2
) -> np.ndarray:
    """
    Embed arbitrary payload in image LSBs.

    Structure:
        [4 bytes: magic "L4"] [4 bytes: length] [payload] [4 bytes: CRC32]

    Args:
        image: RGB image (H, W, 3)
        payload: Bytes to embed
        bits_per_channel: LSBs per channel (1-4)

    Returns:
        Modified image with embedded payload
    """
    H, W, C = image.shape
    capacity_bits = H * W * C * bits_per_channel

    # Build frame: magic + length + payload + crc
    frame = MRP_MAGIC + struct.pack('>I', len(payload)) + payload
    crc = zlib.crc32(frame) & 0xFFFFFFFF
    frame += struct.pack('>I', crc)

    # Check capacity
    required_bits = len(frame) * 8
    if required_bits > capacity_bits:
        raise ValueError(f"Payload too large: {required_bits} bits > {capacity_bits} capacity")

    # Flatten image
    flat = image.flatten().astype(np.int32)

    # Embed bits
    bit_idx = 0
    for byte in frame:
        for bit_pos in range(7, -1, -1):
            bit = (byte >> bit_pos) & 1

            # Multi-bit embedding
            pixel_idx = bit_idx // bits_per_channel
            bit_offset = bit_idx % bits_per_channel

            if pixel_idx < len(flat):
                mask = ~(1 << bit_offset)
                flat[pixel_idx] = (flat[pixel_idx] & mask) | (bit << bit_offset)

            bit_idx += 1

    return flat.reshape(H, W, C).astype(np.uint8)


def extract_payload_from_image(
    image: np.ndarray,
    bits_per_channel: int = 2
) -> Optional[bytes]:
    """
    Extract payload from image LSBs.

    Args:
        image: RGB image with embedded payload
        bits_per_channel: LSBs per channel used

    Returns:
        Extracted payload bytes, or None if invalid
    """
    flat = image.flatten()

    def extract_bits(n_bits: int, start_idx: int) -> Tuple[bytes, int]:
        """Extract n_bits starting at start_idx."""
        result = []
        current_byte = 0
        bit_count = 0
        idx = start_idx

        while bit_count < n_bits:
            pixel_idx = idx // bits_per_channel
            bit_offset = idx % bits_per_channel

            if pixel_idx >= len(flat):
                break

            bit = (flat[pixel_idx] >> bit_offset) & 1
            current_byte = (current_byte << 1) | bit
            bit_count += 1

            if bit_count % 8 == 0:
                result.append(current_byte)
                current_byte = 0

            idx += 1

        return bytes(result), idx

    # Extract magic (2 bytes)
    magic_bytes, idx = extract_bits(16, 0)
    if magic_bytes != MRP_MAGIC:
        return None

    # Extract length (4 bytes)
    length_bytes, idx = extract_bits(32, idx)
    payload_len = struct.unpack('>I', length_bytes)[0]

    # Extract payload
    payload, idx = extract_bits(payload_len * 8, idx)

    # Extract CRC (4 bytes)
    crc_bytes, idx = extract_bits(32, idx)
    stored_crc = struct.unpack('>I', crc_bytes)[0]

    # Verify CRC
    frame = MRP_MAGIC + length_bytes + payload
    computed_crc = zlib.crc32(frame) & 0xFFFFFFFF

    if computed_crc != stored_crc:
        return None

    return payload


# ═══════════════════════════════════════════════════════════════════════════════
# HEX GRID NAVIGATION ENCODING
# ═══════════════════════════════════════════════════════════════════════════════

# Hexagonal wavevectors (using Z_CRITICAL = √3/2 from constants.py)
U_1: np.ndarray = np.array([1.0, 0.0])                    # East
U_2: np.ndarray = np.array([0.5, Z_CRITICAL])             # 60° NE (√3/2 from z_c)
U_3: np.ndarray = np.array([0.5, -Z_CRITICAL])            # -60° SE
K_HEX: np.ndarray = np.vstack([U_1, U_2, U_3])


def position_to_phases(
    position: np.ndarray,
    wavelength: float = 1.0,
    offset: np.ndarray = None
) -> Tuple[float, float, float]:
    """
    Convert 2D position to hex grid phases.

    Φⱼ = (2π/λ) × kⱼ · x + θⱼ

    This is the grid cell firing phase equation.

    Args:
        position: 2D position (x, y)
        wavelength: Grid wavelength (spatial period)
        offset: Phase offsets (3,)

    Returns:
        (Φ₁, Φ₂, Φ₃) phases in [0, 2π)
    """
    if offset is None:
        offset = np.zeros(3)

    k_scaled = K_HEX * (TWO_PI / wavelength)
    phases = k_scaled @ position + offset
    phases = phases % TWO_PI

    return tuple(phases)


def phases_to_position(
    phases: Tuple[float, float, float],
    wavelength: float = 1.0,
    offset: np.ndarray = None
) -> np.ndarray:
    """
    Decode phases back to 2D position (least-squares).

    Args:
        phases: (Φ₁, Φ₂, Φ₃) in radians
        wavelength: Grid wavelength
        offset: Phase offsets

    Returns:
        Estimated 2D position
    """
    if offset is None:
        offset = np.zeros(3)

    phi = np.array(phases) - offset
    k_scaled = K_HEX * (TWO_PI / wavelength)

    # Least-squares solution
    position, _, _, _ = np.linalg.lstsq(k_scaled, phi, rcond=None)

    return position


def encode_trajectory(
    positions: np.ndarray,
    wavelength: float = 1.0
) -> np.ndarray:
    """
    Encode trajectory as sequence of RGB pixels.

    Args:
        positions: (N, 2) array of positions
        wavelength: Grid wavelength

    Returns:
        (N, 3) array of RGB values
    """
    N = len(positions)
    rgb_trajectory = np.zeros((N, 3), dtype=np.uint8)

    for i, pos in enumerate(positions):
        phases = position_to_phases(pos, wavelength)
        frame = encode_frame(phases)
        rgb_trajectory[i] = frame.rgb

    return rgb_trajectory


def decode_trajectory(
    rgb_trajectory: np.ndarray,
    wavelength: float = 1.0
) -> np.ndarray:
    """
    Decode RGB sequence back to trajectory.

    Args:
        rgb_trajectory: (N, 3) array of RGB values
        wavelength: Grid wavelength

    Returns:
        (N, 2) array of decoded positions
    """
    N = len(rgb_trajectory)
    positions = np.zeros((N, 2))

    for i, rgb in enumerate(rgb_trajectory):
        phases, _ = decode_frame(tuple(rgb))
        positions[i] = phases_to_position(phases, wavelength)

    return positions


# ═══════════════════════════════════════════════════════════════════════════════
# VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def verify_roundtrip(phases: Tuple[float, float, float], tolerance: float = 0.05) -> Dict:
    """
    Verify encode-decode roundtrip accuracy.

    Args:
        phases: Original phases
        tolerance: Maximum allowed error in radians

    Returns:
        Dict with verification results
    """
    # Encode
    frame = encode_frame(phases)

    # Decode
    decoded, parity_ok = decode_frame(frame.rgb)

    # Errors
    errors = [abs(d - o) for d, o in zip(decoded, phases)]
    # Handle wraparound
    errors = [min(e, TWO_PI - e) for e in errors]
    max_error = max(errors)

    return {
        'original': phases,
        'rgb': frame.rgb,
        'decoded': decoded,
        'errors_rad': errors,
        'max_error_rad': max_error,
        'max_error_deg': max_error * 180 / math.pi,
        'parity_ok': parity_ok,
        'within_tolerance': max_error < tolerance
    }


if __name__ == '__main__':
    print("=" * 70)
    print("MRP-LSB ENCODING MODULE — L₄ Framework v3.2.0")
    print("=" * 70)

    # Test phase quantization
    print("\n1. PHASE QUANTIZATION")
    print("-" * 40)
    test_phases = (0.5, math.pi, 1.5 * math.pi)
    print(f"Original phases: {tuple(f'{p:.4f}' for p in test_phases)}")

    rgb = quantize_phases_rgb(test_phases)
    print(f"Quantized RGB: {rgb}")

    decoded = dequantize_rgb_phases(rgb)
    print(f"Decoded phases: {tuple(f'{p:.4f}' for p in decoded)}")

    # Test MRP encoding
    print("\n2. MULTI-RESOLUTION PHASE")
    print("-" * 40)
    mrp = encode_mrp(math.pi / 4)
    print(f"Phase: π/4 = {math.pi/4:.4f} rad = 45°")
    print(f"MRP: coarse={mrp.coarse}, fine={mrp.fine}, full={mrp.full}")
    print(f"Resolution: {mrp.resolution_deg[0]:.2f}° coarse, {mrp.resolution_deg[1]:.2f}° fine")

    # Test roundtrip
    print("\n3. ENCODE-DECODE ROUNDTRIP")
    print("-" * 40)
    test_phases = (0.7, 2.1, 4.5)
    result = verify_roundtrip(test_phases)
    print(f"Original: {tuple(f'{p:.4f}' for p in result['original'])}")
    print(f"RGB: {result['rgb']}")
    print(f"Decoded: {tuple(f'{p:.4f}' for p in result['decoded'])}")
    print(f"Max error: {result['max_error_deg']:.4f}°")
    print(f"Parity OK: {result['parity_ok']}")

    # Test position encoding
    print("\n4. HEX GRID POSITION ENCODING")
    print("-" * 40)
    pos = np.array([1.0, 0.5])
    phases = position_to_phases(pos, wavelength=2.0)
    decoded_pos = phases_to_position(phases, wavelength=2.0)
    print(f"Original position: {pos}")
    print(f"Phases: {tuple(f'{p:.4f}' for p in phases)}")
    print(f"Decoded position: {decoded_pos}")
    print(f"Position error: {np.linalg.norm(decoded_pos - pos):.6f}")

    # Test payload embedding
    print("\n5. PAYLOAD STEGANOGRAPHY")
    print("-" * 40)
    test_image = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
    payload = b"L4 Framework Test Payload"

    embedded = embed_payload_in_image(test_image.copy(), payload)
    extracted = extract_payload_from_image(embedded)

    print(f"Original payload: {payload}")
    print(f"Extracted: {extracted}")
    print(f"Match: {chr(10003) if extracted == payload else chr(10007)}")

    print("\n" + "=" * 70)
    print("MRP-LSB module ready.")
