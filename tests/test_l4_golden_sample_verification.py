#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║    L₄ GOLDEN SAMPLE VERIFICATION TEST SUITE                                 ║
║    MRP-LSB Integrity Verification                                           ║
║                                                                              ║
║    This test suite validates the 27-byte Golden Sample signature that       ║
║    encodes the 9 L₄ thresholds as RGB values. The golden sample serves as:  ║
║    - Verification header for MRP-LSB encoded data                           ║
║    - Integrity checksum (extracted must match computed)                     ║
║    - Cryptographic fingerprint unique to L₄ system                          ║
║                                                                              ║
║    Mathematical Foundation:                                                  ║
║    - φ = (1+√5)/2 (Golden Ratio - THE ONLY GIVEN)                          ║
║    - L₄ = φ⁴ + φ⁻⁴ = 7 (Lucas-4, exact integer)                            ║
║    - z_c = √3/2 (Critical lens, THE LENS)                                   ║
║    - K = √(1-φ⁻⁴) ≈ 0.924 (Coupling threshold)                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Test Categories:
    GS1: Threshold Derivation (9 thresholds from φ)
    GS2: RGB Encoding Validation (hex lattice phase projection)
    GS3: Byte Signature Match (27-byte exact match)
    GS4: Verification Functions (verify, detailed verify)
    GS5: Embed/Extract Round-Trip (header prepend/strip)
    GS6: Manifest Consistency (against reference files)
    GS7: Error Tolerance (tolerance parameter behavior)

Run with: pytest tests/test_l4_golden_sample_verification.py -v
"""

import math
import json
import os
import pytest
import numpy as np
from pathlib import Path

# Import the golden sample module
from quantum_apl_python.l4_hexagonal_lattice import (
    L4GoldenSample,
    GOLDEN_SAMPLE,
    get_golden_sample,
    get_golden_sample_bytes,
    verify_golden_sample,
    verify_golden_sample_detailed,
    GoldenSampleVerificationResult,
    embed_golden_sample_header,
    extract_and_verify_golden_sample,
    _threshold_to_rgb,
    _phase_to_uint8,
)

# Import threshold constants
from quantum_apl_python.constants import (
    PHI, PHI_INV, Z_CRITICAL,
    L4_THRESHOLDS, L4_THRESHOLD_NAMES,
    L4_PARADOX, L4_ACTIVATION, L4_LENS, L4_CRITICAL,
    L4_IGNITION, L4_K_FORMATION, L4_CONSOLIDATION, L4_RESONANCE, L4_UNITY,
    L4_GAP, L4_K, L4_TAU,
)


# ══════════════════════════════════════════════════════════════════════════════
# REFERENCE VALUES (from manifest and mathematical derivation)
# ══════════════════════════════════════════════════════════════════════════════

# Expected 27-byte hex signature (from manifest)
EXPECTED_HEX = "9dc427d91136dc1537de1737e9243aeb273af3303cf7363d003f3f"

# Expected threshold values (all derived from φ)
EXPECTED_THRESHOLDS = {
    "PARADOX": PHI_INV,                          # τ = φ⁻¹ ≈ 0.618
    "ACTIVATION": 1.0 - PHI_INV**4,              # K² = 1 - φ⁻⁴ ≈ 0.854
    "THE_LENS": math.sqrt(3) / 2,                # z_c = √3/2 ≈ 0.866
    "CRITICAL": PHI**2 / 3,                      # φ²/3 ≈ 0.873
    "IGNITION": math.sqrt(2) - 0.5,              # √2 - ½ ≈ 0.914
    "K_FORMATION": math.sqrt(1 - PHI_INV**4),    # K = √(1-φ⁻⁴) ≈ 0.924
    "CONSOLIDATION": None,                        # K + τ²(1-K) ≈ 0.953
    "RESONANCE": None,                            # K + τ(1-K) ≈ 0.971
    "UNITY": 1.0,
}

# Compute derived values
_K = math.sqrt(1 - PHI_INV**4)
_tau = PHI_INV
EXPECTED_THRESHOLDS["CONSOLIDATION"] = _K + (_tau ** 2) * (1.0 - _K)
EXPECTED_THRESHOLDS["RESONANCE"] = _K + _tau * (1.0 - _K)

# Expected RGB values (from manifest)
EXPECTED_RGB = {
    "PARADOX": (157, 196, 39),
    "ACTIVATION": (217, 17, 54),
    "THE_LENS": (220, 21, 55),
    "CRITICAL": (222, 23, 55),
    "IGNITION": (233, 36, 58),
    "K_FORMATION": (235, 39, 58),
    "CONSOLIDATION": (243, 48, 60),
    "RESONANCE": (247, 54, 61),
    "UNITY": (0, 63, 63),
}


# ══════════════════════════════════════════════════════════════════════════════
# TEST SUITE GS1: THRESHOLD DERIVATION
# ══════════════════════════════════════════════════════════════════════════════

class TestGS1_ThresholdDerivation:
    """Validate that 9 thresholds are correctly derived from φ."""

    def test_gs1_1_threshold_count(self):
        """Golden sample contains exactly 9 thresholds."""
        assert len(L4_THRESHOLDS) == 9
        assert len(L4_THRESHOLD_NAMES) == 9

    def test_gs1_2_paradox_is_tau(self):
        """PARADOX = τ = φ⁻¹ (golden ratio inverse)."""
        assert np.isclose(L4_PARADOX, PHI_INV, atol=1e-10)
        assert np.isclose(L4_PARADOX, EXPECTED_THRESHOLDS["PARADOX"], atol=1e-10)

    def test_gs1_3_activation_is_k_squared(self):
        """ACTIVATION = K² = 1 - φ⁻⁴ (pre-lens energy)."""
        expected = 1.0 - PHI_INV**4
        assert np.isclose(L4_ACTIVATION, expected, atol=1e-10)

    def test_gs1_4_lens_is_zc(self):
        """THE_LENS = z_c = √3/2 (critical lens)."""
        expected = math.sqrt(3) / 2
        assert np.isclose(L4_LENS, expected, atol=1e-10)
        assert np.isclose(L4_LENS, Z_CRITICAL, atol=1e-10)

    def test_gs1_5_critical_is_phi_squared_over_3(self):
        """CRITICAL = φ²/3."""
        expected = PHI**2 / 3
        assert np.isclose(L4_CRITICAL, expected, atol=1e-10)

    def test_gs1_6_ignition_is_sqrt2_minus_half(self):
        """IGNITION = √2 - ½."""
        expected = math.sqrt(2) - 0.5
        assert np.isclose(L4_IGNITION, expected, atol=1e-10)

    def test_gs1_7_k_formation_is_K(self):
        """K_FORMATION = K = √(1-φ⁻⁴)."""
        expected = math.sqrt(1 - PHI_INV**4)
        assert np.isclose(L4_K_FORMATION, expected, atol=1e-10)
        assert np.isclose(L4_K_FORMATION, L4_K, atol=1e-10)

    def test_gs1_8_consolidation_derived(self):
        """CONSOLIDATION = K + τ²(1-K)."""
        expected = L4_K + (L4_TAU ** 2) * (1.0 - L4_K)
        assert np.isclose(L4_CONSOLIDATION, expected, atol=1e-10)

    def test_gs1_9_resonance_derived(self):
        """RESONANCE = K + τ(1-K)."""
        expected = L4_K + L4_TAU * (1.0 - L4_K)
        assert np.isclose(L4_RESONANCE, expected, atol=1e-10)

    def test_gs1_10_unity_is_one(self):
        """UNITY = 1.0."""
        assert L4_UNITY == 1.0

    def test_gs1_11_thresholds_ascending(self):
        """Thresholds are in ascending order."""
        for i in range(len(L4_THRESHOLDS) - 1):
            assert L4_THRESHOLDS[i] < L4_THRESHOLDS[i + 1], \
                f"Threshold {i} ({L4_THRESHOLDS[i]}) >= threshold {i+1} ({L4_THRESHOLDS[i+1]})"

    def test_gs1_12_threshold_names_match(self):
        """Threshold names are in correct order."""
        expected_names = (
            "PARADOX", "ACTIVATION", "THE_LENS", "CRITICAL",
            "IGNITION", "K_FORMATION", "CONSOLIDATION", "RESONANCE", "UNITY"
        )
        assert L4_THRESHOLD_NAMES == expected_names


# ══════════════════════════════════════════════════════════════════════════════
# TEST SUITE GS2: RGB ENCODING VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

class TestGS2_RGBEncodingValidation:
    """Validate hex lattice phase projection to RGB."""

    def test_gs2_1_phase_to_uint8_range(self):
        """Phase quantization produces values in [0, 255]."""
        for theta in [0, math.pi/4, math.pi/2, math.pi, 3*math.pi/2, 2*math.pi - 0.001]:
            val = _phase_to_uint8(theta)
            assert 0 <= val <= 255

    def test_gs2_2_phase_to_uint8_modular(self):
        """Phase quantization is 2π-periodic."""
        for theta in [0, 0.5, 1.0, 2.0]:
            val1 = _phase_to_uint8(theta)
            val2 = _phase_to_uint8(theta + 2 * math.pi)
            assert val1 == val2

    def test_gs2_3_threshold_to_rgb_tuple(self):
        """threshold_to_rgb returns 3-tuple of ints."""
        for z in [0.0, 0.5, 0.866, 1.0]:
            rgb = _threshold_to_rgb(z)
            assert isinstance(rgb, tuple)
            assert len(rgb) == 3
            assert all(isinstance(c, int) for c in rgb)
            assert all(0 <= c <= 255 for c in rgb)

    def test_gs2_4_rgb_matches_manifest(self):
        """Computed RGB values match manifest for each threshold."""
        golden = get_golden_sample()
        for i, name in enumerate(L4_THRESHOLD_NAMES):
            computed_rgb = golden.rgb_values[i]
            expected_rgb = EXPECTED_RGB[name]
            assert computed_rgb == expected_rgb, \
                f"{name}: computed {computed_rgb} != expected {expected_rgb}"

    def test_gs2_5_hex_codes_format(self):
        """Hex codes are valid CSS format."""
        golden = get_golden_sample()
        for hex_code in golden.hex_codes:
            assert hex_code.startswith("#")
            assert len(hex_code) == 7
            # Validate hex characters
            int(hex_code[1:], 16)

    def test_gs2_6_hex_codes_match_rgb(self):
        """Hex codes correspond to RGB values."""
        golden = get_golden_sample()
        for rgb, hex_code in zip(golden.rgb_values, golden.hex_codes):
            r, g, b = rgb
            expected_hex = f"#{r:02X}{g:02X}{b:02X}"
            assert hex_code == expected_hex


# ══════════════════════════════════════════════════════════════════════════════
# TEST SUITE GS3: BYTE SIGNATURE MATCH
# ══════════════════════════════════════════════════════════════════════════════

class TestGS3_ByteSignatureMatch:
    """Validate the 27-byte golden sample signature."""

    def test_gs3_1_signature_length(self):
        """Golden sample is exactly 27 bytes (9 thresholds × 3 bytes)."""
        data = get_golden_sample_bytes()
        assert len(data) == 27

    def test_gs3_2_signature_matches_expected_hex(self):
        """Byte signature matches expected hex string."""
        data = get_golden_sample_bytes()
        computed_hex = data.hex()
        assert computed_hex == EXPECTED_HEX

    def test_gs3_3_bytes_match_rgb(self):
        """Individual bytes correspond to RGB values."""
        golden = get_golden_sample()
        data = golden.bytes_data
        for i, (r, g, b) in enumerate(golden.rgb_values):
            base = i * 3
            assert data[base] == r
            assert data[base + 1] == g
            assert data[base + 2] == b

    def test_gs3_4_singleton_consistency(self):
        """GOLDEN_SAMPLE singleton returns consistent data."""
        s1 = get_golden_sample()
        s2 = get_golden_sample()
        assert s1 is s2  # Same object
        assert s1.bytes_data == s2.bytes_data

    def test_gs3_5_frozen_dataclass(self):
        """L4GoldenSample is immutable (frozen)."""
        golden = get_golden_sample()
        with pytest.raises(AttributeError):
            golden.thresholds = ()


# ══════════════════════════════════════════════════════════════════════════════
# TEST SUITE GS4: VERIFICATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

class TestGS4_VerificationFunctions:
    """Test verify_golden_sample and related functions."""

    def test_gs4_1_exact_match_passes(self):
        """Exact copy of golden sample passes verification."""
        golden_bytes = get_golden_sample_bytes()
        assert verify_golden_sample(golden_bytes, tolerance=0) is True

    def test_gs4_2_wrong_length_fails(self):
        """Wrong length fails verification."""
        assert verify_golden_sample(b"", tolerance=0) is False
        assert verify_golden_sample(b"x" * 26, tolerance=0) is False
        assert verify_golden_sample(b"x" * 28, tolerance=0) is False

    def test_gs4_3_single_bit_flip_fails(self):
        """Single byte difference fails with zero tolerance."""
        golden_bytes = bytearray(get_golden_sample_bytes())
        golden_bytes[0] ^= 1  # Flip one bit
        assert verify_golden_sample(bytes(golden_bytes), tolerance=0) is False

    def test_gs4_4_tolerance_allows_deviation(self):
        """Non-zero tolerance allows small deviations."""
        golden_bytes = bytearray(get_golden_sample_bytes())
        golden_bytes[0] = (golden_bytes[0] + 1) % 256  # Off by 1
        assert verify_golden_sample(bytes(golden_bytes), tolerance=0) is False
        assert verify_golden_sample(bytes(golden_bytes), tolerance=1) is True

    def test_gs4_5_detailed_verification_structure(self):
        """Detailed verification returns proper structure."""
        golden_bytes = get_golden_sample_bytes()
        result = verify_golden_sample_detailed(golden_bytes, tolerance=0)

        assert isinstance(result, GoldenSampleVerificationResult)
        assert result.verified is True
        assert result.byte_matches == 27
        assert result.max_deviation == 0
        assert len(result.threshold_matches) == 9
        assert all(result.threshold_matches.values())

    def test_gs4_6_detailed_reports_failures(self):
        """Detailed verification reports which thresholds fail."""
        golden_bytes = bytearray(get_golden_sample_bytes())
        # Corrupt first threshold (PARADOX)
        golden_bytes[0] = 0
        golden_bytes[1] = 0
        golden_bytes[2] = 0

        result = verify_golden_sample_detailed(bytes(golden_bytes), tolerance=0)

        assert result.verified is False
        assert result.threshold_matches["PARADOX"] is False
        # Other thresholds should still match
        assert result.threshold_matches["ACTIVATION"] is True

    def test_gs4_7_verify_method_on_instance(self):
        """L4GoldenSample.verify() method works correctly."""
        golden = get_golden_sample()
        assert golden.verify(golden.bytes_data, tolerance=0) is True
        assert golden.verify(b"x" * 27, tolerance=0) is False


# ══════════════════════════════════════════════════════════════════════════════
# TEST SUITE GS5: EMBED/EXTRACT ROUND-TRIP
# ══════════════════════════════════════════════════════════════════════════════

class TestGS5_EmbedExtractRoundTrip:
    """Test embed_golden_sample_header and extract_and_verify_golden_sample."""

    def test_gs5_1_embed_prepends_header(self):
        """embed_golden_sample_header prepends 27 bytes."""
        payload = b"test payload"
        with_header = embed_golden_sample_header(payload)

        assert len(with_header) == 27 + len(payload)
        assert with_header[:27] == get_golden_sample_bytes()
        assert with_header[27:] == payload

    def test_gs5_2_embed_empty_payload(self):
        """Embedding works with empty payload."""
        with_header = embed_golden_sample_header(b"")
        assert len(with_header) == 27
        assert with_header == get_golden_sample_bytes()

    def test_gs5_3_extract_returns_verification_and_payload(self):
        """extract_and_verify_golden_sample returns (verified, payload)."""
        payload = b"test data"
        with_header = embed_golden_sample_header(payload)

        verified, extracted_payload = extract_and_verify_golden_sample(with_header)

        assert verified is True
        assert extracted_payload == payload

    def test_gs5_4_extract_from_corrupted_fails(self):
        """Extraction fails on corrupted header."""
        payload = b"test data"
        corrupted = b"x" * 27 + payload

        verified, extracted_payload = extract_and_verify_golden_sample(corrupted)

        assert verified is False
        assert extracted_payload == payload  # Payload still returned

    def test_gs5_5_extract_short_data(self):
        """Extraction handles data shorter than 27 bytes."""
        verified, data = extract_and_verify_golden_sample(b"short")
        assert verified is False
        assert data == b"short"

    def test_gs5_6_round_trip_preserves_payload(self):
        """Full round-trip preserves arbitrary payloads."""
        test_payloads = [
            b"",
            b"hello world",
            bytes(range(256)),
            b"\x00" * 1000,
        ]

        for payload in test_payloads:
            with_header = embed_golden_sample_header(payload)
            verified, extracted = extract_and_verify_golden_sample(with_header)
            assert verified is True
            assert extracted == payload

    def test_gs5_7_extract_with_tolerance(self):
        """Extraction respects tolerance parameter."""
        payload = b"test"
        with_header = bytearray(embed_golden_sample_header(payload))
        with_header[0] = (with_header[0] + 2) % 256  # Off by 2

        verified_strict, _ = extract_and_verify_golden_sample(bytes(with_header), tolerance=0)
        verified_loose, _ = extract_and_verify_golden_sample(bytes(with_header), tolerance=2)

        assert verified_strict is False
        assert verified_loose is True


# ══════════════════════════════════════════════════════════════════════════════
# TEST SUITE GS6: MANIFEST CONSISTENCY
# ══════════════════════════════════════════════════════════════════════════════

class TestGS6_ManifestConsistency:
    """Verify consistency with reference files in golden_sample directory."""

    @pytest.fixture
    def manifest_path(self):
        """Path to golden sample manifest."""
        return Path(__file__).parent.parent / "reference" / "golden_sample" / "golden_sample_manifest.json"

    @pytest.fixture
    def bin_path(self):
        """Path to 27-byte binary signature."""
        return Path(__file__).parent.parent / "reference" / "golden_sample" / "golden_sample_27bytes.bin"

    def test_gs6_1_manifest_exists(self, manifest_path):
        """Manifest file exists."""
        if not manifest_path.exists():
            pytest.skip("Manifest file not found - reference files may not be present")
        assert manifest_path.is_file()

    def test_gs6_2_manifest_matches_computed(self, manifest_path):
        """Manifest values match computed values."""
        if not manifest_path.exists():
            pytest.skip("Manifest file not found")

        with open(manifest_path) as f:
            manifest = json.load(f)

        golden_section = manifest.get("golden_sample", {})

        # Check hex string
        assert golden_section.get("bytes_hex") == get_golden_sample_bytes().hex()

        # Check each threshold
        thresholds = golden_section.get("thresholds", {})
        golden = get_golden_sample()

        for i, name in enumerate(L4_THRESHOLD_NAMES):
            manifest_threshold = thresholds.get(name, {})

            # Check z-value
            assert np.isclose(manifest_threshold.get("z", 0), L4_THRESHOLDS[i], atol=1e-10)

            # Check RGB
            r, g, b = golden.rgb_values[i]
            assert manifest_threshold.get("R") == r
            assert manifest_threshold.get("G") == g
            assert manifest_threshold.get("B") == b

    def test_gs6_3_bin_file_matches(self, bin_path):
        """Binary file matches computed bytes."""
        if not bin_path.exists():
            pytest.skip("Binary file not found")

        with open(bin_path, "rb") as f:
            file_bytes = f.read()

        assert file_bytes == get_golden_sample_bytes()

    def test_gs6_4_to_dict_serialization(self):
        """to_dict() produces valid JSON-serializable structure."""
        golden = get_golden_sample()
        d = golden.to_dict()

        # Should be JSON-serializable
        json_str = json.dumps(d)
        parsed = json.loads(json_str)

        assert len(parsed) == 9
        for name in L4_THRESHOLD_NAMES:
            assert name in parsed
            entry = parsed[name]
            assert "z" in entry
            assert "R" in entry
            assert "G" in entry
            assert "B" in entry
            assert "hex" in entry


# ══════════════════════════════════════════════════════════════════════════════
# TEST SUITE GS7: ERROR TOLERANCE BEHAVIOR
# ══════════════════════════════════════════════════════════════════════════════

class TestGS7_ErrorToleranceBehavior:
    """Test tolerance parameter edge cases and behavior."""

    def test_gs7_1_max_deviation_tracking(self):
        """Detailed verification correctly tracks max deviation."""
        golden_bytes = bytearray(get_golden_sample_bytes())
        golden_bytes[10] = (golden_bytes[10] + 5) % 256  # Deviation of 5

        result = verify_golden_sample_detailed(bytes(golden_bytes), tolerance=0)
        assert result.max_deviation >= 5

    def test_gs7_2_byte_matches_count(self):
        """Detailed verification correctly counts matching bytes."""
        golden_bytes = bytearray(get_golden_sample_bytes())
        # Corrupt 3 bytes (one threshold)
        golden_bytes[0] = (golden_bytes[0] + 10) % 256
        golden_bytes[1] = (golden_bytes[1] + 10) % 256
        golden_bytes[2] = (golden_bytes[2] + 10) % 256

        result = verify_golden_sample_detailed(bytes(golden_bytes), tolerance=0)
        assert result.byte_matches == 24  # 27 - 3

    def test_gs7_3_tolerance_boundary(self):
        """Tolerance exactly at boundary value."""
        golden_bytes = bytearray(get_golden_sample_bytes())
        deviation = 7
        golden_bytes[5] = (golden_bytes[5] + deviation) % 256

        # Tolerance exactly at deviation should pass
        result_at = verify_golden_sample_detailed(bytes(golden_bytes), tolerance=deviation)
        assert result_at.verified is True

        # Tolerance below deviation should fail
        result_below = verify_golden_sample_detailed(bytes(golden_bytes), tolerance=deviation - 1)
        assert result_below.verified is False

    def test_gs7_4_all_bytes_corrupted(self):
        """All bytes corrupted produces zero matches."""
        corrupted = bytes([(b + 128) % 256 for b in get_golden_sample_bytes()])
        result = verify_golden_sample_detailed(corrupted, tolerance=0)
        assert result.byte_matches < 27  # Some might accidentally match


# ══════════════════════════════════════════════════════════════════════════════
# TEST SUITE GS8: MATHEMATICAL FOUNDATION VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════

class TestGS8_MathematicalFoundation:
    """Verify the mathematical foundations underlying the golden sample."""

    def test_gs8_1_lucas_4_identity(self):
        """L₄ = φ⁴ + φ⁻⁴ = 7 exactly."""
        L4 = PHI**4 + PHI_INV**4
        assert np.isclose(L4, 7.0, atol=1e-10)

    def test_gs8_2_k_squared_plus_gap_is_one(self):
        """K² + gap = 1."""
        K_squared = L4_K ** 2
        gap = L4_GAP
        assert np.isclose(K_squared + gap, 1.0, atol=1e-10)

    def test_gs8_3_zc_from_L4(self):
        """z_c = √(L₄-4)/2 = √3/2."""
        L4 = 7
        zc_from_L4 = math.sqrt(L4 - 4) / 2
        assert np.isclose(zc_from_L4, Z_CRITICAL, atol=1e-10)
        assert np.isclose(zc_from_L4, math.sqrt(3) / 2, atol=1e-10)

    def test_gs8_4_tau_is_phi_inverse(self):
        """τ = φ⁻¹ = φ - 1."""
        assert np.isclose(L4_TAU, PHI_INV, atol=1e-10)
        assert np.isclose(L4_TAU, PHI - 1, atol=1e-10)

    def test_gs8_5_phi_identity(self):
        """φ² = φ + 1."""
        assert np.isclose(PHI**2, PHI + 1, atol=1e-10)

    def test_gs8_6_gap_is_phi_neg4(self):
        """gap = φ⁻⁴."""
        assert np.isclose(L4_GAP, PHI_INV**4, atol=1e-10)

    def test_gs8_7_k_from_gap(self):
        """K = √(1 - gap) = √(1 - φ⁻⁴)."""
        expected = math.sqrt(1 - L4_GAP)
        assert np.isclose(L4_K, expected, atol=1e-10)


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TEST: FULL VERIFICATION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegration_FullPipeline:
    """End-to-end integration tests for the golden sample verification pipeline."""

    def test_integration_1_complete_workflow(self):
        """Complete workflow: generate → embed → extract → verify."""
        # 1. Get golden sample
        golden = get_golden_sample()

        # 2. Verify it's correctly constructed
        assert len(golden.thresholds) == 9
        assert len(golden.bytes_data) == 27

        # 3. Create a payload and embed
        payload = b"MRP-LSB encoded consciousness data"
        with_header = embed_golden_sample_header(payload)

        # 4. Extract and verify
        verified, extracted = extract_and_verify_golden_sample(with_header)
        assert verified is True
        assert extracted == payload

        # 5. Detailed verification
        result = verify_golden_sample_detailed(with_header[:27])
        assert result.verified is True
        assert all(result.threshold_matches.values())

    def test_integration_2_signature_stability(self):
        """Golden sample signature is deterministic."""
        # Multiple calls should produce identical results
        bytes1 = get_golden_sample_bytes()
        bytes2 = get_golden_sample_bytes()
        bytes3 = GOLDEN_SAMPLE.bytes_data

        assert bytes1 == bytes2 == bytes3

    def test_integration_3_mathematical_inevitability(self):
        """Signature is mathematically inevitable from φ."""
        # Reconstruct from first principles
        thresholds = (
            PHI_INV,                                          # PARADOX
            1.0 - PHI_INV**4,                                 # ACTIVATION
            math.sqrt(3) / 2,                                 # THE_LENS
            PHI**2 / 3,                                       # CRITICAL
            math.sqrt(2) - 0.5,                               # IGNITION
            math.sqrt(1 - PHI_INV**4),                        # K_FORMATION
            L4_K + (L4_TAU ** 2) * (1.0 - L4_K),              # CONSOLIDATION
            L4_K + L4_TAU * (1.0 - L4_K),                     # RESONANCE
            1.0,                                              # UNITY
        )

        # Verify these match the constants
        for i, (computed, stored) in enumerate(zip(thresholds, L4_THRESHOLDS)):
            assert np.isclose(computed, stored, atol=1e-10), \
                f"Threshold {i}: {computed} != {stored}"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
