#!/usr/bin/env python3
"""
Test Suite for MRP Phase-A Verification

This module tests the Multi-Channel Resonance Protocol verifier
with various scenarios including valid data, corruption, and edge cases.
"""

import base64
import hashlib
import json
import os
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from mrp_verify import MRPVerifier


class TestMRPVerification(unittest.TestCase):
    """Test cases for MRP Phase-A verification."""

    def setUp(self):
        """Set up test fixtures with valid MRP payloads."""
        self.temp_dir = tempfile.mkdtemp()
        self.verifier = MRPVerifier()

        # Create test payloads
        self.r_payload = {
            "type": "primary",
            "data": "Secret message in R channel",
            "timestamp": "2025-01-12T12:00:00Z",
        }

        self.g_payload = {
            "type": "secondary",
            "data": "Additional data in G channel",
            "metadata": {"version": 1},
        }

        # Compute verification data
        r_min = json.dumps(self.r_payload, separators=(",", ":")).encode()
        g_min = json.dumps(self.g_payload, separators=(",", ":")).encode()
        self.r_b64 = base64.b64encode(r_min)
        self.g_b64 = base64.b64encode(g_min)

        # Calculate CRCs and SHA
        self.crc_r = format(zlib.crc32(self.r_b64) & 0xFFFFFFFF, "08X")
        self.crc_g = format(zlib.crc32(self.g_b64) & 0xFFFFFFFF, "08X")
        self.sha_r = hashlib.sha256(self.r_b64).hexdigest()

        # Generate parity block
        parity = bytearray(len(self.g_b64))
        for i in range(len(self.g_b64)):
            if i < len(self.r_b64):
                parity[i] = self.r_b64[i] ^ self.g_b64[i]
            else:
                parity[i] = self.g_b64[i]
        self.parity_b64 = base64.b64encode(parity).decode()

        # Create B channel verification payload
        self.b_payload = {
            "crc_r": self.crc_r,
            "crc_g": self.crc_g,
            "sha256_msg_b64": self.sha_r,
            "ecc_scheme": "parity",
            "parity_block_b64": self.parity_b64,
        }

        # Write payload files
        self.r_path = Path(self.temp_dir) / "mrp_R_payload.json"
        self.g_path = Path(self.temp_dir) / "mrp_G_payload.json"
        self.b_path = Path(self.temp_dir) / "mrp_B_payload.json"

        with open(self.r_path, "w") as f:
            json.dump(self.r_payload, f, indent=2)
        with open(self.g_path, "w") as f:
            json.dump(self.g_payload, f, indent=2)
        with open(self.b_path, "w") as f:
            json.dump(self.b_payload, f, indent=2)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_valid_mrp_verification(self):
        """Test verification passes with valid MRP data."""
        report = self.verifier.verify(
            r_payload_path=self.r_path,
            g_payload_path=self.g_path,
            b_payload_path=self.b_path,
        )

        self.assertTrue(report["mrp_ok"], "MRP verification should pass")
        self.assertTrue(report["checks"]["crc_r_ok"])
        self.assertTrue(report["checks"]["crc_g_ok"])
        self.assertTrue(report["checks"]["sha256_r_b64_ok"])
        self.assertTrue(report["checks"]["ecc_scheme_ok"])
        self.assertTrue(report["checks"]["parity_block_ok"])

    def test_crc_r_mismatch(self):
        """Test verification fails when R CRC is corrupted."""
        # Corrupt CRC_R
        corrupt_b = dict(self.b_payload)
        corrupt_b["crc_r"] = "00000000"

        corrupt_b_path = Path(self.temp_dir) / "corrupt_B.json"
        with open(corrupt_b_path, "w") as f:
            json.dump(corrupt_b, f)

        report = self.verifier.verify(
            r_payload_path=self.r_path,
            g_payload_path=self.g_path,
            b_payload_path=corrupt_b_path,
        )

        self.assertFalse(report["mrp_ok"])
        self.assertFalse(report["checks"]["crc_r_ok"])
        self.assertTrue(report["checks"]["crc_g_ok"])

    def test_crc_g_mismatch(self):
        """Test verification fails when G CRC is corrupted."""
        corrupt_b = dict(self.b_payload)
        corrupt_b["crc_g"] = "DEADBEEF"

        corrupt_b_path = Path(self.temp_dir) / "corrupt_B.json"
        with open(corrupt_b_path, "w") as f:
            json.dump(corrupt_b, f)

        report = self.verifier.verify(
            r_payload_path=self.r_path,
            g_payload_path=self.g_path,
            b_payload_path=corrupt_b_path,
        )

        self.assertFalse(report["mrp_ok"])
        self.assertTrue(report["checks"]["crc_r_ok"])
        self.assertFalse(report["checks"]["crc_g_ok"])

    def test_sha256_mismatch(self):
        """Test verification fails when SHA-256 is corrupted."""
        corrupt_b = dict(self.b_payload)
        corrupt_b["sha256_msg_b64"] = "0" * 64

        corrupt_b_path = Path(self.temp_dir) / "corrupt_B.json"
        with open(corrupt_b_path, "w") as f:
            json.dump(corrupt_b, f)

        report = self.verifier.verify(
            r_payload_path=self.r_path,
            g_payload_path=self.g_path,
            b_payload_path=corrupt_b_path,
        )

        self.assertFalse(report["mrp_ok"])
        self.assertFalse(report["checks"]["sha256_r_b64_ok"])

    def test_parity_block_mismatch(self):
        """Test verification fails when parity block is corrupted."""
        corrupt_b = dict(self.b_payload)
        corrupt_b["parity_block_b64"] = base64.b64encode(b"invalid").decode()

        corrupt_b_path = Path(self.temp_dir) / "corrupt_B.json"
        with open(corrupt_b_path, "w") as f:
            json.dump(corrupt_b, f)

        report = self.verifier.verify(
            r_payload_path=self.r_path,
            g_payload_path=self.g_path,
            b_payload_path=corrupt_b_path,
        )

        self.assertFalse(report["mrp_ok"])
        self.assertFalse(report["checks"]["parity_block_ok"])

    def test_wrong_ecc_scheme(self):
        """Test verification fails with wrong ECC scheme."""
        corrupt_b = dict(self.b_payload)
        corrupt_b["ecc_scheme"] = "reed-solomon"

        corrupt_b_path = Path(self.temp_dir) / "corrupt_B.json"
        with open(corrupt_b_path, "w") as f:
            json.dump(corrupt_b, f)

        report = self.verifier.verify(
            r_payload_path=self.r_path,
            g_payload_path=self.g_path,
            b_payload_path=corrupt_b_path,
        )

        self.assertFalse(report["mrp_ok"])
        self.assertFalse(report["checks"]["ecc_scheme_ok"])

    def test_sidecar_verification(self):
        """Test verification with valid sidecar metadata."""
        # Create valid sidecar
        sidecar = {
            "file": "mrp_test_image.png",
            "sha256_msg_b64": self.sha_r,
            "channels": {
                "R": {
                    "payload_len": len(self.r_b64),
                    "used_bits": (len(self.r_b64) + 14) * 8,
                    "capacity_bits": 512 * 512,
                },
                "G": {
                    "payload_len": len(self.g_b64),
                    "used_bits": (len(self.g_b64) + 14) * 8,
                    "capacity_bits": 512 * 512,
                },
                "B": {
                    "payload_len": len(json.dumps(self.b_payload, separators=(",", ":")).encode()),
                    "used_bits": (
                        len(json.dumps(self.b_payload, separators=(",", ":")).encode()) + 14
                    )
                    * 8,
                    "capacity_bits": 512 * 512,
                },
            },
            "headers": {
                "R": {"magic": "MRP1", "channel": "R", "flags": 1},
                "G": {"magic": "MRP1", "channel": "G", "flags": 1},
                "B": {"magic": "MRP1", "channel": "B", "flags": 1},
            },
        }

        sidecar_path = Path(self.temp_dir) / "sidecar.json"
        with open(sidecar_path, "w") as f:
            json.dump(sidecar, f, indent=2)

        report = self.verifier.verify(
            r_payload_path=self.r_path,
            g_payload_path=self.g_path,
            b_payload_path=self.b_path,
            sidecar_path=sidecar_path,
        )

        self.assertTrue(report["mrp_ok"])
        self.assertTrue(report["checks"]["sidecar_sha256_ok"])
        self.assertTrue(report["checks"]["sidecar_used_bits_math_ok"])
        self.assertTrue(report["checks"]["sidecar_capacity_bits_ok"])
        self.assertTrue(report["checks"]["sidecar_header_magic_ok"])
        self.assertTrue(report["checks"]["sidecar_header_flags_crc_ok"])

    def test_sidecar_sha256_mismatch(self):
        """Test verification fails when sidecar SHA256 doesn't match."""
        sidecar = {
            "sha256_msg_b64": "0" * 64,
            "channels": {
                "R": {"payload_len": 0, "used_bits": 112, "capacity_bits": 512 * 512},
                "G": {"payload_len": 0, "used_bits": 112, "capacity_bits": 512 * 512},
                "B": {"payload_len": 0, "used_bits": 112, "capacity_bits": 512 * 512},
            },
            "headers": {
                "R": {"magic": "MRP1", "flags": 1},
                "G": {"magic": "MRP1", "flags": 1},
                "B": {"magic": "MRP1", "flags": 1},
            },
        }

        sidecar_path = Path(self.temp_dir) / "bad_sidecar.json"
        with open(sidecar_path, "w") as f:
            json.dump(sidecar, f)

        report = self.verifier.verify(
            r_payload_path=self.r_path,
            g_payload_path=self.g_path,
            b_payload_path=self.b_path,
            sidecar_path=sidecar_path,
        )

        self.assertFalse(report["mrp_ok"])
        self.assertFalse(report["checks"]["sidecar_sha256_ok"])

    def test_sidecar_bad_used_bits(self):
        """Test verification fails when used_bits math is wrong."""
        sidecar = {
            "sha256_msg_b64": self.sha_r,
            "channels": {
                "R": {
                    "payload_len": len(self.r_b64),
                    "used_bits": 9999,  # Wrong!
                    "capacity_bits": 512 * 512,
                },
                "G": {
                    "payload_len": len(self.g_b64),
                    "used_bits": (len(self.g_b64) + 14) * 8,
                    "capacity_bits": 512 * 512,
                },
                "B": {
                    "payload_len": 100,
                    "used_bits": (100 + 14) * 8,
                    "capacity_bits": 512 * 512,
                },
            },
            "headers": {
                "R": {"magic": "MRP1", "flags": 1},
                "G": {"magic": "MRP1", "flags": 1},
                "B": {"magic": "MRP1", "flags": 1},
            },
        }

        sidecar_path = Path(self.temp_dir) / "bad_sidecar.json"
        with open(sidecar_path, "w") as f:
            json.dump(sidecar, f)

        report = self.verifier.verify(
            r_payload_path=self.r_path,
            g_payload_path=self.g_path,
            b_payload_path=self.b_path,
            sidecar_path=sidecar_path,
        )

        self.assertFalse(report["mrp_ok"])
        self.assertFalse(report["checks"]["sidecar_used_bits_math_ok"])

    def test_sidecar_bad_magic(self):
        """Test verification fails when header magic is wrong."""
        sidecar = {
            "sha256_msg_b64": self.sha_r,
            "channels": {
                "R": {
                    "payload_len": len(self.r_b64),
                    "used_bits": (len(self.r_b64) + 14) * 8,
                    "capacity_bits": 512 * 512,
                },
                "G": {
                    "payload_len": len(self.g_b64),
                    "used_bits": (len(self.g_b64) + 14) * 8,
                    "capacity_bits": 512 * 512,
                },
                "B": {
                    "payload_len": 100,
                    "used_bits": (100 + 14) * 8,
                    "capacity_bits": 512 * 512,
                },
            },
            "headers": {
                "R": {"magic": "LSB1", "flags": 1},  # Wrong magic!
                "G": {"magic": "MRP1", "flags": 1},
                "B": {"magic": "MRP1", "flags": 1},
            },
        }

        sidecar_path = Path(self.temp_dir) / "bad_sidecar.json"
        with open(sidecar_path, "w") as f:
            json.dump(sidecar, f)

        report = self.verifier.verify(
            r_payload_path=self.r_path,
            g_payload_path=self.g_path,
            b_payload_path=self.b_path,
            sidecar_path=sidecar_path,
        )

        self.assertFalse(report["mrp_ok"])
        self.assertFalse(report["checks"]["sidecar_header_magic_ok"])

    def test_computed_values(self):
        """Test that computed values are correct."""
        report = self.verifier.verify(
            r_payload_path=self.r_path,
            g_payload_path=self.g_path,
            b_payload_path=self.b_path,
        )

        computed = report["computed"]
        self.assertEqual(computed["crc_r"], self.crc_r)
        self.assertEqual(computed["crc_g"], self.crc_g)
        self.assertEqual(computed["sha256_r_b64"], self.sha_r)

    def test_empty_payload(self):
        """Test verification with empty payloads."""
        empty_r = {}
        empty_g = {}

        r_path = Path(self.temp_dir) / "empty_R.json"
        g_path = Path(self.temp_dir) / "empty_G.json"

        with open(r_path, "w") as f:
            json.dump(empty_r, f)
        with open(g_path, "w") as f:
            json.dump(empty_g, f)

        # Compute B payload for empty data
        r_min = json.dumps(empty_r, separators=(",", ":")).encode()
        g_min = json.dumps(empty_g, separators=(",", ":")).encode()
        r_b64 = base64.b64encode(r_min)
        g_b64 = base64.b64encode(g_min)

        crc_r = format(zlib.crc32(r_b64) & 0xFFFFFFFF, "08X")
        crc_g = format(zlib.crc32(g_b64) & 0xFFFFFFFF, "08X")
        sha_r = hashlib.sha256(r_b64).hexdigest()

        parity = bytearray(len(g_b64))
        for i in range(len(g_b64)):
            if i < len(r_b64):
                parity[i] = r_b64[i] ^ g_b64[i]
            else:
                parity[i] = g_b64[i]
        parity_b64 = base64.b64encode(parity).decode()

        b_payload = {
            "crc_r": crc_r,
            "crc_g": crc_g,
            "sha256_msg_b64": sha_r,
            "ecc_scheme": "parity",
            "parity_block_b64": parity_b64,
        }

        b_path = Path(self.temp_dir) / "empty_B.json"
        with open(b_path, "w") as f:
            json.dump(b_payload, f)

        report = self.verifier.verify(
            r_payload_path=r_path,
            g_payload_path=g_path,
            b_payload_path=b_path,
        )

        self.assertTrue(report["mrp_ok"])

    def test_large_payload(self):
        """Test verification with larger payloads."""
        large_r = {"data": "A" * 10000, "items": list(range(100))}
        large_g = {"data": "B" * 10000, "items": list(range(100))}

        r_path = Path(self.temp_dir) / "large_R.json"
        g_path = Path(self.temp_dir) / "large_G.json"

        with open(r_path, "w") as f:
            json.dump(large_r, f)
        with open(g_path, "w") as f:
            json.dump(large_g, f)

        # Compute B payload
        r_min = json.dumps(large_r, separators=(",", ":")).encode()
        g_min = json.dumps(large_g, separators=(",", ":")).encode()
        r_b64 = base64.b64encode(r_min)
        g_b64 = base64.b64encode(g_min)

        crc_r = format(zlib.crc32(r_b64) & 0xFFFFFFFF, "08X")
        crc_g = format(zlib.crc32(g_b64) & 0xFFFFFFFF, "08X")
        sha_r = hashlib.sha256(r_b64).hexdigest()

        parity = bytearray(len(g_b64))
        for i in range(len(g_b64)):
            if i < len(r_b64):
                parity[i] = r_b64[i] ^ g_b64[i]
            else:
                parity[i] = g_b64[i]
        parity_b64 = base64.b64encode(parity).decode()

        b_payload = {
            "crc_r": crc_r,
            "crc_g": crc_g,
            "sha256_msg_b64": sha_r,
            "ecc_scheme": "parity",
            "parity_block_b64": parity_b64,
        }

        b_path = Path(self.temp_dir) / "large_B.json"
        with open(b_path, "w") as f:
            json.dump(b_payload, f)

        report = self.verifier.verify(
            r_payload_path=r_path,
            g_payload_path=g_path,
            b_payload_path=b_path,
        )

        self.assertTrue(report["mrp_ok"])

    def test_asymmetric_payload_lengths(self):
        """Test verification when R and G have different lengths."""
        short_r = {"x": 1}
        long_g = {"data": "X" * 1000, "nested": {"a": 1, "b": 2, "c": 3}}

        r_path = Path(self.temp_dir) / "short_R.json"
        g_path = Path(self.temp_dir) / "long_G.json"

        with open(r_path, "w") as f:
            json.dump(short_r, f)
        with open(g_path, "w") as f:
            json.dump(long_g, f)

        # Compute B payload
        r_min = json.dumps(short_r, separators=(",", ":")).encode()
        g_min = json.dumps(long_g, separators=(",", ":")).encode()
        r_b64 = base64.b64encode(r_min)
        g_b64 = base64.b64encode(g_min)

        crc_r = format(zlib.crc32(r_b64) & 0xFFFFFFFF, "08X")
        crc_g = format(zlib.crc32(g_b64) & 0xFFFFFFFF, "08X")
        sha_r = hashlib.sha256(r_b64).hexdigest()

        # Parity with asymmetric lengths
        parity = bytearray(len(g_b64))
        for i in range(len(g_b64)):
            if i < len(r_b64):
                parity[i] = r_b64[i] ^ g_b64[i]
            else:
                parity[i] = g_b64[i]
        parity_b64 = base64.b64encode(parity).decode()

        b_payload = {
            "crc_r": crc_r,
            "crc_g": crc_g,
            "sha256_msg_b64": sha_r,
            "ecc_scheme": "parity",
            "parity_block_b64": parity_b64,
        }

        b_path = Path(self.temp_dir) / "asym_B.json"
        with open(b_path, "w") as f:
            json.dump(b_payload, f)

        report = self.verifier.verify(
            r_payload_path=r_path,
            g_payload_path=g_path,
            b_payload_path=b_path,
        )

        self.assertTrue(report["mrp_ok"])


class TestMRPVerifierHelpers(unittest.TestCase):
    """Test helper methods of MRPVerifier."""

    def setUp(self):
        self.verifier = MRPVerifier()

    def test_compute_crc32(self):
        """Test CRC32 computation."""
        data = b"Hello, World!"
        crc = self.verifier.compute_crc32(data)
        expected = format(zlib.crc32(data) & 0xFFFFFFFF, "08X")
        self.assertEqual(crc, expected)

    def test_compute_sha256(self):
        """Test SHA-256 computation."""
        data = b"Test data"
        sha = self.verifier.compute_sha256(data)
        expected = hashlib.sha256(data).hexdigest()
        self.assertEqual(sha, expected)

    def test_compute_parity_equal_length(self):
        """Test parity computation with equal length inputs."""
        r = b"AAAA"
        g = b"BBBB"
        parity = self.verifier.compute_parity(r, g)
        expected = bytes([a ^ b for a, b in zip(r, g)])
        self.assertEqual(parity, expected)

    def test_compute_parity_r_shorter(self):
        """Test parity computation when R is shorter than G."""
        r = b"AA"
        g = b"BBBB"
        parity = self.verifier.compute_parity(r, g)
        expected = bytes([r[0] ^ g[0], r[1] ^ g[1], g[2], g[3]])
        self.assertEqual(parity, expected)

    def test_minify_json(self):
        """Test JSON minification."""
        data = {"key": "value", "list": [1, 2, 3]}
        result = self.verifier.minify_json(data)
        expected = b'{"key":"value","list":[1,2,3]}'
        self.assertEqual(result, expected)

    def test_build_mrp_header(self):
        """Test MRP header construction."""
        payload_b64 = b"SGVsbG8="  # "Hello" in base64
        header = self.verifier.build_mrp_header("R", payload_b64, include_crc=True)

        self.assertEqual(len(header), 14)
        self.assertEqual(header[0:4], b"MRP1")
        self.assertEqual(header[4:5], b"R")
        self.assertEqual(header[5], 0x01)  # CRC flag

    def test_parse_mrp_header(self):
        """Test MRP header parsing."""
        # Build a header
        payload_b64 = b"SGVsbG8="
        header = self.verifier.build_mrp_header("G", payload_b64, include_crc=True)

        # Parse it back
        parsed = self.verifier.parse_mrp_header(header)

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["magic"], "MRP1")
        self.assertEqual(parsed["channel"], "G")
        self.assertEqual(parsed["flags"], 0x01)
        self.assertEqual(parsed["length"], len(payload_b64))


def run_quick_demo():
    """Run a quick demonstration of MRP verification."""
    print("=" * 60)
    print("MRP Phase-A Verification Demo")
    print("=" * 60)

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test payloads
        r_payload = {
            "type": "primary",
            "data": "Secret message in R channel",
            "timestamp": "2025-01-12T12:00:00Z",
        }

        g_payload = {
            "type": "secondary",
            "data": "Additional data in G channel",
            "metadata": {"version": 1},
        }

        # Compute verification data
        r_min = json.dumps(r_payload, separators=(",", ":")).encode()
        g_min = json.dumps(g_payload, separators=(",", ":")).encode()
        r_b64 = base64.b64encode(r_min)
        g_b64 = base64.b64encode(g_min)

        crc_r = format(zlib.crc32(r_b64) & 0xFFFFFFFF, "08X")
        crc_g = format(zlib.crc32(g_b64) & 0xFFFFFFFF, "08X")
        sha_r = hashlib.sha256(r_b64).hexdigest()

        parity = bytearray(len(g_b64))
        for i in range(len(g_b64)):
            if i < len(r_b64):
                parity[i] = r_b64[i] ^ g_b64[i]
            else:
                parity[i] = g_b64[i]
        parity_b64 = base64.b64encode(parity).decode()

        b_payload = {
            "crc_r": crc_r,
            "crc_g": crc_g,
            "sha256_msg_b64": sha_r,
            "ecc_scheme": "parity",
            "parity_block_b64": parity_b64,
        }

        # Save payloads
        r_path = temp_path / "R.json"
        g_path = temp_path / "G.json"
        b_path = temp_path / "B.json"

        with open(r_path, "w") as f:
            json.dump(r_payload, f, indent=2)
        with open(g_path, "w") as f:
            json.dump(g_payload, f, indent=2)
        with open(b_path, "w") as f:
            json.dump(b_payload, f, indent=2)

        print(f"\nCreated MRP payloads:")
        print(f"  R CRC32: {crc_r}")
        print(f"  G CRC32: {crc_g}")
        print(f"  SHA256:  {sha_r[:32]}...")
        print()

        # Run verification
        verifier = MRPVerifier()
        report = verifier.verify(
            r_payload_path=r_path,
            g_payload_path=g_path,
            b_payload_path=b_path,
        )

        verifier.print_verification_summary()

        print()
        print("=" * 60)
        print(f"Test result: {'PASS' if report['mrp_ok'] else 'FAIL'}")
        print("=" * 60)

        return report["mrp_ok"]


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        success = run_quick_demo()
        sys.exit(0 if success else 1)
    else:
        # Run unit tests
        unittest.main(verbosity=2)
