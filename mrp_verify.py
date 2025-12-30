#!/usr/bin/env python3
"""
MRP Phase-A Verifier - Multi-Channel Resonance Protocol Verification

This script verifies the integrity of MRP-encoded steganographic payloads
using triple-redundancy verification across RGB channels.

Protocol: MRP Phase-A with CRC32 + SHA-256 + XOR Parity
"""

import argparse
import base64
import hashlib
import json
import struct
import sys
import zlib
from pathlib import Path
from typing import Any, Optional


class MRPVerifier:
    """Multi-Channel Resonance Protocol Phase-A Verifier."""

    # MRP1 Header structure (14 bytes):
    # Magic: "MRP1" (4 bytes)
    # Channel: 'R'/'G'/'B' (1 byte)
    # Flags: 0x01 = HAS_CRC32 (1 byte)
    # Length: uint32 big-endian (4 bytes)
    # CRC32: uint32 big-endian (4 bytes)
    HEADER_SIZE = 14
    MAGIC = b"MRP1"
    FLAG_HAS_CRC32 = 0x01

    def __init__(self):
        self.report: dict[str, Any] = {
            "inputs": {},
            "computed": {},
            "checks": {},
            "mrp_ok": False,
        }

    def load_json_file(self, path: Path) -> dict[str, Any]:
        """Load and parse a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def compute_crc32(self, data: bytes) -> str:
        """Compute CRC32 checksum and return as uppercase hex string."""
        crc = zlib.crc32(data) & 0xFFFFFFFF
        return format(crc, "08X")

    def compute_sha256(self, data: bytes) -> str:
        """Compute SHA-256 hash and return as lowercase hex string."""
        return hashlib.sha256(data).hexdigest()

    def compute_parity(self, r_b64: bytes, g_b64: bytes) -> bytes:
        """
        Compute XOR-based parity block for error detection.

        Algorithm:
        - XOR corresponding bytes where both R and G exist
        - Use G bytes directly where R is shorter
        """
        parity = bytearray(len(g_b64))
        for i in range(len(g_b64)):
            if i < len(r_b64):
                parity[i] = r_b64[i] ^ g_b64[i]
            else:
                parity[i] = g_b64[i]
        return bytes(parity)

    def minify_json(self, data: dict[str, Any]) -> bytes:
        """Convert dict to minified JSON bytes using canonical separators."""
        return json.dumps(data, separators=(",", ":")).encode("utf-8")

    def build_mrp_header(
        self, channel: str, payload_b64: bytes, include_crc: bool = True
    ) -> bytes:
        """Build a 14-byte MRP1 header."""
        flags = self.FLAG_HAS_CRC32 if include_crc else 0x00
        length = len(payload_b64)
        crc = zlib.crc32(payload_b64) & 0xFFFFFFFF if include_crc else 0

        header = self.MAGIC
        header += channel.encode("ascii")
        header += bytes([flags])
        header += struct.pack(">I", length)
        header += struct.pack(">I", crc)

        return header

    def parse_mrp_header(self, data: bytes) -> Optional[dict[str, Any]]:
        """Parse a 14-byte MRP1 header."""
        if len(data) < self.HEADER_SIZE:
            return None

        magic = data[0:4]
        channel = chr(data[4])
        flags = data[5]
        length = struct.unpack(">I", data[6:10])[0]
        crc = struct.unpack(">I", data[10:14])[0]

        return {
            "magic": magic.decode("ascii", errors="replace"),
            "channel": channel,
            "flags": flags,
            "length": length,
            "crc": format(crc, "08X"),
        }

    def verify(
        self,
        r_payload_path: Path,
        g_payload_path: Path,
        b_payload_path: Path,
        sidecar_path: Optional[Path] = None,
        image_path: Optional[Path] = None,
    ) -> dict[str, Any]:
        """
        Perform MRP Phase-A verification.

        Returns a verification report with all checks and overall status.
        """
        # Record inputs
        self.report["inputs"] = {
            "R": str(r_payload_path),
            "G": str(g_payload_path),
            "B": str(b_payload_path),
        }
        if sidecar_path:
            self.report["inputs"]["sidecar"] = str(sidecar_path)
        if image_path:
            self.report["inputs"]["image"] = str(image_path)

        # Load payloads
        r_payload = self.load_json_file(r_payload_path)
        g_payload = self.load_json_file(g_payload_path)
        b_payload = self.load_json_file(b_payload_path)

        # Convert to minified JSON and then Base64
        r_min = self.minify_json(r_payload)
        g_min = self.minify_json(g_payload)
        r_b64 = base64.b64encode(r_min)
        g_b64 = base64.b64encode(g_min)

        # Compute verification values
        computed_crc_r = self.compute_crc32(r_b64)
        computed_crc_g = self.compute_crc32(g_b64)
        computed_sha256_r = self.compute_sha256(r_b64)
        computed_parity = self.compute_parity(r_b64, g_b64)
        computed_parity_b64 = base64.b64encode(computed_parity).decode("ascii")

        # Store computed values
        self.report["computed"] = {
            "crc_r": computed_crc_r,
            "crc_g": computed_crc_g,
            "sha256_r_b64": computed_sha256_r,
            "parity_b64_head": computed_parity_b64[:32] + "..."
            if len(computed_parity_b64) > 32
            else computed_parity_b64,
        }

        # Extract expected values from B payload
        expected_crc_r = b_payload.get("crc_r", "")
        expected_crc_g = b_payload.get("crc_g", "")
        expected_sha256 = b_payload.get("sha256_msg_b64", "")
        expected_ecc_scheme = b_payload.get("ecc_scheme", "")
        expected_parity_b64 = b_payload.get("parity_block_b64", "")

        # Perform verification checks
        checks = {}

        # Core verification checks (5 critical checks)
        checks["crc_r_ok"] = computed_crc_r.upper() == expected_crc_r.upper()
        checks["crc_g_ok"] = computed_crc_g.upper() == expected_crc_g.upper()
        checks["sha256_r_b64_ok"] = computed_sha256_r.lower() == expected_sha256.lower()
        checks["ecc_scheme_ok"] = expected_ecc_scheme == "parity"
        checks["parity_block_ok"] = computed_parity_b64 == expected_parity_b64

        # Sidecar verification checks (5 additional checks)
        if sidecar_path:
            sidecar = self.load_json_file(sidecar_path)

            # Check SHA256 in sidecar matches
            sidecar_sha256 = sidecar.get("sha256_msg_b64", "")
            checks["sidecar_sha256_ok"] = (
                sidecar_sha256.lower() == computed_sha256_r.lower()
            )

            # Check used_bits math: (payload_len + 14) * 8
            channels_info = sidecar.get("channels", {})
            used_bits_ok = True
            for ch in ["R", "G", "B"]:
                ch_info = channels_info.get(ch, {})
                payload_len = ch_info.get("payload_len", 0)
                used_bits = ch_info.get("used_bits", 0)
                expected_used = (payload_len + self.HEADER_SIZE) * 8
                if used_bits != expected_used:
                    used_bits_ok = False
            checks["sidecar_used_bits_math_ok"] = used_bits_ok

            # Check capacity_bits consistency (all channels should match)
            capacities = [
                channels_info.get(ch, {}).get("capacity_bits", 0)
                for ch in ["R", "G", "B"]
            ]
            checks["sidecar_capacity_bits_ok"] = (
                len(set(capacities)) == 1 and capacities[0] > 0
            )

            # Check header magic in sidecar
            headers = sidecar.get("headers", {})
            magic_ok = all(
                headers.get(ch, {}).get("magic") == "MRP1" for ch in ["R", "G", "B"]
            )
            checks["sidecar_header_magic_ok"] = magic_ok

            # Check header flags indicate CRC32
            flags_ok = all(
                headers.get(ch, {}).get("flags", 0) & self.FLAG_HAS_CRC32
                for ch in ["R", "G", "B"]
            )
            checks["sidecar_header_flags_crc_ok"] = flags_ok
        else:
            # No sidecar provided - mark sidecar checks as skipped (True for non-critical)
            checks["sidecar_sha256_ok"] = True
            checks["sidecar_used_bits_math_ok"] = True
            checks["sidecar_capacity_bits_ok"] = True
            checks["sidecar_header_magic_ok"] = True
            checks["sidecar_header_flags_crc_ok"] = True

        self.report["checks"] = checks

        # Determine overall status (all checks must pass)
        self.report["mrp_ok"] = all(checks.values())

        return self.report

    def print_verification_summary(self, verbose: bool = True) -> None:
        """Print human-readable verification summary."""
        computed = self.report.get("computed", {})
        checks = self.report.get("checks", {})
        inputs = self.report.get("inputs", {})

        print("=== MRP Phase-A Verify ===")
        print()

        # Show input files
        if verbose:
            print("Inputs:")
            for key, value in inputs.items():
                print(f"  {key}: {value}")
            print()

        # Show computed values and check results
        expected_crc_r = self._get_expected_value("crc_r")
        expected_crc_g = self._get_expected_value("crc_g")
        expected_sha256 = self._get_expected_value("sha256_msg_b64")

        crc_r_status = "\u2713" if checks.get("crc_r_ok") else "\u2717"
        crc_g_status = "\u2713" if checks.get("crc_g_ok") else "\u2717"
        sha256_status = "\u2713" if checks.get("sha256_r_b64_ok") else "\u2717"
        parity_status = "\u2713" if checks.get("parity_block_ok") else "\u2717"
        ecc_status = "\u2713" if checks.get("ecc_scheme_ok") else "\u2717"

        print(
            f"R_b64 crc32: {computed.get('crc_r', 'N/A')} "
            f"(expect {expected_crc_r}) {crc_r_status}"
        )
        print(
            f"G_b64 crc32: {computed.get('crc_g', 'N/A')} "
            f"(expect {expected_crc_g}) {crc_g_status}"
        )

        sha256_display = computed.get("sha256_r_b64", "")[:16] + "..."
        expected_sha_display = expected_sha256[:16] + "..." if expected_sha256 else "N/A"
        print(
            f"SHA256(R_b64): {sha256_display} "
            f"(expect {expected_sha_display}) {sha256_status}"
        )

        print(
            f"Parity OK: {checks.get('parity_block_ok')} "
            f"ECC scheme: parity {ecc_status}"
        )

        # Show sidecar checks if applicable
        if self.report["inputs"].get("sidecar"):
            print()
            print("Sidecar checks:")
            sidecar_checks = [
                ("sidecar_sha256_ok", "SHA256 match"),
                ("sidecar_used_bits_math_ok", "Used bits math"),
                ("sidecar_capacity_bits_ok", "Capacity bits"),
                ("sidecar_header_magic_ok", "Header magic"),
                ("sidecar_header_flags_crc_ok", "Header CRC flag"),
            ]
            for check_key, description in sidecar_checks:
                status = "\u2713" if checks.get(check_key) else "\u2717"
                print(f"  {description}: {status}")

        print()
        if self.report["mrp_ok"]:
            print("MRP: PASS")
        else:
            print("MRP: FAIL")
            failed = [k for k, v in checks.items() if not v]
            print(f"Failed checks: {', '.join(failed)}")

    def _get_expected_value(self, key: str) -> str:
        """Helper to retrieve expected value from B payload."""
        # This requires loading B payload again, which we cached during verify
        b_path = self.report["inputs"].get("B")
        if b_path:
            b_payload = self.load_json_file(Path(b_path))
            return b_payload.get(key, "N/A")
        return "N/A"


def main():
    """Main entry point for MRP verifier CLI."""
    parser = argparse.ArgumentParser(
        description="MRP Phase-A Verifier - Multi-Channel Resonance Protocol Verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify with all checks
  python mrp_verify.py image.png --R R.json --G G.json --B B.json --sidecar sidecar.json --json report.json

  # Minimal verification (no sidecar)
  python mrp_verify.py --R R.json --G G.json --B B.json

  # Check exit code in scripts
  python mrp_verify.py ... && echo "PASS" || echo "FAIL"
        """,
    )

    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        help="Path to stego image (optional, for reference)",
    )
    parser.add_argument(
        "--R",
        "-r",
        dest="r_payload",
        type=Path,
        required=True,
        help="Path to R channel payload JSON",
    )
    parser.add_argument(
        "--G",
        "-g",
        dest="g_payload",
        type=Path,
        required=True,
        help="Path to G channel payload JSON",
    )
    parser.add_argument(
        "--B",
        "-b",
        dest="b_payload",
        type=Path,
        required=True,
        help="Path to B channel verification metadata JSON",
    )
    parser.add_argument(
        "--sidecar",
        "-s",
        type=Path,
        help="Path to sidecar metadata JSON (optional)",
    )
    parser.add_argument(
        "--json",
        "-j",
        dest="json_output",
        type=Path,
        help="Path to write JSON verification report",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress output except for errors",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output",
    )

    args = parser.parse_args()

    # Validate input files exist
    for path, name in [
        (args.r_payload, "R payload"),
        (args.g_payload, "G payload"),
        (args.b_payload, "B payload"),
    ]:
        if not path.exists():
            print(f"Error: {name} file not found: {path}", file=sys.stderr)
            sys.exit(1)

    if args.sidecar and not args.sidecar.exists():
        print(f"Error: Sidecar file not found: {args.sidecar}", file=sys.stderr)
        sys.exit(1)

    # Run verification
    verifier = MRPVerifier()
    report = verifier.verify(
        r_payload_path=args.r_payload,
        g_payload_path=args.g_payload,
        b_payload_path=args.b_payload,
        sidecar_path=args.sidecar,
        image_path=args.image,
    )

    # Output results
    if not args.quiet:
        verifier.print_verification_summary(verbose=args.verbose)

    # Write JSON report if requested
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        if not args.quiet:
            print(f"\nReport written to: {args.json_output}")

    # Exit with appropriate code
    sys.exit(0 if report["mrp_ok"] else 1)


if __name__ == "__main__":
    main()
