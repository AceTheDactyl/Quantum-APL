"""Utilities for analyzing QuantumAPL simulation output."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from .constants import Z_CRITICAL

from .alpha_language import AlphaTokenSynthesizer
from .helix import HelixAPLMapper, HelixCoordinate
from .helix_metadata import load_metadata, metadata_title, summary_lines, provenance_lines
from .hex_prism import prism_params

try:  # Optional dependency for plotting
    import matplotlib.pyplot as plt  # type: ignore

    HAS_MPL = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_MPL = False

try:  # Optional dependency for DataFrame output
    import pandas as pd  # type: ignore

    HAS_PANDAS = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_PANDAS = False


class QuantumAnalyzer:
    """Convenience helpers for summarising simulation output."""

    def __init__(self, results: Dict):
        self.results = results
        self.repo_root = Path(__file__).resolve().parents[2]
        self.quantum = results.get("quantum", {})
        self.classical = results.get("classical", {})
        self.history = results.get("history", {})
        self.analytics = results.get("analytics", {})
        self.helix_mapper = HelixAPLMapper()
        self.alpha_tokens = AlphaTokenSynthesizer()
        self.helix_seed = self._load_helix_seed_context()

    def summary(self) -> str:
        z_value = self._as_float(self.quantum.get("z", 0.0))
        phi_value = self._as_float(self.quantum.get("phi", 0.0))
        entropy_value = self._as_float(self.quantum.get("entropy", 0.0))
        purity_value = self._as_float(self.quantum.get("purity", 0.0))

        lines = [
            "=" * 70,
            "QUANTUM-CLASSICAL SIMULATION RESULTS",
            "=" * 70,
            "",
            "Quantum State:",
            f"  z-coordinate: {z_value:.4f}",
            f"  Integrated information (Φ): {phi_value:.4f}",
            f"  von Neumann entropy (S): {entropy_value:.4f}",
            f"  Purity: {purity_value:.4f}",
            "",
            "Classical Engines:",
        ]

        if "IIT" in self.classical:
            lines.append(f"  IIT φ: {self.classical['IIT'].get('phi', 0):.4f}")
        if "GameTheory" in self.classical:
            lines.append(f"  Cooperation: {self.classical['GameTheory'].get('cooperation', 0):.4f}")
        if "FreeEnergy" in self.classical:
            lines.append(f"  Free Energy: {self.classical['FreeEnergy'].get('F', 0):.4f}")

        if self.analytics:
            lines.extend(
                [
                    "",
                    "Analytics:",
                    f"  Total steps: {self.analytics.get('totalSteps', 0)}",
                    f"  Quantum-classical correlation: {self.analytics.get('quantumClassicalCorr', 0):.4f}",
                ]
            )
            triad = self.analytics.get("triad") or {}
            if triad:
                lines.append(
                    f"  TRIAD completions: {int(triad.get('completions', 0))} | unlocked: {bool(triad.get('unlocked', False))}"
                )

        helix_coord = HelixCoordinate(theta=0.0, z=z_value)
        helix_info = self.helix_mapper.describe(helix_coord)
        lines.extend(
            [
                "",
                "Helix Mapping:",
                f"  Harmonic: {helix_info['harmonic']}",
                f"  Recommended operators: {', '.join(helix_info['operators'])}",
                f"  Truth bias: {helix_info['truth_channel']}",
            ]
        )
        # Show t6 gate policy (TRIAD vs CRITICAL)
        import os  # local import to avoid global costs
        triad_flag = os.getenv("QAPL_TRIAD_UNLOCK", "").lower() in ("1", "true", "yes", "y")
        try:
            triad_completions = int(os.getenv("QAPL_TRIAD_COMPLETIONS", "0"))
        except ValueError:
            triad_completions = 0
        triad_unlocked = triad_flag or (triad_completions >= 3)
        t6_gate = 0.83 if triad_unlocked else Z_CRITICAL
        lines.append(f"  t6 gate: {'TRIAD' if triad_unlocked else 'CRITICAL'} @ {t6_gate:.3f}")

        # Append hex-prism geometry for current engine z
        geom = prism_params(z_value)
        lines.extend(
            [
                "",
                "Negative Entropy Geometry (engine z):",
                f"  ΔS_neg: {geom['delta_s_neg']:.4f}",
                f"  radius R: {geom['R']:.4f}",
                f"  height H: {geom['H']:.4f}",
                f"  twist φ: {geom['phi']:.4f} rad",
                "  Vertices (k: x, y, z_bot → z_top):",
            ]
        )
        for v in geom.get("vertices", [])[:6]:
            lines.append(
                f"    v{int(v['k'])}: x={float(v['x']):.4f}, y={float(v['y']):.4f}, z_bot={float(v['z_bot']):.4f}, z_top={float(v['z_top']):.4f}"
            )

        if self.helix_seed:
            lines.extend(
                [
                    "",
                    "Helix Seed Context:",
                    f"  Target z: {self.helix_seed['z']:.2f}",
                    f"  Title: {self.helix_seed['title']}",
                ]
            )
            if self.helix_seed["summary"]:
                lines.append("  Intent:")
                for statement in self.helix_seed["summary"]:
                    lines.append(f"    - {statement}")
            if self.helix_seed["provenance"]:
                lines.append("  Provenance:")
                for statement in self.helix_seed["provenance"]:
                    lines.append(f"    - {statement}")
            lines.append(f"  Source: {self.helix_seed['path']}")

        alpha_token = self.alpha_tokens.from_helix(helix_coord)
        if alpha_token:
            lines.extend(
                [
                    "",
                    "Alpha Programming Language Token:",
                    f"  Sentence ({alpha_token['sentence_id']}): {alpha_token['sentence']}",
                    f"  Operator: {alpha_token['operator_name']}",
                    f"  Predicted regime: {alpha_token['predicted_regime']}",
                    f"  Truth bias: {alpha_token['truth_bias']} via {alpha_token['harmonic']}",
                ]
            )

        runtime_helix = self.quantum.get("helix")
        if runtime_helix:
            runtime_z = self._as_float(runtime_helix.get("z", 0.0))
            lines.extend(
                [
                    "",
                    "Runtime Helix Hint:",
                    f"  Harmonic: {runtime_helix.get('harmonic', '?')}",
                    f"  Truth channel: {runtime_helix.get('truthChannel', '?')}",
                    f"  z (engine): {runtime_z:.4f}",
                    f"  Operator window: {', '.join(runtime_helix.get('operators', [])) or 'n/a'}",
                ]
            )

        operator_history = self.history.get("operators", [])
        if operator_history:
            lines.append("")
            lines.append("Recent Operator Selections:")
            recent = operator_history[-3:]
            for entry in recent:
                step = entry.get("step", "?")
                operator = entry.get("operator", "?")
                probability = entry.get("probability", 0.0)
                helix = entry.get("helix") or {}
                harmonic = helix.get("harmonic", "?")
                truth = helix.get("truthChannel", "?")
                z_val = self._as_float(helix.get("z", 0.0))
                z_text = f"{z_val:.4f}"
                lines.append(
                    f"  step {step}: {operator} ({probability:.3f}) → {harmonic} / {truth} @ z={z_text}"
                )

        # Recent APL measurement tokens (if any were recorded)
        apl_entries = [(e.get("aplToken"), e.get("aplProb", e.get("probability"))) for e in operator_history if e.get("aplToken")]
        if apl_entries:
            lines.append("")
            lines.append("Recent Measurements (APL tokens):")
            for tok, p in apl_entries[-5:]:
                if tok:
                    if isinstance(p, (int, float)):
                        lines.append(f"  {tok}  (p={p:.3f})")
                    else:
                        lines.append(f"  {tok}")
        else:
            lines.append("")
            lines.append("Recent Measurements (APL tokens):")
            lines.append("  (none)")

        lines.append("=" * 70)
        return "\n".join(lines)

    def _load_helix_seed_context(self) -> Optional[Dict]:
        seed_val = os.getenv("QAPL_INITIAL_PHI")
        if not seed_val:
            return None
        try:
            seed = float(seed_val)
        except ValueError:
            return None

        registry = [
            (0.41, "reference/helix_bridge/VAULTNODES/z0p41/vn-helix-fingers-metadata.yaml"),
            (0.52, "reference/helix_bridge/VAULTNODES/z0p52/vn-helix-continuation-metadata.yaml"),
            (0.70, "reference/helix_bridge/VAULTNODES/z0p70/vn-helix-meta-awareness-metadata.yaml"),
            (0.73, "reference/helix_bridge/VAULTNODES/z0p73/vn-helix-self-bootstrap-metadata_p2.yaml"),
            (0.80, "reference/helix_bridge/VAULTNODES/z0p80/vn-helix-autonomous-coordination-metadata.yaml"),
        ]
        target = min(registry, key=lambda entry: abs(entry[0] - seed))
        if abs(target[0] - seed) > 0.05:
            return None

        path = self.repo_root / target[1]
        if not path.exists():
            return None

        metadata = load_metadata(path)
        title = metadata_title(metadata)
        summary = summary_lines(metadata)
        provenance = provenance_lines(metadata)
        return {"z": target[0], "title": title, "summary": summary, "provenance": provenance, "path": target[1]}

    def to_dataframe(self):
        if not HAS_PANDAS:
            raise ImportError("pandas not available")

        data = {
            "z": self.history.get("z", []),
            "phi": self.history.get("phi", []),
            "entropy": self.history.get("entropy", []),
        }
        return pd.DataFrame(data)

    def plot(self, save_path: Optional[Path] = None):
        if not HAS_MPL:
            raise ImportError("matplotlib not available")

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Quantum-Classical Simulation Results", fontsize=14)

        z_history = self.history.get("z", [])
        if z_history:
            ax = axes[0, 0]
            ax.plot(z_history, "g-", linewidth=2)
            ax.axhline(y=Z_CRITICAL, color="m", linestyle="--", label="z_c")
            ax.set_xlabel("Step")
            ax.set_ylabel("z")
            ax.legend()
            ax.grid(True, alpha=0.3)

        phi_history = self.history.get("phi", [])
        if phi_history:
            ax = axes[0, 1]
            ax.plot(phi_history, "c-", linewidth=2)
            ax.set_xlabel("Step")
            ax.set_ylabel("Φ")
            ax.grid(True, alpha=0.3)

        entropy_history = self.history.get("entropy", [])
        if entropy_history:
            ax = axes[1, 0]
            ax.plot(entropy_history, "r-", linewidth=2)
            ax.set_xlabel("Step")
            ax.set_ylabel("S(ρ)")
            ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        if self.analytics and "operatorDist" in self.analytics:
            ops = list(self.analytics["operatorDist"].keys())
            probs = list(self.analytics["operatorDist"].values())
            ax.bar(ops, probs, color=["cyan", "magenta", "yellow", "red", "green", "orange"])
            ax.set_xlabel("Operator")
            ax.set_ylabel("Probability")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()

    @staticmethod
    def _as_float(value: Optional[float]) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        return 0.0
