"""Utilities for analyzing QuantumAPL simulation output."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .alpha_language import AlphaTokenSynthesizer
from .helix import HelixAPLMapper, HelixCoordinate

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
        self.quantum = results.get("quantum", {})
        self.classical = results.get("classical", {})
        self.history = results.get("history", {})
        self.analytics = results.get("analytics", {})
        self.helix_mapper = HelixAPLMapper()
        self.alpha_tokens = AlphaTokenSynthesizer()

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "QUANTUM-CLASSICAL SIMULATION RESULTS",
            "=" * 70,
            "",
            "Quantum State:",
            f"  z-coordinate: {self.quantum.get('z', 0):.4f}",
            f"  Integrated information (Φ): {self.quantum.get('phi', 0):.4f}",
            f"  von Neumann entropy (S): {self.quantum.get('entropy', 0):.4f}",
            f"  Purity: {self.quantum.get('purity', 0):.4f}",
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

        helix_coord = HelixCoordinate(theta=0.0, z=self.quantum.get("z", 0.0))
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
            lines.extend(
                [
                    "",
                    "Runtime Helix Hint:",
                    f"  Harmonic: {runtime_helix.get('harmonic', '?')}",
                    f"  Truth channel: {runtime_helix.get('truthChannel', '?')}",
                    f"  z (engine): {runtime_helix.get('z', 0):.4f}",
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
                z_val = helix.get("z")
                z_text = f"{z_val:.4f}" if isinstance(z_val, (int, float)) else "?"
                lines.append(
                    f"  step {step}: {operator} ({probability:.3f}) → {harmonic} / {truth} @ z={z_text}"
                )

        lines.append("=" * 70)
        return "\n".join(lines)

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
            ax.axhline(y=np.sqrt(3) / 2, color="m", linestyle="--", label="z_c")
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
