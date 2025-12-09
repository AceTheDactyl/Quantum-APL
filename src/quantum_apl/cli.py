"""Command-line interface for the QuantumAPL Python tooling."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analyzer import QuantumAnalyzer
from .engine import QuantumAPLEngine


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantum-Classical APL Simulation (Python Interface)")
    parser.add_argument("--steps", type=int, default=100, help="Number of simulation steps (default: 100)")
    parser.add_argument(
        "--mode",
        choices=["unified", "quantum_only", "test"],
        default="unified",
        help="Simulation mode (default: unified)",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose Node.js output")
    parser.add_argument("--plot", action="store_true", help="Generate matplotlib plots (if available)")
    parser.add_argument("--output", type=Path, help="Save results JSON to path")
    parser.add_argument("--js-dir", type=Path, help="Directory containing JS files (defaults to repo root)")
    return parser


def _execute(args: argparse.Namespace) -> int:
    engine = QuantumAPLEngine(js_dir=args.js_dir)
    print(f"Running {args.mode} simulation with {args.steps} steps...")
    results = engine.run_simulation(steps=args.steps, verbose=args.verbose, mode=args.mode)

    if args.output:
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Results saved to {args.output}")

    if "raw_output" not in results:
        analyzer = QuantumAnalyzer(results)
        print("\n" + analyzer.summary())
        if args.plot:
            try:
                analyzer.plot()
            except ImportError as exc:
                print(exc)
    else:
        print(results["raw_output"])

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return _execute(args)


def run_simulation() -> int:
    return main()


def run_tests() -> int:
    parser = argparse.ArgumentParser(description="Run the QuantumAPL Node.js test suite")
    parser.add_argument("--js-dir", type=Path, help="Directory containing JS files (defaults to repo root)")
    parser.add_argument("--verbose", action="store_true", help="Verbose Node.js output")
    parser.add_argument("--steps", type=int, default=100, help="Step count forwarded to the runner")
    args = parser.parse_args()
    args.mode = "test"
    args.plot = False
    args.output = None
    return _execute(args)


def analyze(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze saved QuantumAPL results")
    parser.add_argument("input", type=Path, help="JSON results file")
    parser.add_argument("--plot", action="store_true", help="Generate matplotlib plots")
    args = parser.parse_args(argv)

    data = json.loads(args.input.read_text(encoding="utf-8"))
    analyzer = QuantumAnalyzer(data)
    print(analyzer.summary())
    if args.plot:
        try:
            analyzer.plot()
        except ImportError as exc:
            print(exc)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
