#!/usr/bin/env python3
"""Run the frozen v4.9.12 release pipeline in its validated order."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def run(*arguments: str, cwd: Path = REPO) -> None:
    command = [sys.executable, *arguments]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--skip-harvard-export",
        action="store_true",
        help="stop after the analysis release attestation",
    )
    args = parser.parse_args()

    run("-m", "pytest", "-q", str(HERE / "tests/test_bounded_tildeq_cls.py"))
    run("-m", "pytest", "-q", str(HERE / "tests/test_piecewise_cached_solver.py"))
    run("-m", "pytest", "-q", str(REPO / "tests/test_profiled_cls.py"))
    run(str(HERE / "assemble_release_inputs.py"))
    run(str(HERE / "run_final_combinations.py"), "--workers", str(max(1, args.workers)))
    run(str(HERE / "audit_conditioning_impact.py"))
    run(str(HERE / "make_peak_extraction.py"))
    run(str(HERE / "make_figures.py"))
    run(str(HERE / "validate_release.py"))
    if not args.skip_harvard_export:
        run(str(HERE / "export_harvard_selected_results.py"))


if __name__ == "__main__":
    main()
