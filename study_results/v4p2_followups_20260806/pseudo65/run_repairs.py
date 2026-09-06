#!/usr/bin/env python3
"""Run unchanged-card optimizer repeats for masses pending review."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

CONFIGS = {
    "gp_mean": (
        HERE / "configs" / "config_obsUL90_2021_10pct_gpmean_replacement_v4p2.yaml"
    ),
    "functional_form": (
        HERE / "configs" / "config_obsUL90_2021_10pct_funcform_replacement_v4p2.yaml"
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", choices=tuple(CONFIGS), required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--mass-file", type=Path)
    args = parser.parse_args()

    mass_file = args.mass_file or (
        HERE / "derived" / f"{args.lane}_repair_masses.txt"
    )
    masses = [
        float(line.strip())
        for line in mass_file.read_text().splitlines()
        if line.strip()
    ]
    if not masses:
        print(f"{args.lane}: no pending masses")
        return

    environment = dict(os.environ)
    environment.setdefault("MPLCONFIGDIR", "/tmp/codex-mpl")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for index, mass in enumerate(masses, start=1):
        mass_tag = f"m{int(round(1000.0 * mass)):03d}"
        output = (
            HERE
            / "runs"
            / args.lane
            / "repairs"
            / mass_tag
            / f"round_{args.round:02d}"
        )
        output.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "hps_gpr.cli",
            "scan",
            "--config",
            str(CONFIGS[args.lane].relative_to(REPO)),
            "--output-dir",
            str(output.relative_to(REPO)),
            "--mass-min",
            f"{mass:.3f}",
            "--mass-max",
            f"{mass:.3f}",
        ]
        print(
            f"[{index}/{len(masses)}] {args.lane} unchanged-card repeat "
            f"at {1000.0 * mass:.0f} MeV"
        )
        with (output / "scan.log").open("w") as log:
            result = subprocess.run(
                command,
                cwd=REPO,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0 or not (output / "results_single.csv").exists():
            raise RuntimeError(
                f"Repair failed for {args.lane} at {mass:.3f} GeV; "
                f"see {output / 'scan.log'}"
            )


if __name__ == "__main__":
    main()
