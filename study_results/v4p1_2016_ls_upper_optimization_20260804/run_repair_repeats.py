#!/usr/bin/env python3
"""Repeat unchanged-card fits for the isolated low-LML diagnostic branches."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
V4_WORKTREE = Path(
    "/Users/emryspeets/Desktop/gp_mods/hps-gpr-analysis-note-v4-20260803"
)
CONFIG_DIR = (
    REPO / "study_configs" / "v4p1_2016_ls_upper_optimization_20260804"
)
TARGETS = {
    10: (54, 62, 76, 86, 116),
    12: (43, 125, 145),
    15: (52, 112, 156),
    20: (99, 149, 150),
}
ATTEMPTS_BY_FACTOR = {
    10: (1, 2, 3),
    12: (2, 3),
    15: (2, 3),
    20: (2, 3),
}


def main() -> None:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "MPLCONFIGDIR": "/private/tmp/hps_gpr_2016_ls_repair_repeats_mpl",
            "XDG_CACHE_HOME": "/private/tmp/hps_gpr_2016_ls_repair_repeats_cache",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    records = []
    for factor, masses_mev in TARGETS.items():
        config = (
            CONFIG_DIR
            / f"config_2016full_wide_support_lsupper{factor:02d}.yaml"
        )
        if not config.is_file():
            raise RuntimeError(f"Missing diagnostic card: {config}")
        for mass_mev in masses_mev:
            mass_gev = mass_mev / 1000.0
            for attempt in ATTEMPTS_BY_FACTOR[factor]:
                output = (
                    HERE
                    / f"k{factor:02d}"
                    / "repairs"
                    / f"m{mass_mev:03d}_attempt_{attempt:02d}"
                )
                result = output / "results_single.csv"
                if result.exists():
                    records.append(
                        {
                            "upper_factor": factor,
                            "mass_MeV": mass_mev,
                            "attempt": attempt,
                            "output": str(output.relative_to(REPO)),
                            "status": "existing_not_overwritten",
                        }
                    )
                    continue
                command = [
                    "python3",
                    "-m",
                    "hps_gpr.cli",
                    "scan",
                    "--config",
                    str(config),
                    "--output-dir",
                    str(output),
                    "--mass-min",
                    f"{mass_gev:.3f}",
                    "--mass-max",
                    f"{mass_gev:.3f}",
                ]
                started = time.time()
                subprocess.run(
                    command,
                    cwd=V4_WORKTREE,
                    env=env,
                    check=True,
                )
                records.append(
                    {
                        "upper_factor": factor,
                        "mass_MeV": mass_mev,
                        "attempt": attempt,
                        "command": command,
                        "output": str(output.relative_to(REPO)),
                        "elapsed_seconds": time.time() - started,
                        "status": "completed",
                    }
                )
    manifest = HERE / "repair_repeat_run_manifest.json"
    manifest.write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} repeat-fit records to {manifest}")


if __name__ == "__main__":
    main()
