#!/usr/bin/env python3
"""Run an isolated 24-start reproducibility audit of the 40 MeV finalist."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
FIT_SCRIPT = HERE / "fit_rigid_candidates.py"
FIT_JSON = HERE / "derived/rigid_candidate_fits.json"


def load_module():
    spec = importlib.util.spec_from_file_location("v4p8_rigid_restart", FIT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {FIT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    fitmod = load_module()
    original = json.loads(FIT_JSON.read_text(encoding="utf-8"))["supports"]["40MeV"]
    fixed = {
        "x0": float(original["x0_selection"]["x0_GeV"]),
        "xt_fixed": float(original["x0_selection"]["parameters"]["xt"]),
        "w_fixed": float(original["x0_selection"]["parameters"]["w"]),
        "support_lo": 0.040,
        "support_hi": 0.300,
    }
    family = fitmod.families(0.040, x0=fixed["x0"])["ggt26_6_fixed_turn"]
    output: dict[str, object] = {
        "status": "exploratory scratch reproducibility audit",
        "family": "ggt26_6_fixed_turn",
        "support_GeV": [0.040, 0.300],
        "n_restarts": 24,
        "sources": {},
    }
    for index, source in enumerate(("one_pct", "ten_pct")):
        values, centers, _ = fitmod.load_source(*fitmod.SOURCES[source])
        mask = (centers >= 0.040) & (centers < 0.300)
        params = original["fits"][source]["raw"]["ggt26_6_fixed_turn"][
            "parameters"
        ]
        seed = np.asarray(
            [
                float(params["a"]),
                math.log(float(params["lambda"])),
                math.log(float(params["power"])),
                float(params["d2"]),
                float(params["d6"]),
            ]
        )
        fit, all_fits = fitmod.fit_family(
            centers[mask],
            values[mask],
            family,
            fixed,
            seed_override=seed,
            n_restarts=24,
            rng_seed=20270000 + index,
        )
        output["sources"][source] = {
            "parameters": fit.parameters,
            "fit_success": fit.success,
            "best_restart": fit.restart,
            "restart_summary": fitmod.restart_summary(all_fits),
            "native_metrics": fitmod.metrics(values[mask], fit.expected, 6),
        }
    out = HERE / "derived/finalist_restart_audit.json"
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(out), **output}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
