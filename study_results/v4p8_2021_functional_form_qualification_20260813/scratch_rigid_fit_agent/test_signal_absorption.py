#!/usr/bin/env python3
"""Asimov signal-absorption diagnostic for the sparse six-parameter candidate.

This is an influence diagnostic, not a coverage or sensitivity calculation.
The fitted analytic mean is perturbed by a Gaussian signal whose count amplitude
is expressed in local matched-filter standard deviations, then the same
background-only family is refit.  The Poisson-metric projection of the fitted
background change onto the signal template measures absorption.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
FIT_SCRIPT = HERE / "fit_rigid_candidates.py"
FIT_JSON = HERE / "derived/rigid_candidate_fits.json"
MASSES = (0.065, 0.090, 0.120, 0.180, 0.210)
STRENGTHS = (1.0, 3.0, 5.0)
SUPPORT_LO = 0.040
SUPPORT_HI = 0.300


def load_fit_module():
    spec = importlib.util.spec_from_file_location("v4p8_rigid_fit", FIT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {FIT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restarts", type=int, default=8)
    args = parser.parse_args()

    fitmod = load_fit_module()
    payload = json.loads(FIT_JSON.read_text(encoding="utf-8"))
    record = payload["supports"]["40MeV"]
    fixed = {
        "x0": float(record["x0_selection"]["x0_GeV"]),
        "xt_fixed": float(record["x0_selection"]["parameters"]["xt"]),
        "w_fixed": float(record["x0_selection"]["parameters"]["w"]),
        "support_lo": SUPPORT_LO,
        "support_hi": SUPPORT_HI,
    }
    family = fitmod.families(SUPPORT_LO, x0=fixed["x0"])[
        "ggt26_6_fixed_turn"
    ]

    rows: list[dict[str, object]] = []
    details: dict[str, object] = {
        "status": "Asimov influence diagnostic only; not coverage or sensitivity",
        "support_GeV": [SUPPORT_LO, SUPPORT_HI],
        "masses_GeV": list(MASSES),
        "strengths_matched_filter_sigma": list(STRENGTHS),
        "sources": {},
    }
    for source_index, source in enumerate(("one_pct", "ten_pct")):
        values, centers, _ = fitmod.load_source(*fitmod.SOURCES[source])
        mask = (centers >= SUPPORT_LO) & (centers < SUPPORT_HI)
        x = centers[mask]
        raw_fit = record["fits"][source]["raw"]["ggt26_6_fixed_turn"]
        pars = raw_fit["parameters"]
        seed = np.asarray(
            [
                float(pars["a"]),
                math.log(float(pars["lambda"])),
                math.log(float(pars["power"])),
                float(pars["d2"]),
                float(pars["d6"]),
            ],
            dtype=float,
        )
        base, _ = fitmod.profiled_model(
            x, values[mask], family, seed, fixed
        )
        source_details: list[dict[str, object]] = []
        for mass_index, mass in enumerate(MASSES):
            resolution = float(fitmod.sigma_2021(np.asarray([mass]))[0])
            template = np.exp(-0.5 * ((x - mass) / resolution) ** 2)
            template /= float(np.sum(template))
            fisher = float(np.sum(template * template / base))
            sigma_amplitude = 1.0 / math.sqrt(fisher)
            for strength in STRENGTHS:
                injected_amplitude = strength * sigma_amplitude
                target = base + injected_amplitude * template
                refit, all_refits = fitmod.fit_family(
                    x,
                    target,
                    family,
                    fixed,
                    seed_override=seed,
                    n_restarts=args.restarts,
                    rng_seed=(
                        20260813
                        + 100000 * source_index
                        + 1000 * mass_index
                        + int(10 * strength)
                    ),
                )
                delta = refit.expected - base
                projected_absorbed = float(
                    np.sum(template * delta / base) / fisher
                )
                absorption_fraction = projected_absorbed / injected_amplitude
                window = np.abs(x - mass) <= 2.25 * resolution
                window_absorbed = float(np.sum(delta[window]))
                window_injected = float(injected_amplitude * np.sum(template[window]))
                window_fraction = window_absorbed / window_injected
                restart_summary = fitmod.restart_summary(all_refits)
                row = {
                    "source": source,
                    "mass_MeV": 1000.0 * mass,
                    "resolution_MeV": 1000.0 * resolution,
                    "strength_matched_filter_sigma": strength,
                    "injected_count_amplitude": injected_amplitude,
                    "projected_absorbed_count_amplitude": projected_absorbed,
                    "poisson_metric_absorption_fraction": absorption_fraction,
                    "window_2p25sigma_absorption_fraction": window_fraction,
                    "objective_best": restart_summary["objective_best"],
                    "objective_spread": restart_summary["objective_spread"],
                    "n_restarts": restart_summary["n_restarts"],
                    "n_within_1e-9_objective": restart_summary[
                        "n_within_1e-9_objective"
                    ],
                    "fit_success": refit.success,
                }
                rows.append(row)
                source_details.append(row)
        details["sources"][source] = source_details

    out_csv = HERE / "derived/signal_absorption_diagnostic.csv"
    out_json = HERE / "derived/signal_absorption_diagnostic.json"
    with out_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out_json.write_text(json.dumps(details, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "csv": str(out_csv),
                "json": str(out_json),
                "rows": len(rows),
                "max_absorption_fraction": max(
                    abs(float(row["poisson_metric_absorption_fraction"]))
                    for row in rows
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
