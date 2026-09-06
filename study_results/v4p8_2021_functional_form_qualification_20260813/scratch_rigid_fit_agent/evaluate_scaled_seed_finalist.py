#!/usr/bin/env python3
"""Evaluate the 1%-shape-frozen, normalization-scaled six-parameter finalist.

The candidate shape is fit only to the 1% source.  For native 10%, all shape
parameters are frozen and only the total normalization is scaled.  This script
keeps native/rebinned Poisson diagnostics and normalization-only signal
absorption in a machine-readable ledger.
"""

from __future__ import annotations

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
REBIN_FACTORS = (1, 5, 20, 40, 80)
MASSES = (0.065, 0.090, 0.120, 0.180, 0.210)


def load_fit_module():
    spec = importlib.util.spec_from_file_location("v4p8_rigid_fit_scaled", FIT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {FIT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    fitmod = load_fit_module()
    payload = json.loads(FIT_JSON.read_text(encoding="utf-8"))
    metric_rows: list[dict[str, object]] = []
    absorption_rows: list[dict[str, object]] = []
    output: dict[str, object] = {
        "status": "exploratory scratch finalist; no toys produced",
        "policy": "fit six-parameter shape to 1% source; freeze shape and scale normalization for 10%",
        "supports": {},
    }
    for support_lo in (0.030, 0.040):
        support_name = f"{int(1000 * support_lo)}MeV"
        record = payload["supports"][support_name]
        fixed = {
            "x0": float(record["x0_selection"]["x0_GeV"]),
            "xt_fixed": float(record["x0_selection"]["parameters"]["xt"]),
            "w_fixed": float(record["x0_selection"]["parameters"]["w"]),
            "support_lo": support_lo,
            "support_hi": 0.300,
        }
        family = fitmod.families(support_lo, x0=fixed["x0"])[
            "ggt26_6_fixed_turn"
        ]
        one_fit = record["fits"]["one_pct"]["raw"]["ggt26_6_fixed_turn"]
        pars = one_fit["parameters"]
        shape_seed = np.asarray(
            [
                float(pars["a"]),
                math.log(float(pars["lambda"])),
                math.log(float(pars["power"])),
                float(pars["d2"]),
                float(pars["d6"]),
            ]
        )
        support_record: dict[str, object] = {
            "n_free_parameters_in_1pct_shape_fit": 6,
            "n_free_parameters_in_10pct_scaled_application": 1,
            "one_pct_shape_parameters": pars,
            "sources": {},
        }
        for source in ("one_pct", "ten_pct"):
            values, centers, _ = fitmod.load_source(*fitmod.SOURCES[source])
            mask = (centers >= support_lo) & (centers < 0.300)
            x = centers[mask]
            observed = values[mask]
            expected, log_a = fitmod.profiled_model(
                x, observed, family, shape_seed, fixed
            )
            source_metrics: dict[str, object] = {}
            npars = 6 if source == "one_pct" else 1
            for factor in REBIN_FACTORS:
                metric = fitmod.metrics(
                    fitmod.rebin_sum(observed, factor),
                    fitmod.rebin_sum(expected, factor),
                    npars,
                )
                source_metrics[f"rebin{factor}"] = metric
                metric_rows.append(
                    {
                        "support_low_MeV": 1000 * support_lo,
                        "source": source,
                        "shape_source": "one_pct",
                        "shape_refit": source == "one_pct",
                        "n_free_parameters": npars,
                        "rebin_factor": factor,
                        "bin_width_MeV": 1000
                        * factor
                        * float(np.median(np.diff(x))),
                        **metric,
                    }
                )
            support_record["sources"][source] = {
                "profiled_log_normalization": log_a,
                "metrics": source_metrics,
            }

            if support_lo == 0.040:
                total = float(np.sum(expected))
                for mass in MASSES:
                    resolution = float(fitmod.sigma_2021(np.asarray([mass]))[0])
                    template = np.exp(-0.5 * ((x - mass) / resolution) ** 2)
                    template /= float(np.sum(template))
                    fisher = float(np.sum(template * template / expected))
                    # With shape fixed, a refit of normalization alone changes
                    # expected by expected*Ainj/total.  The fraction below is
                    # independent of the chosen injected amplitude.
                    poisson_projection_fraction = 1.0 / (total * fisher)
                    window = np.abs(x - mass) <= 2.25 * resolution
                    window_fraction = float(np.sum(expected[window]) / total) / float(
                        np.sum(template[window])
                    )
                    absorption_rows.append(
                        {
                            "source": source,
                            "support_low_MeV": 40.0,
                            "mass_MeV": 1000 * mass,
                            "resolution_MeV": 1000 * resolution,
                            "fit_policy": "normalization_only_shape_frozen",
                            "poisson_metric_absorption_fraction": poisson_projection_fraction,
                            "window_2p25sigma_absorption_fraction": window_fraction,
                        }
                    )
        output["supports"][support_name] = support_record

    metrics_csv = HERE / "derived/scaled_one_pct_seed_metrics.csv"
    absorption_csv = HERE / "derived/scaled_one_pct_seed_absorption.csv"
    output_json = HERE / "derived/scaled_one_pct_seed_finalist.json"
    with metrics_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    with absorption_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(absorption_rows[0]))
        writer.writeheader()
        writer.writerows(absorption_rows)
    output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "metrics_csv": str(metrics_csv),
                "absorption_csv": str(absorption_csv),
                "json": str(output_json),
                "max_normalization_only_absorption": max(
                    float(row["poisson_metric_absorption_fraction"])
                    for row in absorption_rows
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
