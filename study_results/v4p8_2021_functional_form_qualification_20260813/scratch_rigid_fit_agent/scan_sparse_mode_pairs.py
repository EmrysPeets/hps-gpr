#!/usr/bin/env python3
"""Scan two broad Chebyshev correction modes for the six-parameter family.

The threshold constants are frozen from the 1% generalized-gamma
reconnaissance.  Each row fits five shape coordinates plus profiled
normalization (six free parameters total) to 1%, evaluates that frozen shape on
native 10% after normalization scaling, and also records an independently
refitted native-10% comparison.  This is a development scan, not validation.
"""

from __future__ import annotations

import csv
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.chebyshev import chebvander
from scipy.special import expit


HERE = Path(__file__).resolve().parent
FIT_SCRIPT = HERE / "fit_rigid_candidates.py"
FIT_JSON = HERE / "derived/rigid_candidate_fits.json"


def load_module():
    spec = importlib.util.spec_from_file_location("v4p8_mode_scan", FIT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {FIT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    fitmod = load_module()
    prior = json.loads(FIT_JSON.read_text(encoding="utf-8"))["supports"]["40MeV"]
    fixed = {
        "x0": float(prior["x0_selection"]["x0_GeV"]),
        "xt_fixed": float(prior["x0_selection"]["parameters"]["xt"]),
        "w_fixed": float(prior["x0_selection"]["parameters"]["w"]),
        "support_lo": 0.040,
        "support_hi": 0.300,
    }
    one_values, one_centers, _ = fitmod.load_source(*fitmod.SOURCES["one_pct"])
    ten_values, ten_centers, _ = fitmod.load_source(*fitmod.SOURCES["ten_pct"])
    mask = (one_centers >= 0.040) & (one_centers < 0.300)
    ten_mask = (ten_centers >= 0.040) & (ten_centers < 0.300)
    x = one_centers[mask]
    one = one_values[mask]
    ten = ten_values[ten_mask]
    if not np.array_equal(x, ten_centers[ten_mask]):
        raise RuntimeError("source binning mismatch")
    base = prior["fits"]["one_pct"]["raw"]["ggt6_fixed_x0"]["parameters"]
    seed = np.asarray(
        [
            float(base["a"]),
            math.log(float(base["lambda"])),
            math.log(float(base["power"])),
            0.0,
            0.0,
        ]
    )
    rows: list[dict[str, object]] = []
    for mode_a, mode_b in itertools.combinations(range(1, 8), 2):
        def logshape(xvalues, q, constants, ia=mode_a, ib=mode_b):
            a, log_lam, log_power, da, db = q
            lam = math.exp(log_lam)
            power = math.exp(log_power)
            z = xvalues - constants["x0"]
            u = 2.0 * (xvalues - constants["support_lo"]) / (
                constants["support_hi"] - constants["support_lo"]
            ) - 1.0
            matrix = chebvander(u, max(ia, ib))
            turn = np.clip(
                expit(
                    (xvalues - constants["xt_fixed"])
                    / constants["w_fixed"]
                ),
                1e-300,
                1.0,
            )
            return (
                np.log(turn)
                + a * np.log(z)
                - np.power(z / lam, power)
                + da * matrix[:, ia]
                + db * matrix[:, ib]
            )

        family = fitmod.Family(
            name="ggt34_6_fixed_turn",
            n_shape=5,
            bounds=(
                (0.2, 12.0),
                (math.log(0.0005), math.log(0.300)),
                (math.log(0.25), math.log(3.0)),
                (-2.0, 2.0),
                (-2.0, 2.0),
            ),
            seed=seed,
            logshape=logshape,
        )
        one_fit, one_all = fitmod.fit_family(
            x,
            one,
            family,
            fixed,
            n_restarts=4,
            rng_seed=20261000 + 10 * mode_a + mode_b,
        )
        scaled_ten = one_fit.expected * float(np.sum(ten) / np.sum(one_fit.expected))
        ten_fit, ten_all = fitmod.fit_family(
            x,
            ten,
            family,
            fixed,
            seed_override=one_fit.shape,
            n_restarts=4,
            rng_seed=20262000 + 10 * mode_a + mode_b,
        )
        one_metric = fitmod.metrics(one, one_fit.expected, 6)
        scaled_metric = fitmod.metrics(ten, scaled_ten, 1)
        ten_metric = fitmod.metrics(ten, ten_fit.expected, 6)
        rows.append(
            {
                "mode_a": mode_a,
                "mode_b": mode_b,
                "one_pct_pearson_chi2ndf": one_metric["pearson_chi2ndf"],
                "one_pct_deviance_ndf": one_metric["poisson_deviance_ndf"],
                "scaled_shape_ten_pct_pearson_chi2ndf": scaled_metric[
                    "pearson_chi2ndf"
                ],
                "scaled_shape_ten_pct_deviance_ndf": scaled_metric[
                    "poisson_deviance_ndf"
                ],
                "independent_ten_pct_pearson_chi2ndf": ten_metric[
                    "pearson_chi2ndf"
                ],
                "independent_ten_pct_deviance_ndf": ten_metric[
                    "poisson_deviance_ndf"
                ],
                "one_pct_best_objective": one_all[0].objective,
                "one_pct_n_within_1e9": fitmod.restart_summary(one_all)[
                    "n_within_1e-9_objective"
                ],
                "ten_pct_best_objective": ten_all[0].objective,
                "ten_pct_n_within_1e9": fitmod.restart_summary(ten_all)[
                    "n_within_1e-9_objective"
                ],
                "one_pct_a": one_fit.shape[0],
                "one_pct_lambda": math.exp(one_fit.shape[1]),
                "one_pct_power": math.exp(one_fit.shape[2]),
                "one_pct_mode_a_coefficient": one_fit.shape[3],
                "one_pct_mode_b_coefficient": one_fit.shape[4],
            }
        )
    rows.sort(key=lambda row: float(row["scaled_shape_ten_pct_deviance_ndf"]))
    out = HERE / "derived/sparse_mode_pair_scan.csv"
    with out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(out), "rows": len(rows), "best": rows[0]}, indent=2))


if __name__ == "__main__":
    main()
