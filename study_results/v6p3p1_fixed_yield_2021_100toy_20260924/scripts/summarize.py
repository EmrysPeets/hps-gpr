#!/usr/bin/env python3
"""Validate saved cohort rows and compute conditional statistics; generates no toys.

Every bootstrap replicate resamples the same 100 evaluation background IDs in
all cells. The frozen Gaussian pilot reference is never refitted/resampled.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta

BASE = Path(__file__).resolve().parents[1]
MASSES = list(range(60, 241, 20))
SHAPES = ["gaussian", "mc"]
LEVELS = [0, 1, 3, 5]
MASTER = 63220260924
NTOYS = 100


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def boolean(series):
    values = series.astype(str).str.lower()
    if not values.isin(["true", "false", "1", "0", "1.0", "0.0"]).all():
        raise ValueError(f"Malformed boolean column {series.name}")
    return values.isin(["true", "1", "1.0"])


def clean_json(x):
    if isinstance(x, dict):
        return {str(k): clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def cp_interval(k, n, alpha=.05):
    if n == 0:
        return np.nan, np.nan
    return (0. if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1)),
            1. if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k)))


def sample_stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    sd = float(np.std(values, ddof=1)) if n > 1 else np.nan
    return {"n": n, "mean": float(np.mean(values)) if n else np.nan,
            "sd": sd, "se": sd / np.sqrt(n) if n else np.nan}


def bootstrap_stat(values, indices, kind="mean"):
    sample = np.asarray(values, dtype=float)[indices]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        estimates = np.nanstd(sample, axis=1, ddof=1) if kind == "sd" else np.nanmean(sample, axis=1)
    return estimates


def interval(estimates):
    estimates = estimates[np.isfinite(estimates)]
    if len(estimates) < 2:
        return {"bootstrap_se": np.nan, "bootstrap_ci95_lo": np.nan, "bootstrap_ci95_hi": np.nan,
                "bootstrap_valid_replicates": len(estimates)}
    lo, hi = np.quantile(estimates, [.025, .975])
    return {"bootstrap_se": np.std(estimates, ddof=1), "bootstrap_ci95_lo": lo,
            "bootstrap_ci95_hi": hi, "bootstrap_valid_replicates": len(estimates)}


def add_stats(result, name, values, indices=None, width=False):
    result.update({f"{name}_{key}": val for key, val in sample_stats(values).items()})
    if indices is not None:
        result.update({f"{name}_{key}": val for key, val in interval(bootstrap_stat(values, indices)).items()})
        if width:
            result.update({f"{name}_sd_{key}": val for key, val in interval(bootstrap_stat(values, indices, "sd")).items()})


def read_rows(path, cohort):
    rows = pd.read_csv(path)
    for col in ("fit_valid", "profile_valid"):
        if col in rows:
            rows[col] = boolean(rows[col])
    for col in ("mass_MeV", "toy", "z"):
        rows[col] = rows[col].astype(int)
    if "cohort" in rows and not rows.cohort.eq(cohort).all():
        raise ValueError(f"Incorrect cohort in {path}")
    if rows.duplicated(["mass_MeV", "shape", "z", "toy"]).any():
        raise ValueError(f"Duplicate row keys in {path}")
    required = {(m, s, z, t) for m in MASSES for s in SHAPES
                for z in ([0] if cohort == "pilot" else LEVELS) for t in range(NTOYS)}
    found = set(rows[["mass_MeV", "shape", "z", "toy"]].itertuples(index=False, name=None))
    if found != required:
        raise ValueError(f"{cohort}: missing={len(required-found)}, unexpected={len(found-required)} rows")
    good = rows.fit_valid
    if not np.isfinite(rows.loc[good, ["Ahat", "sigma_postfit"]].to_numpy()).all():
        raise ValueError("Nonfinite coordinate labeled fit_valid")
    if not rows.loc[good, "sigma_postfit"].gt(0).all():
        raise ValueError("Nonpositive error labeled fit_valid")
    if cohort == "evaluation":
        profile_good = rows.profile_valid
        if (profile_good & ~good).any():
            raise ValueError("Profile success without valid free fit")
        if not np.isfinite(rows.loc[profile_good, "q_true"]).all() or (rows.loc[profile_good, "q_true"] < -2e-6).any():
            raise ValueError("Invalid q_true labeled profile_valid")
        if not np.allclose(rows.loc[good, "pull"], (rows.loc[good, "Ahat"] - rows.loc[good, "A_expected"]) / rows.loc[good, "sigma_postfit"], rtol=1e-9, atol=1e-9):
            raise ValueError("Pull does not use expected full-template yield")
        for col, threshold in [("profile_contains68", 1.), ("profile_contains95", 3.841459)]:
            if col in rows:
                supplied = boolean(rows.loc[profile_good, col])
                if not supplied.eq(rows.loc[profile_good, "q_true"].clip(lower=0) <= threshold).all():
                    raise ValueError(f"Containment mismatch: {col}")
        for (mass, z), cell in rows.groupby(["mass_MeV", "z"]):
            if cell.A_expected.nunique() != 1 or cell.s0.nunique() != 1:
                raise ValueError("Expected yield/reference is not common to all toys and shapes")
            if not np.allclose(cell.A_expected, z * cell.s0, rtol=1e-12, atol=1e-12):
                raise ValueError("Expected yield is not z*s0")
        if "background_hash" in rows and rows.groupby("toy").background_hash.nunique().max() != 1:
            raise ValueError("Evaluation background is not shared across all cells")
        counts = ["actual_full", "actual_support", "actual_window", "actual_training", "actual_outside_support"]
        if all(c in rows for c in counts):
            x = rows[counts].to_numpy(dtype=float)
            if not np.isfinite(x).all() or (x < 0).any() or not np.equal(x, np.floor(x)).all():
                raise ValueError("Realized signal counts must be nonnegative integers")
            if not np.array_equal(rows.actual_full, rows.actual_support + rows.actual_outside_support):
                raise ValueError("Full/support/outside count identity fails")
            if not np.array_equal(rows.actual_support, rows.actual_window + rows.actual_training):
                raise ValueError("Support/window/training count identity fails")
            if not (rows.loc[rows.z.eq(0), counts] == 0).all().all():
                raise ValueError("Nonzero null signal")
    return rows.sort_values(["mass_MeV", "shape", "z", "toy"])


def summarize(base=BASE, n_bootstrap=2000):
    results = base / "results"
    pilot = read_rows(results / "pilot_rows.csv", "pilot")
    evaluation = read_rows(results / "evaluation_rows.csv", "evaluation")
    if "background_hash" in pilot and "background_hash" in evaluation:
        if set(pilot.background_hash) & set(evaluation.background_hash):
            raise ValueError("Pilot and evaluation backgrounds overlap")
        if pilot.groupby("toy").background_hash.nunique().max() != 1:
            raise ValueError("Pilot background is not shared across cells")
    reference_path = base / "pilot_reference.json"
    if not reference_path.is_file():
        raise ValueError("Frozen pilot_reference.json is required")
    reference = json.loads(reference_path.read_text())
    indices = np.stack([np.random.default_rng(np.random.SeedSequence([MASTER, 4, 0, rep, 0, 0])).integers(0, NTOYS, NTOYS) for rep in range(n_bootstrap)])
    pilot_summaries = []
    for mass in MASSES:
        g = pilot[(pilot.mass_MeV == mass) & pilot["shape"].eq("gaussian")]
        if not g.fit_valid.all():
            raise ValueError(f"Mass {mass}: Gaussian pilot lacks 100 valid errors")
        s0 = g.sigma_postfit.mean()
        if not np.isclose(reference["masses"][str(mass)]["s0"], s0, rtol=1e-11, atol=1e-11):
            raise ValueError("Frozen reference JSON disagrees with saved Gaussian pilot")
        if not np.allclose(evaluation.loc[evaluation.mass_MeV.eq(mass), "s0"], s0, rtol=1e-11, atol=1e-11):
            raise ValueError("Frozen scale differs from 100 Gaussian pilot error mean")
        for shape in SHAPES:
            cell = pilot[(pilot.mass_MeV == mass) & pilot["shape"].eq(shape)]
            good = cell[cell.fit_valid]
            item = {"mass_MeV": mass, "shape": shape, "attempted": len(cell), "fit_valid": len(good), "s0": s0}
            add_stats(item, "sigma_postfit", good.sigma_postfit)
            add_stats(item, "Ahat", good.Ahat)
            item["sigma_postfit_cv"] = item["sigma_postfit_sd"] / item["sigma_postfit_mean"]
            item["mean_error_over_gaussian_reference"] = item["sigma_postfit_mean"] / s0
            pilot_summaries.append(item)
    arrays = {}
    summaries = []
    for mass in MASSES:
        for shape in SHAPES:
            shape_rows = evaluation[(evaluation.mass_MeV == mass) & evaluation["shape"].eq(shape)]
            null = shape_rows[shape_rows.z.eq(0)].set_index("toy").reindex(range(NTOYS))
            null_A = null.Ahat.where(null.fit_valid).to_numpy()
            for z in LEVELS:
                cell = shape_rows[shape_rows.z.eq(z)].set_index("toy").reindex(range(NTOYS))
                valid = cell.fit_valid.to_numpy()
                pvalid = cell.profile_valid.to_numpy()
                A = float(cell.A_expected.iloc[0])
                s0 = float(cell.s0.iloc[0])
                vals = {"Ahat": cell.Ahat.where(valid).to_numpy(),
                        "bias": (cell.Ahat - A).where(valid).to_numpy(),
                        "pull": cell.pull.where(valid).to_numpy(),
                        "sigma_postfit": cell.sigma_postfit.where(valid).to_numpy(),
                        "error_ratio": (cell.sigma_postfit / s0).where(valid).to_numpy()}
                if z > 0:
                    vals["raw_recovery"] = vals["Ahat"] / A
                    vals["paired_response"] = (vals["Ahat"] - null_A) / A
                item = {"mass_MeV": mass, "shape": shape, "z": z, "A_expected": A, "s0": s0,
                        "attempted": NTOYS, "fit_valid": int(valid.sum()), "profile_valid": int(pvalid.sum()),
                        "profile_valid_fraction": float(pvalid.mean())}
                for col in ("support_fraction", "window_fraction", "training_fraction"):
                    if col in cell:
                        if cell[col].nunique() != 1:
                            raise ValueError("Template fraction changes within a cell")
                        item[col] = float(cell[col].iloc[0])
                for name, values in vals.items():
                    add_stats(item, name, values, indices, width=name == "pull")
                for col in ("actual_full", "actual_support", "actual_window", "actual_training", "actual_outside_support"):
                    if col in cell:
                        add_stats(item, col, cell[col])
                for label, threshold in [("68", 1.), ("95", 3.841459)]:
                    contains = (cell.q_true.clip(lower=0) <= threshold).to_numpy()
                    n = int(pvalid.sum()); k = int((contains & pvalid).sum()); f = NTOYS - n
                    lo, hi = cp_interval(k, n)
                    item.update({f"contain{label}_k": k, f"contain{label}_n": n,
                                 f"contain{label}_fraction": k/n if n else np.nan,
                                 f"contain{label}_cp95_lo": lo, f"contain{label}_cp95_hi": hi,
                                 f"contain{label}_all_attempt_lo": k/NTOYS,
                                 f"contain{label}_all_attempt_hi": (k+f)/NTOYS})
                    vals[f"contain{label}"] = np.where(pvalid, contains.astype(float), np.nan)
                arrays[(mass, shape, z)] = vals
                summaries.append(item)
    paired = []
    for mass in MASSES:
        for z in LEVELS:
            gauss = arrays[(mass, "gaussian", z)]; mc = arrays[(mass, "mc", z)]
            for name in gauss:
                mask = np.isfinite(gauss[name]) & np.isfinite(mc[name])
                g, m = np.where(mask, gauss[name], np.nan), np.where(mask, mc[name], np.nan)
                d = m - g
                row = {"mass_MeV": mass, "z": z, "metric": name, "direction": "mc_minus_gaussian",
                       "complete_pairs": int(mask.sum()), "missing_pairs": int((~mask).sum()),
                       "missing_toy_ids": ",".join(map(str, np.flatnonzero(~mask)))}
                row.update({f"difference_{k}": v for k, v in sample_stats(d).items()})
                row.update(interval(bootstrap_stat(d, indices)))
                paired.append(row)
                if name == "pull":
                    width_row = dict(row, metric="pull_width")
                    width_row["difference_mean"] = np.nanstd(m, ddof=1) - np.nanstd(g, ddof=1)
                    for key in ("difference_sd", "difference_se"):
                        width_row[key] = np.nan
                    width_row.update(interval(bootstrap_stat(m, indices, "sd") - bootstrap_stat(g, indices, "sd")))
                    paired.append(width_row)
    pilot_summary = pd.DataFrame(pilot_summaries)
    summary = pd.DataFrame(summaries)
    comparisons = pd.DataFrame(paired)
    failures = pd.concat([pilot.loc[~pilot.fit_valid].assign(failure_stage="pilot_free_fit"),
                          evaluation.loc[~evaluation.fit_valid].assign(failure_stage="evaluation_free_fit"),
                          evaluation.loc[evaluation.fit_valid & ~evaluation.profile_valid].assign(failure_stage="evaluation_truth_profile")], ignore_index=True)
    failure_cols = [c for c in ["cohort", "mass_MeV", "shape", "z", "toy", "failure_stage", "failure_reason", "fit_valid", "profile_valid", "fit_score", "profile_score", "background_hash", "counts_hash"] if c in failures]
    pilot_summary.to_csv(results / "pilot_summary.csv", index=False)
    summary.to_csv(results / "evaluation_summary.csv", index=False)
    comparisons.to_csv(results / "paired_shape_comparisons.csv", index=False)
    failures[failure_cols].to_csv(results / "failure_ledger.csv", index=False)
    manifest = {"schema_version": 1, "dataset": "2021 10%", "master_seed": MASTER,
                "bootstrap": {"replicates": n_bootstrap, "unit": "whole evaluation toy ID shared across all cells", "keys": "[63220260924, 4, 0, replicate, 0, 0]", "interval": "percentile 95%", "pilot_reference": "held fixed"},
                "pilot_rows": len(pilot), "evaluation_rows": len(evaluation),
                "planned_pilot_cells": 20, "completed_pilot_cells": 20,
                "planned_evaluation_cells": 80, "completed_evaluation_cells": 80,
                "all_planned_toy_ids_present": True,
                "cells_with_100_valid_free_fits": int(summary.fit_valid.eq(NTOYS).sum()),
                "cells_with_100_valid_truth_profiles": int(summary.profile_valid.eq(NTOYS).sum()),
                "cells_with_unresolved_profiles": summary.loc[summary.profile_valid.lt(NTOYS), ["mass_MeV", "shape", "z", "fit_valid", "profile_valid"]].to_dict("records"),
                "pilot_fit_valid": int(pilot.fit_valid.sum()), "evaluation_fit_valid": int(evaluation.fit_valid.sum()),
                "evaluation_profile_valid": int(evaluation.profile_valid.sum()), "failed_rows": len(failures),
                "containment": {"threshold68": 1., "threshold95": 3.841459, "binomial_interval": "95% Clopper-Pearson", "all_attempt_bounds": "[k/100, (k+unresolved)/100]; accounting bounds, not confidence intervals"},
                "source_hashes": {str(p.relative_to(base)): sha(p) for p in [results / "pilot_rows.csv", results / "evaluation_rows.csv", reference_path, Path(__file__).resolve()]},
                "pilot": pilot_summaries, "evaluation": summaries, "paired_shape_comparisons": paired,
                "limitations": "Conditional on pinned background, fixed empirical templates and declared archived-kernel GP procedure. No unconditional coverage, physical-background adequacy, discovery or exclusion claim."}
    (results / "summary.json").write_text(json.dumps(clean_json(manifest), indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k in ["pilot_rows", "evaluation_rows", "pilot_fit_valid", "evaluation_fit_valid", "evaluation_profile_valid", "failed_rows"]}))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_replicates < 100:
        parser.error("Use at least 100 bootstrap replicates")
    summarize(args.base.resolve(), args.bootstrap_replicates)
