#!/usr/bin/env python3
"""Read saved v4.9.13 results and validation toys; never generate or fit toys."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import beta, kurtosis, skew

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CAL = ROOT / "study_results/v4p9p13_calibration_20260905"
SUMMARY = CAL / "summary"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def interval(count, size):
    return [float(beta.ppf(.025, count, size-count+1)) if count else 0.,
            float(beta.ppf(.975, count+1, size-count)) if count < size else 1.]


def records(frame):
    return json.loads(frame.to_json(orient="records", double_precision=15))


def main():
    observed = pd.read_csv(SUMMARY / "observed_calibrated_limits.csv")
    truth_limits = pd.read_csv(SUMMARY / "truth_specific_limits.csv")
    validation = pd.read_csv(SUMMARY / "validation_summary.csv")
    original = json.loads((SUMMARY / "calibration_summary.json").read_text())
    assert len(observed) == 456 and original["complete_grid"]
    assert not observed.duplicated(["scope_key", "mass_MeV"]).any()
    source_hashes = {}
    for filename in ("observed_calibrated_limits.csv", "truth_specific_limits.csv", "validation_summary.csv"):
        path = SUMMARY / filename
        digest = sha(path)
        assert digest == original["output_sha256"][filename], filename
        source_hashes[str(path.relative_to(ROOT))] = digest
    assert observed.numerical_audit_passed.all()

    common = {64, 65, 66, 71, 73, 74, 75, 77, 78}
    extra = {"individual_2015_full": {22, 39, 50, 51},
             "individual_2016_full": {90, 118},
             "individual_2021_10pct": {182}, "all_2015_2016_2021": set()}
    selected = observed[[int(r.mass_MeV) in common | extra[r.scope_key]
                         for r in observed.itertuples()]].copy()
    selected["profiled_calibrated_over_parent_asymptotic"] = (
        selected.eps2_profiled_calibrated / selected.eps2_current_display)
    selected["profiled_calibrated_over_same_solver_asymptotic"] = (
        selected.eps2_profiled_calibrated / selected.eps2_profiled_asymptotic)
    rows = []
    for point in selected.itertuples():
        path = Path(point.checkpoint_path).parent / "validation_toys.csv.gz"
        data = pd.read_csv(path)
        digest = sha(path)
        source_hashes[str(path.relative_to(ROOT))] = digest
        for (truth, method), sample in data[data.strength == 0].groupby(["truth", "method"]):
            ref = validation[(validation.scope_key == point.scope_key)
                             & (validation.mass_MeV == point.mass_MeV)
                             & (validation.strength == 0)
                             & (validation.truth == truth)
                             & (validation.method == method)].iloc[0]
            assert len(sample) == int(ref.n) == 500
            assert digest == ref.validation_toy_ledger_sha256
            assert np.isclose(sample.Ahat.mean(), ref.Ahat_mean, rtol=1e-10, atol=1e-7)
            r = sample.signed_r.to_numpy()
            r_obs = getattr(point, f"signed_r_{method}_asymptotic")
            positive = r_obs > 0
            tail_count = int(np.sum(r >= r_obs)) if positive else len(r)
            row = dict(scope_key=point.scope_key, mass_MeV=int(point.mass_MeV),
                       truth=truth, method=method, n=len(r), observed_signed_r=float(r_obs),
                       null_signed_r_mean=float(r.mean()), null_signed_r_sd=float(r.std(ddof=1)),
                       null_signed_r_skew=float(skew(r, bias=False)),
                       null_signed_r_excess_kurtosis=float(kurtosis(r, bias=False)),
                       null_signed_r_q05=float(np.quantile(r, .05)),
                       null_signed_r_q50=float(np.quantile(r, .50)),
                       null_signed_r_q95=float(np.quantile(r, .95)),
                       direct_tail_count=tail_count, direct_tail_fraction=tail_count/len(r),
                       direct_tail_ci95_low=interval(tail_count, len(r))[0],
                       direct_tail_ci95_high=interval(tail_count, len(r))[1],
                       signal_bias_sigma=float(ref.signal_bias_sigma),
                       linearized_zero_noise_bias_sigma=float(ref.linearized_zero_noise_bias_sigma),
                       linearized_sampling_sd_sigma=float(ref.linearized_sampling_sd_sigma),
                       raw_local_rejection_count=int(ref.raw_local_rejection_count),
                       calibrated_local_rejection_count=int(ref.local_rejection_count),
                       observed_q0_is_zero=not positive,
                       source_toy_ledger=str(path.relative_to(ROOT)))
            rows.append(row)
    nulls = pd.DataFrame(rows)
    selected.to_csv(HERE / "selected_observed_points.csv", index=False)
    nulls.to_csv(HERE / "selected_null_diagnostics.csv", index=False)
    selected_keys = set(zip(selected.scope_key, selected.mass_MeV))
    cells = validation[[tuple(pair) in selected_keys
                        for pair in zip(validation.scope_key, validation.mass_MeV)]].copy()
    columns = ["scope_key", "mass_MeV", "truth", "method", "strength", "n", "Atrue",
               "raw_local_rejection_count", "local_rejection_count", "raw_exclusion_count",
               "exclusion_count", "signal_bias_sigma", "checkpoint_path"]
    cells[columns].to_csv(HERE / "selected_validation_decisions.csv", index=False)

    metrics = []
    for scope, sample in observed.groupby("scope_key", sort=False):
        ratios = sample.eps2_profiled_calibrated / sample.eps2_current_display
        metrics.append(dict(scope_key=scope, mass_points=len(sample),
                            median_profiled_calibrated_over_parent_asymptotic=float(np.median(ratios)),
                            minimum_profiled_calibrated_over_parent_asymptotic=float(ratios.min()),
                            maximum_profiled_calibrated_over_parent_asymptotic=float(ratios.max()),
                            median_fixed_over_profiled_calibrated=float(sample.ratio_fixed_over_profiled_calibrated.median()),
                            profiled_resolved=int((sample.status_profiled == "resolved").sum()),
                            fixed_resolved=int((sample.status_fixed == "resolved").sum())))
    flag_counts = []
    for (method, truth), sample in validation.groupby(["method", "truth"]):
        for label, mask, column in [("local", sample.strength == 0, "raw_local_holm_reject_0p05"),
                                    ("exclusion", sample.strength > 0, "raw_exclusion_holm_reject_0p05")]:
            s = sample[mask]
            flag_counts.append(dict(method=method, truth=truth, test=label,
                                    tested_cells=len(s), raw_flagged_cells=int(s[column].sum()),
                                    calibrated_flagged_cells=int(s[("local" if label == "local" else "exclusion") + "_holm_reject_0p05"].sum())))
    m66 = observed[(observed.scope_key == "all_2015_2016_2021") & (observed.mass_MeV == 66)]
    key_null = nulls[(nulls.scope_key == "all_2015_2016_2021") & (nulls.mass_MeV == 66) & (nulls.method == "profiled")]
    key_truth = truth_limits[(truth_limits.scope_key == "all_2015_2016_2021") & (truth_limits.mass_MeV == 66) & (truth_limits.method == "profiled")]
    key_power = cells[(cells.scope_key == "all_2015_2016_2021") & (cells.mass_MeV == 66) & (cells.method == "profiled")]
    for path in [CAL / "calibration_core.py", CAL / "run_calibration.py", CAL / "collect_results.py",
                 CAL / "PROTOCOL.md", CAL / "history_review.md", CAL / "note/reverse_truth_validation.tex",
                 ROOT / "study_results/v4p9p7_2016_support_combined_100toy_20260902/SCIENTIFIC_SCOPE_CLARIFICATION.md"]:
        source_hashes[str(path.relative_to(ROOT))] = sha(path)
    result = dict(schema_version=1, study="Independent read-only HEP statistical review of v4.9.13",
                  no_new_toys_or_fits=True, selected_observed_points=len(selected),
                  selected_null_cells=len(nulls), selected_validation_cells=len(cells),
                  source_summary_hashes_and_selected_toy_moments_verified=True,
                  scope_metrics=metrics, raw_vs_calibrated_validation_flags=flag_counts,
                  combined_66_observed=records(m66)[0], combined_66_truth_specific=records(key_truth),
                  combined_66_null_diagnostics=records(key_null), combined_66_validation=records(key_power[columns]),
                  endpoint_status_counts=original["endpoint_status_counts"],
                  p0_status_counts=original["p0_status_counts"], source_sha256=source_hashes,
                  boundaries=["Selected points are retrospective diagnostics, not independent model-selection controls.",
                              "Direct 500-toy intervals condition on the fixed generating truth and procedure.",
                              "Decision counts condition on the selected finite calibration bank; no bank MC uncertainty is propagated.",
                              "Mean offsets under generating truths are not measurements of bias in observed data.",
                              "Observed-limit ratios are not expected-limit or global-discovery sensitivity.",
                              "Null moments and skew/kurtosis are descriptive finite-sample checks, not Gaussian-tail certification."])
    (HERE / "diagnosis_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"selected_points": len(selected), "null_cells": len(nulls),
                      "validation_cells": len(cells), "verified": True}))


if __name__ == "__main__":
    main()
