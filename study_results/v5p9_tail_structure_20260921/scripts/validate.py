"""Independent deterministic QA for the v5.9 exterior-tail study.

This script owns only QA artifacts.  It never changes scan results, chooses a
shape from data, refits a GP kernel, generates toys, or repairs failed rows.
Run ``python scripts/validate.py --shapes-only`` before scan completion; run
without that option for the complete numerical and provenance checks.
"""
from pathlib import Path
import argparse
import json
import math
import os
import sys

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import ndtr

B = Path(__file__).resolve().parents[1]
QA = B / "qa"
A = 2.25
H = 0.5
SHAPES = [("gaussian", 1.0)] + [
    (family, kappa) for family in ("dilation", "curvature")
    for kappa in (1.1, 1.2, 1.3)]
WINDOWS = {"primary": 2.25, "guard": 4.5}
DATA = {year: dict(np.load(B / "inputs" / ("spectrum_" + year + ".npz")))
        for year in ("2015", "2016", "2021")}
DOMAINS = {year: list(range(int(d["masses"][0]),
                           (100 if year == "2015" else int(d["masses"][-1])) + 1))
           for year, d in DATA.items()}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def reference_density(z, kappa, family):
    """Independent scalar expression, deliberately not tail_model.density."""
    z = abs(float(z))
    if z <= A or family == "gaussian" or kappa == 1.0:
        return math.exp(-0.5 * z * z)
    t = z - A
    if family == "dilation":
        mapped = A + t / (1.0 + (kappa - 1.0) * (1.0 - math.exp(-t / H)))
        return math.exp(-0.5 * mapped * mapped)
    if family == "curvature":
        return math.exp(-0.5 * A * A - A * t - 0.5 * (t / kappa) ** 2)
    raise ValueError("Undeclared shape family: " + str(family))


def integral(lo, hi, kappa, family):
    """Adaptive quadrature split explicitly at both core-tail boundaries."""
    cuts = [float(lo)] + [v for v in (-A, A) if lo < v < hi] + [float(hi)]
    return sum(quad(reference_density, left, right, args=(kappa, family),
                    epsabs=2e-13, epsrel=2e-12, limit=150)[0]
               for left, right in zip(cuts[:-1], cuts[1:]))


def mass_sigma(year, mass):
    return float(np.polynomial.polynomial.polyval(mass / 1000., DATA[year]["sigma_coeffs"]))


def geometry(year, mass, window):
    d = DATA[year]
    width = WINDOWS[window] * mass_sigma(year, mass)
    low, high = mass / 1000. - width, mass / 1000. + width
    mask = (d["x"] >= low) & (d["x"] <= high)
    return dict(mask=mask, low=low, high=high,
                left=int(np.count_nonzero(d["x"] < low)),
                right=int(np.count_nonzero(d["x"] > high)))


def shape_checks(check):
    from tail_model import density, weights
    shape_rows, bin_rows, geometry_rows = [], [], []
    grid = np.r_[np.linspace(-15., 15., 12001), -A, A]
    max_density_error = 0.
    continuity_error = 0.
    derivative_error = 0.
    for family, kappa in SHAPES:
        implementation_family = "dilation" if family == "gaussian" else family
        actual = np.asarray(density(grid, kappa, family=implementation_family), float)
        reference = np.array([reference_density(v, kappa, family) for v in grid])
        max_density_error = max(max_density_error, float(np.max(abs(actual - reference))))
        check(np.all(np.isfinite(actual)) and np.min(actual) >= 0,
              "density_positive_" + family + "_" + str(kappa))
        ordered = np.asarray(density(np.linspace(0., 15., 6001), kappa,
                                     family=implementation_family), float)
        check(np.max(np.diff(ordered)) <= 1e-14,
              "density_monotone_" + family + "_" + str(kappa))
        h = 1e-6
        values = np.asarray(density(np.array([A-h, A, A+h]), kappa,
                                    family=implementation_family), float)
        continuity_error = max(continuity_error, float(np.max(abs(values - values[1]))))
        dl, dr = (values[1] - values[0]) / h, (values[2] - values[1]) / h
        derivative_error = max(derivative_error, float(abs(dl-dr)))
        full = 2. * integral(0., np.inf, kappa, family)
        inside = integral(-A, A, kappa, family)
        second = 2. * quad(lambda z: z*z*reference_density(z, kappa, family),
                          0., np.inf, points=None, epsabs=2e-11,
                          epsrel=2e-11, limit=150)[0]
        shape_rows.append(dict(family=family, kappa=kappa,
                               full_line_integral_reference=full,
                               core_fraction_reference=inside/full,
                               tail_fraction_reference=1.-inside/full,
                               rms_sigma_reference=math.sqrt(second/full)))

    check(max_density_error < 2e-13, "density_independent_formula",
          max_abs_error=max_density_error)
    check(continuity_error < 1e-6, "density_boundary_continuity",
          max_near_boundary_change=continuity_error)
    check(derivative_error < 2e-6, "density_first_derivative_continuity",
          max_finite_difference_discrepancy=derivative_error)

    largest_weight_error = 0.
    largest_normalization_error = 0.
    largest_gaussian_error = 0.
    largest_metadata_error = 0.
    largest_interior_nonproportionality = 0.
    for year, masses in DOMAINS.items():
        anchors = sorted(set([masses[0], masses[len(masses)//2], masses[-1]]))
        for mass in anchors:
            d = DATA[year]
            sigma = mass_sigma(year, mass)
            zedges = (d["edges"] - mass/1000.) / sigma
            gaussian = np.diff(ndtr(zedges))
            gaussian /= gaussian.sum()
            mask = geometry(year, mass, "primary")["mask"]
            interior = (zedges[:-1] >= -A) & (zedges[1:] <= A) & mask
            straddle = (((zedges[:-1] < -A) & (zedges[1:] > -A)) |
                        ((zedges[:-1] < A) & (zedges[1:] > A)))
            probes = {0, len(gaussian)-1}
            probes.update(np.flatnonzero(straddle))
            for target in (-6., -4.5, -A, -1., 0., 1., A, 4.5, 6.):
                i = int(np.clip(np.searchsorted(zedges, target)-1, 0, len(gaussian)-1))
                probes.update(j for j in (i-1, i, i+1) if 0 <= j < len(gaussian))
            for family, kappa in SHAPES:
                api_family = "dilation" if family == "gaussian" else family
                probability, metadata = weights(year, mass, kappa, family=api_family)
                probability = np.asarray(probability, float)
                check(probability.shape == gaussian.shape and np.all(np.isfinite(probability))
                      and np.min(probability) >= 0,
                      "bin_weights_" + year + "_" + str(mass) + "_" + family + "_" + str(kappa))
                largest_normalization_error = max(largest_normalization_error,
                                                   abs(float(probability.sum())-1.))
                norm = integral(zedges[0], zedges[-1], kappa, family)
                full = 2. * integral(0., np.inf, kappa, family)
                expected_meta = dict(normalization_integral=norm, full_line_integral=full,
                                     outside_support_fraction=1.-norm/full)
                for name, expected in expected_meta.items():
                    error = abs(float(metadata[name])-expected)
                    largest_metadata_error = max(largest_metadata_error, error)
                if kappa == 1.:
                    largest_gaussian_error = max(largest_gaussian_error,
                                                float(np.max(abs(probability-gaussian))))
                gaussian_norm = integral(zedges[0], zedges[-1], 1., "gaussian")
                scale = gaussian_norm / norm
                departures = probability - gaussian * scale
                interior_error = float(np.max(abs(departures[interior]), initial=0.))
                largest_interior_nonproportionality = max(largest_interior_nonproportionality,
                                                          interior_error)
                geometry_rows.append(dict(year=year, mass_MeV=mass, family=family, kappa=kappa,
                    primary_bins=int(mask.sum()), wholly_core_bins=int(interior.sum()),
                    straddling_bins=int(straddle.sum()),
                    fitted_straddling_bins=int(np.count_nonzero(mask & straddle)),
                    exact_core_scale=scale, interior_max_abs_departure=interior_error,
                    fitted_max_abs_departure=float(np.max(abs(departures[mask]), initial=0.)),
                    fitted_boundary_departure_L1=float(np.sum(abs(departures[mask & straddle]))),
                    actual_primary_probability=float(probability[mask].sum()),
                    purely_scaled_primary_probability=float((gaussian[mask]*scale).sum())))
                for index in sorted(probes):
                    reference = integral(zedges[index], zedges[index+1], kappa, family)/norm
                    error = abs(float(probability[index])-reference)
                    largest_weight_error = max(largest_weight_error, error)
                    bin_rows.append(dict(year=year, mass_MeV=mass, family=family, kappa=kappa,
                        bin_index=index, z_low=zedges[index], z_high=zedges[index+1],
                        straddles_core_boundary=bool(straddle[index]),
                        probability_implementation=float(probability[index]),
                        probability_adaptive_quadrature=reference, absolute_error=error))
    check(largest_normalization_error < 2e-12, "unit_bin_normalization",
          max_abs_error=largest_normalization_error)
    check(largest_weight_error < 5e-10, "independent_adaptive_bin_quadrature",
          max_abs_error=largest_weight_error, checked_bins=len(bin_rows))
    check(largest_gaussian_error < 2e-12, "gaussian_cdf_recovery",
          max_abs_error=largest_gaussian_error)
    check(largest_metadata_error < 5e-10, "independent_normalization_metadata",
          max_abs_error=largest_metadata_error)
    check(largest_interior_nonproportionality < 5e-10, "wholly_core_bin_proportionality",
          max_abs_error=largest_interior_nonproportionality)
    pd.DataFrame(shape_rows).to_csv(QA / "shape_quadrature_validation.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(QA / "bin_quadrature_validation.csv", index=False)
    pd.DataFrame(geometry_rows).to_csv(QA / "bin_straddling_validation.csv", index=False)


def scalar_invariance(check):
    from common import moving_context
    from limit_solver import OneSignalProfile
    part = moving_context("2015", 66.)
    signal = part["S"][:, 0]
    scalar = 0.973
    nominal = OneSignalProfile(part["b"], part["L"], signal).limit(part["n"])
    rescaled = OneSignalProfile(part["b"], part["L"], scalar*signal).limit(part["n"])
    relative_limit_error = abs(rescaled["A90"]*scalar/nominal["A90"]-1.)
    signed_r_error = abs(rescaled["signed_r"]-nominal["signed_r"])
    p_error = abs(rescaled["p0_fixed_mass"]-nominal["p0_fixed_mass"])
    report = dict(year="2015", mass_MeV=66., scalar=scalar,
                  nominal={k: nominal[k] for k in ("A90", "signed_r", "p0_fixed_mass", "max_score")},
                  rescaled={k: rescaled[k] for k in ("A90", "signed_r", "p0_fixed_mass", "max_score")},
                  relative_inverse_scale_limit_error=relative_limit_error,
                  signed_r_absolute_error=signed_r_error, p0_absolute_error=p_error,
                  interpretation="Exact scalar-template invariance only; real boundary bins are audited separately.")
    check(relative_limit_error < 3e-6 and signed_r_error < 2e-5 and p_error < 2e-6,
          "profile_pure_scalar_invariance", **report)
    write_json(QA / "scalar_invariance.json", report)


def row_key(row):
    return (str(row["scope"]), int(round(float(row["mass_MeV"]))),
            str(row["window"]), str(row["family"]), round(float(row["kappa"]), 8))


def scan_checks(check):
    scans_path = B / "derived/scans.csv"
    excluded_path = B / "derived/excluded.csv"
    if not scans_path.exists():
        check(False, "scan_csv_exists", path=str(scans_path))
        return
    scans = pd.read_csv(scans_path, dtype={"scope": str})
    required = {"scope", "mass_MeV", "window", "family", "kappa", "epsilon2_90",
                "display_epsilon2_90", "p0_fixed_mass", "Z0", "signed_r", "max_score", "cls"}
    check(required.issubset(scans.columns), "scan_schema", missing=sorted(required-set(scans.columns)))
    if not required.issubset(scans.columns):
        return
    expected, unsupported = set(), set()
    exclusion_geometry = []
    for year, masses in DOMAINS.items():
        for mass in masses:
            for window in WINDOWS:
                geo = geometry(year, mass, window)
                good = min(geo["left"], geo["right"]) >= 3 and geo["mask"].any()
                target = expected if good else unsupported
                for family, kappa in SHAPES:
                    target.add((year, mass, window, family, kappa))
                if not good:
                    exclusion_geometry.append(dict(scope=year, mass_MeV=mass, window=window,
                                                    left=geo["left"], right=geo["right"]))
    keys = [row_key(row) for row in scans.to_dict("records")]
    actual = set(keys)
    check(len(keys) == len(actual), "no_duplicate_scan_rows", rows=len(keys), unique_rows=len(actual))
    check(actual == expected, "complete_supported_row_coverage", expected=len(expected), actual=len(actual),
          missing=[list(x) for x in sorted(expected-actual)[:30]],
          unexpected=[list(x) for x in sorted(actual-expected)[:30]])
    excluded_keys = set()
    if excluded_path.exists() and excluded_path.stat().st_size:
        try:
            excluded = pd.read_csv(excluded_path, dtype={"scope": str, "year": str})
        except pd.errors.EmptyDataError:
            excluded = pd.DataFrame()
        for row in excluded.to_dict("records"):
            year = str(row.get("scope", row.get("year")))
            shapes = [(str(row["family"]), float(row["kappa"]))] if "family" in row else SHAPES
            for family, kappa in shapes:
                excluded_keys.add((year, int(row["mass_MeV"]), str(row["window"]), family, round(kappa, 8)))
    check(excluded_keys == unsupported, "explicit_unsupported_edge_ledger",
          expected=len(unsupported), actual=len(excluded_keys),
          missing=[list(x) for x in sorted(unsupported-excluded_keys)[:30]],
          unexpected=[list(x) for x in sorted(excluded_keys-unsupported)[:30]])
    write_json(QA / "row_coverage_validation.json", dict(
        mass_coordinates=sum(len(m) for m in DOMAINS.values()),
        requested_rows=len(expected)+len(unsupported), supported_rows=len(expected),
        recorded_rows=len(actual), excluded_shape_rows=len(unsupported),
        excluded_coordinates=exclusion_geometry))
    numeric = list(required-{"scope", "window", "family"})
    check(np.all(np.isfinite(scans[numeric].to_numpy(float))), "finite_scan_outputs")
    check(bool((scans["epsilon2_90"] > 0).all() and (scans["display_epsilon2_90"] > 0).all()),
          "positive_observed_limits")
    p = scans["p0_fixed_mass"].to_numpy(float)
    z = scans["Z0"].to_numpy(float)
    r = scans["signed_r"].to_numpy(float)
    check(bool(np.all((p >= 0.) & (p <= .5)) and np.all(z >= 0.)), "bounded_local_p0_and_Z")
    zerr = float(np.max(abs(z-np.maximum(r, 0.))))
    perr = float(np.max(abs(p-ndtr(-z))))
    check(zerr < 2e-10 and perr < 2e-12, "p0_signed_root_consistency",
          Z_absolute_error=zerr, p0_absolute_error=perr)
    cls_error = float(np.max(abs(scans["cls"]-.1)))
    score = float(scans["max_score"].max())
    check(cls_error <= 2e-6, "profile_CLs_endpoint", max_abs_error=cls_error)
    check(score <= 3.01e-5, "profile_KKT_tolerance", maximum_score=score)
    if "min_lambda" in scans:
        check(bool((scans["min_lambda"] >= 0).all()), "nonnegative_profile_Poisson_means",
              minimum_lambda=float(scans["min_lambda"].min()))
    if "monotonicity_error" in scans:
        check(float(scans["monotonicity_error"].max()) <= 5e-5, "CLs_monotonicity",
              maximum_error=float(scans["monotonicity_error"].max()))
    if "ok" in scans:
        check(bool(scans["ok"].astype(str).str.lower().eq("true").all()), "all_rows_converged")

    nominal = pd.read_csv(B / "inputs/nominal_v505_curves.csv")
    scope_map = {"individual_2015_full": "2015", "individual_2016_full": "2016",
                 "individual_2021_10pct": "2021"}
    nominal = nominal[nominal["scope_key"].isin(scope_map)].copy()
    nominal["scope"] = nominal["scope_key"].map(scope_map)
    baseline = scans[(scans["family"] == "gaussian") & (scans["window"] == "primary")]
    replay = baseline.merge(nominal, on=["scope", "mass_MeV"], how="inner", validate="one_to_one")
    replay["relative_electron_limit_error"] = replay["epsilon2_90"]/replay["eps2_90"]-1.
    replay["relative_display_limit_error"] = replay["display_epsilon2_90"]/replay["eps2_observed"]-1.
    replay["Z_absolute_error"] = replay["Z0"]-replay["Z_local_asymptotic"]
    replay["p0_absolute_error"] = replay["p0_fixed_mass"]-replay["p0_local_asymptotic"]
    replay[["scope", "mass_MeV", "epsilon2_90", "display_epsilon2_90", "eps2_90", "eps2_observed",
            "relative_electron_limit_error", "relative_display_limit_error", "Z0", "Z_local_asymptotic", "Z_absolute_error",
            "p0_fixed_mass", "p0_local_asymptotic", "p0_absolute_error"]].to_csv(
                QA / "nominal_replay_validation.csv", index=False)
    limit_error = float(replay["relative_electron_limit_error"].abs().max())
    display_error = float(replay["relative_display_limit_error"].abs().max())
    zerror = float(replay["Z_absolute_error"].abs().max())
    perror = float(replay["p0_absolute_error"].abs().max())
    check(len(replay) == sum(len(m) for m in DOMAINS.values()), "nominal_replay_mass_coverage",
          rows=len(replay))
    # The published table is an external historical comparison, not the exact
    # input-code replay contract.  Preserve every discrepancy and expose the
    # strict agreement result; do not enlarge the replay tolerance to pass it.
    historical = dict(
        classification="Historical published-ledger comparison; not exact code replay",
        strict_agreement_at_2e_minus4=bool(limit_error < 2e-4 and display_error < 2e-4
                                           and zerror < 2e-4 and perror < 2e-4),
        maximum_relative_electron_limit_error=limit_error,
        maximum_relative_display_limit_error=display_error,
        maximum_Z_absolute_error=zerror, maximum_p0_absolute_error=perror,
        detail_csv="nominal_replay_validation.csv",
        explanation="The pinned portable Poisson/GP machinery and published ledger have a pre-existing difference. Both results remain visible; the scan is never altered to match the ledger.")
    write_json(QA / "published_ledger_comparison.json", historical)
    check(bool(np.isfinite([limit_error, display_error, zerror, perror]).all()),
          "historical_published_ledger_differences_recorded", **historical)

    # Separately replay the original portable path: its own Gaussian CDF
    # template, local context, and scalar profile, without tail_model.weights.
    from common import moving_context
    from limit_solver import OneSignalProfile
    direct_rows = []
    for year, mass in (("2015", 66), ("2015", 100), ("2016", 83),
                       ("2021", 197), ("2021", 250)):
        part = moving_context(year, float(mass))
        original = OneSignalProfile(part["b"], part["L"], part["S"][:, 0]).limit(part["n"])
        saved = baseline[(baseline["scope"] == year) & (baseline["mass_MeV"] == mass)]
        if len(saved) != 1:
            check(False, "direct_replay_coordinate_" + year + "_" + str(mass),
                  observed_rows=len(saved))
            continue
        saved = saved.iloc[0]
        direct_rows.append(dict(scope=year, mass_MeV=mass,
            original_epsilon2_90=original["A90"]*1e-8,
            scan_epsilon2_90=float(saved["epsilon2_90"]),
            relative_limit_error=float(saved["epsilon2_90"])/(original["A90"]*1e-8)-1.,
            original_Z0=original["Z0"], scan_Z0=float(saved["Z0"]),
            Z_absolute_error=float(saved["Z0"])-original["Z0"],
            p0_absolute_error=float(saved["p0_fixed_mass"])-original["p0_fixed_mass"]))
    direct = pd.DataFrame(direct_rows)
    direct.to_csv(QA / "portable_machinery_replay.csv", index=False)
    check(len(direct_rows) == 5 and bool((direct["relative_limit_error"].abs() < 2e-7).all()
          and (direct["Z_absolute_error"].abs() < 2e-7).all()
          and (direct["p0_absolute_error"].abs() < 2e-7).all()),
          "exact_portable_machinery_replay",
          maximum_relative_limit_error=float(direct["relative_limit_error"].abs().max()),
          maximum_Z_absolute_error=float(direct["Z_absolute_error"].abs().max()),
          maximum_p0_absolute_error=float(direct["p0_absolute_error"].abs().max()), rows=len(direct_rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes-only", action="store_true")
    args = parser.parse_args()
    QA.mkdir(parents=True, exist_ok=True)
    checks = []
    def check(passed, name, **details):
        checks.append(dict(name=name, passed=bool(passed), **details))
    shape_checks(check)
    scalar_invariance(check)
    if not args.shapes_only:
        scan_checks(check)
    passed = all(row["passed"] for row in checks)
    report = dict(passed=passed, scope="shapes_only" if args.shapes_only else "full_study",
                  checks=checks, passed_checks=sum(row["passed"] for row in checks),
                  total_checks=len(checks),
                  limitations="Deterministic numerical QA; no coverage or global-p calibration.")
    destination = QA / ("shape_validation.json" if args.shapes_only else "validation.json")
    write_json(destination, report)
    print(json.dumps(dict(passed=passed, passed_checks=report["passed_checks"],
                          total_checks=len(checks), report=str(destination),
                          failed=[row["name"] for row in checks if not row["passed"]]), indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
