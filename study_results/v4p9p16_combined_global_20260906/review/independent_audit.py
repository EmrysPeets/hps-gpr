#!/usr/bin/env python3
"""Read saved combined products only; print an independent numerical audit.

No runner imports, likelihood evaluations, random draws, or file writes.
Use --require-complete for the final 232-coordinate product.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
FOLDER = BASE / "global"
METHODS = ("profiled", "fixed")
ENSEMBLES = {"pilot10": 10, "validation1000": 1000, "asimov": 1627}
BLOCKS = {"2015": (0, 484), "2016": (484, 1204), "2021": (1204, 1626)}
SENTINELS = {39, 49, 50, 90, 91, 180}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def membership(mass):
    return [year for year, low, high in
            (("2015", 19, 90), ("2016", 39, 180), ("2021", 50, 250))
            if low <= mass <= high]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    failures = []
    checked = 0

    def check(condition, label):
        nonlocal checked
        checked += 1
        if not bool(condition):
            failures.append(label)

    contract = read_json(FOLDER / "contract.json")
    contract_sha = sha(FOLDER / "contract.json")
    check(contract["mass_grid_MeV"] == list(range(19, 251)), "declared grid")
    check(contract["sentinels_MeV"] == sorted(SENTINELS), "declared sentinels")
    check(contract["membership"] == [dict(mass_MeV=m, datasets=membership(m))
          for m in range(19, 251)], "declared membership")
    for path, digest in contract["source_sha256"].items():
        check(sha(ROOT / path) == digest, "source hash: " + path)

    saved = {e: arrays(FOLDER / "spectra" / (e + ".npz")) for e in ENSEMBLES}
    truth = saved["asimov"]["truth"]
    check(truth.shape == (1626,) and np.all(truth > 0), "positive full truth")
    check(hashlib.sha256(truth.tobytes()).hexdigest() == contract["truth_sha256"],
          "truth bytes")
    for ensemble, nrows in ENSEMBLES.items():
        item = saved[ensemble]
        check(item["counts"].shape == (nrows, 1626), ensemble + " shape")
        check(np.array_equal(item["truth"], truth), ensemble + " truth")
        if ensemble != "asimov":
            parts = []
            for year in BLOCKS:
                path = ROOT / contract["upstream"][year][ensemble]["source"]
                source = arrays(path / "spectra.npz")
                first, last = BLOCKS[year]
                check(np.array_equal(source["truth"], truth[first:last]),
                      ensemble + " source truth " + year)
                parts.append(source["counts"])
            check(np.array_equal(item["counts"], np.concatenate(parts, axis=1)),
                  ensemble + " same-ID source concatenation")
    expected = np.broadcast_to(truth, (1627, 1626)).copy()
    indices = np.arange(1626)
    expected[indices + 1, indices] += np.sqrt(truth)
    check(np.array_equal(saved["asimov"]["counts"], expected), "Asimov bin basis")
    del expected, saved

    completed = []
    response_columns = {method: {} for method in METHODS}
    peak_errors = {method: dict(pilot=0., response=0.) for method in METHODS}
    legacy_flags = []
    for mass in range(19, 251):
        checkpoint = FOLDER / "points" / f"m{mass:03d}.npz"
        auditpath = checkpoint.with_name(f"m{mass:03d}_qa.json")
        if not (checkpoint.exists() and auditpath.exists()):
            continue
        completed.append(mass)
        audit = read_json(auditpath)
        values = arrays(checkpoint)
        keys = membership(mass)
        active = np.concatenate([np.arange(*BLOCKS[year]) for year in keys])
        inactive = np.setdiff1d(np.arange(1626), active)
        check(audit["passed"] and audit["checkpoint_sha256"] == sha(checkpoint)
              and audit["contract_sha256"] == contract_sha, f"m{mass} checkpoint QA")
        check(audit["active_datasets"] == keys, f"m{mass} membership")
        check(audit["observed"]["dataset_set"] == "+".join(keys),
              f"m{mass} observed membership")
        for path, digest in audit["source_reference_sha256"].items():
            check(sha(ROOT / path) == digest, f"m{mass} reference hash: " + path)
        for ensemble, nrows in ENSEMBLES.items():
            for method in METHODS:
                vector = values[ensemble + "_" + method]
                check(vector.shape == (nrows,) and np.isfinite(vector).all(),
                      f"m{mass} {ensemble} {method} shape/finite")
        for method in METHODS:
            r = values["asimov_" + method]
            check(np.all(r[inactive + 1] == r[0]), f"m{mass} {method} inactive rows")
            check(np.linalg.norm(r[1:] - r[0]) > 0, f"m{mass} {method} width")

        if len(keys) == 1:
            for ensemble in ENSEMBLES:
                source_folder = ROOT / contract["upstream"][keys[0]][ensemble]["source"]
                sourcepath = source_folder / checkpoint.name
                upstream = arrays(sourcepath)
                upstream_qa = read_json(sourcepath.with_name(sourcepath.stem + "_qa.json"))
                check(upstream_qa["passed"] and upstream_qa["checkpoint_sha256"] == sha(sourcepath),
                      f"m{mass} {ensemble} upstream QA")
                for method in METHODS:
                    original = upstream[method]
                    if ensemble == "asimov":
                        embedded = np.full(1627, original[0])
                        embedded[active + 1] = original[1:]
                        original = embedded
                    check(np.array_equal(values[ensemble + "_" + method], original),
                          f"m{mass} {ensemble} {method} exact reuse")
            continue

        check(audit.get("memoization_parent_baseline_exact") is True,
              f"m{mass} memoization reference")
        if audit["numerical_backend"] != "exact_cached_cholesky":
            check(audit["parent_gate_passed"] and all(x["passed"] for x in audit["parent_checks"])
                  and not audit["fallback_reasons"], f"m{mass} accepted approximation gates")
        if audit.get("observed_checks", {}).get("v12_investigation_required"):
            legacy_flags.append(mass)
        for method in METHODS:
            observed = audit["observed_checks"][method]
            check(abs(observed["cls"] - .1) <= 2e-6 and observed["max_score"] < 2e-7
                  and observed["min_lambda"] > 0 and observed["monotonicity_error"] <= 5e-5,
                  f"m{mass} {method} observed fit gates")
        referencepath = FOLDER / "references" / checkpoint.name
        reference = arrays(referencepath)
        reference_qa = read_json(referencepath.with_suffix(".json"))
        check(reference_qa["passed"] and reference_qa["reference_sha256"] == sha(referencepath),
              f"m{mass} exact-reference QA")
        check(np.array_equal(reference["active_full_bin_indices"], active),
              f"m{mass} exact-reference active embedding")
        probes = reference["probe_indices"]
        check(probes[0] == 0 and len(np.unique(probes)) == len(probes),
              f"m{mass} exact-reference baseline/stencil")
        for method in METHODS:
            pilot = values["pilot10_" + method]
            exact_pilot = reference["pilot_" + method]
            error = float(np.max(abs(pilot - exact_pilot)))
            peak_errors[method]["pilot"] = max(peak_errors[method]["pilot"], error)
            check(error < 1e-3 and np.array_equal(pilot > 0, exact_pilot > 0),
                  f"m{mass} {method} paired pilot/atom")
            embedded = values["asimov_" + method]
            local = np.r_[embedded[0], embedded[active + 1]]
            exact = reference["response_" + method]
            width = np.linalg.norm(local[1:] - local[0])
            delta = (local[probes] - local[0]) - (exact - exact[0])
            error = float(np.max(abs(delta)))
            peak_errors[method]["response"] = max(peak_errors[method]["response"], error)
            check(np.max(abs(local[probes] - exact)) < 1e-3
                  and abs(local[0] - exact[0]) / width < 1e-3
                  and error < 1e-4 and error / width < 1e-4,
                  f"m{mass} {method} centered response gates")
            if mass in SENTINELS:
                check(np.array_equal(probes, np.arange(len(active) + 1)),
                      f"m{mass} {method} full sentinel basis")
                exact_d = exact[1:] - exact[0]
                check(np.linalg.norm(delta[1:]) / np.linalg.norm(exact_d) < 1e-3
                      and abs(width / np.linalg.norm(exact_d) - 1) < 1e-3,
                      f"m{mass} {method} full response L2/width")
                exact_embedded = np.zeros(1626)
                exact_embedded[active] = exact_d
                response_columns[method][mass] = (exact_embedded, embedded[1:] - embedded[0])

    correlations = {}
    for method, columns in response_columns.items():
        masses = sorted(columns)
        if not masses:
            continue
        exact = np.column_stack([columns[m][0] for m in masses])
        final = np.column_stack([columns[m][1] for m in masses])
        exact /= np.linalg.norm(exact, axis=0)
        final /= np.linalg.norm(final, axis=0)
        difference = float(np.max(abs(exact.T @ exact - final.T @ final)))
        correlations[method] = dict(masses_MeV=masses, max_absolute_difference=difference)
        check(difference < 1e-3, method + " available sentinel correlation gate")
    complete = completed == list(range(19, 251))
    if args.require_complete:
        check(complete, "complete 232-point grid required")
        for method in METHODS:
            check(set(response_columns[method]) == SENTINELS, method + " all sentinels required")
        assembled_path = FOLDER / "scan_vectors.npz"
        summary_path = FOLDER / "summary.json"
        assembled_present = assembled_path.exists() and summary_path.exists()
        check(assembled_present, "assembled products required")
        if assembled_present:
            assembled = arrays(assembled_path)
            final_summary = read_json(summary_path)
            check(final_summary["passed"] and final_summary["complete"]
                  and final_summary["contract_sha256"] == contract_sha
                  and final_summary["vectors_sha256"] == sha(assembled_path)
                  and final_summary["observed_sha256"] == sha(FOLDER / "observed.csv"),
                  "assembled product hashes")
            check(np.array_equal(assembled["masses_MeV"], np.arange(19, 251)),
                  "assembled grid")
            for mass in completed:
                point = arrays(FOLDER / "points" / f"m{mass:03d}.npz")
                for ensemble in ENSEMBLES:
                    for method in METHODS:
                        key = ensemble + "_" + method
                        check(np.array_equal(point[key], assembled[key][:, mass - 19]),
                              f"m{mass} {key} assembled column")
    summary = dict(complete=complete, completed_coordinates=len(completed),
                   checked_conditions=checked, checked_conditions_passed=not failures,
                   failures=failures, max_paired_errors=peak_errors,
                   sentinel_correlation=correlations,
                   v12_investigation_flags=legacy_flags,
                   scope="read-only numerical audit; no physical-background or tail calibration claim")
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
