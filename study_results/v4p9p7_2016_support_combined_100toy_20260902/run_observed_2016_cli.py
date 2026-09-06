#!/usr/bin/env python3
"""Run the archived CLI for full 2016 only after the support freeze gate."""

from __future__ import annotations

import argparse
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from observed_2016_contract import (
    CARD,
    CARD_MANIFEST,
    FREEZE,
    HERE,
    PRIMARY,
    REPAIR_PLAN,
    REPEAT_ROOT,
    STUDY_ID,
    STUDY_SPEC,
    ObservedContractError,
    activate_runtime,
    atomic_csv,
    atomic_json,
    load_json,
    sha256,
    static_preflight,
    validate_card,
    validate_freeze,
    validate_mass_grid,
)


RUN_MANIFEST = "_OBSERVED_RUN_PROVENANCE.json"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--support-freeze", type=Path, default=FREEZE)
    primary = subparsers.add_parser("primary")
    primary.add_argument("--support-freeze", type=Path, default=FREEZE)
    primary.add_argument("--card", type=Path, default=CARD)
    primary.add_argument("--card-manifest", type=Path, default=CARD_MANIFEST)
    repeat = subparsers.add_parser("repeat")
    repeat.add_argument("--support-freeze", type=Path, default=FREEZE)
    repeat.add_argument("--card", type=Path, default=CARD)
    repeat.add_argument("--card-manifest", type=Path, default=CARD_MANIFEST)
    repeat.add_argument("--repair-plan", type=Path, default=REPAIR_PLAN)
    repeat.add_argument("--mass-mev", type=int, required=True)
    repeat.add_argument("--repeat-index", type=int, choices=(1, 2, 3), required=True)
    return parser.parse_args(argv)


def covariance_diagnostics(pred: Any) -> Dict[str, Any]:
    try:
        covariance = np.asarray(pred.cov, dtype=float)
        if (
            covariance.ndim != 2
            or covariance.shape[0] != covariance.shape[1]
            or covariance.shape[0] == 0
            or not np.isfinite(covariance).all()
        ):
            raise ValueError("non-finite or nonsquare covariance")
        symmetric = 0.5 * (covariance + covariance.T)
        eigenvalues = np.linalg.eigvalsh(symmetric)
        scale = max(float(np.max(np.abs(np.diag(symmetric)))), 1.0e-300)
        minimum = float(np.min(eigenvalues))
        relative = minimum / scale
        valid = bool(np.isfinite(relative) and relative >= -0.01)
    except Exception:
        minimum = float("nan")
        relative = float("nan")
        valid = False
    return {
        "optimizer_random_state": int(
            getattr(pred, "optimizer_random_state", -1)
        ),
        "optimizer_warning_count": int(
            getattr(pred, "optimizer_warning_count", 0)
        ),
        "optimizer_warnings": str(getattr(pred, "optimizer_warnings", "")),
        "covariance_valid": valid,
        "covariance_min_eigenvalue": minimum,
        "covariance_min_eigenvalue_relative": relative,
    }


def install_instrumented_scan() -> Dict[str, str]:
    runtime = activate_runtime()
    import hps_gpr.scan as scan_module

    original_evaluate = scan_module.evaluate_single_dataset
    original_run_scan = scan_module.run_scan
    captured: Dict[Tuple[str, float], Dict[str, Any]] = {}
    lock = threading.Lock()

    def evaluate_with_diagnostics(ds: Any, mass: float, config: Any, **kwargs: Any):
        result = original_evaluate(ds, mass, config, **kwargs)
        pred = result[1]
        key = (str(ds.key), round(float(mass), 12))
        with lock:
            if key in captured:
                raise ObservedContractError(f"duplicate observed fit capture: {key}")
            captured[key] = covariance_diagnostics(pred)
        return result

    def run_scan_with_diagnostics(
        datasets: Any,
        config: Any,
        mass_min: Optional[float] = None,
        mass_max: Optional[float] = None,
    ):
        captured.clear()
        scan_module.evaluate_single_dataset = evaluate_with_diagnostics
        try:
            single, combined = original_run_scan(
                datasets, config, mass_min=mass_min, mass_max=mass_max
            )
        finally:
            scan_module.evaluate_single_dataset = original_evaluate
        if single.empty:
            raise ObservedContractError("archived CLI produced no observed rows")
        augmented = single.copy()
        diagnostics = []
        for _, row in augmented.iterrows():
            key = (str(row["dataset"]), round(float(row["mass_GeV"]), 12))
            diagnostics.append(
                captured.get(
                    key,
                    {
                        "optimizer_random_state": -1,
                        "optimizer_warning_count": 0,
                        "optimizer_warnings": "fit raised before diagnostics capture",
                        "covariance_valid": False,
                        "covariance_min_eigenvalue": float("nan"),
                        "covariance_min_eigenvalue_relative": float("nan"),
                    },
                )
            )
        for column in diagnostics[0]:
            augmented[column] = [record[column] for record in diagnostics]
        atomic_csv(Path(config.output_dir) / "results_single.csv", augmented)
        return augmented, combined

    scan_module.run_scan = run_scan_with_diagnostics
    return dict(runtime["origins"])


def execute_archived_cli(
    card: Path,
    output: Path,
    mass_min: float,
    mass_max: float,
) -> Dict[str, str]:
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    origins = install_instrumented_scan()
    from hps_gpr.cli import main as archived_main

    old_cwd = Path.cwd()
    try:
        os.chdir(HERE)
        archived_main.main(
            args=[
                "scan",
                "--config",
                str(card.resolve()),
                "--output-dir",
                str(output.resolve()),
                "--mass-min",
                f"{mass_min:.3f}",
                "--mass-max",
                f"{mass_max:.3f}",
            ],
            prog_name="run_observed_2016_cli.py",
            standalone_mode=False,
        )
    finally:
        os.chdir(old_cwd)
    return origins


def validate_run_output(output: Path, mass_mev: Optional[int]) -> pd.DataFrame:
    result_path = output / "results_single.csv"
    if not result_path.is_file():
        raise ObservedContractError(f"missing archived-CLI result: {result_path}")
    frame = pd.read_csv(result_path)
    if mass_mev is None:
        validate_mass_grid(frame)
    else:
        validate_mass_grid(frame, expected_rows=1)
        if not np.isclose(
            float(frame.iloc[0]["mass_GeV"]), mass_mev / 1000.0, rtol=0.0, atol=1e-12
        ):
            raise ObservedContractError("unchanged-card repeat returned another mass")
    required = {
        "optimizer_random_state",
        "optimizer_warning_count",
        "optimizer_warnings",
        "covariance_valid",
        "covariance_min_eigenvalue",
        "covariance_min_eigenvalue_relative",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ObservedContractError(f"instrumented result lacks columns: {missing}")
    return frame


def validate_numbers_json(
    output: Path, frame: pd.DataFrame
) -> Tuple[Dict[str, str], Sequence[str]]:
    hashes: Dict[str, str] = {}
    missing = []
    for _, row in frame.iterrows():
        mass_mev = int(round(1000.0 * float(row["mass_GeV"])))
        path = output / f"m{mass_mev:03d}MeV" / "2016" / "numbers.json"
        relative = path.relative_to(output).as_posix()
        if not path.is_file():
            missing.append(relative)
            continue
        payload = load_json(path)
        if str(payload.get("dataset")) != "2016" or not np.isclose(
            float(payload.get("mass_GeV", float("nan"))),
            mass_mev / 1000.0,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ObservedContractError(f"numbers JSON coordinate drift: {path}")
        comparisons = {
            "A_up": payload.get("A_up"),
            "eps2_up": payload.get("eps2_up"),
            "p0_analytic": payload.get("p0_analytic"),
            "A_hat": payload.get("A_hat"),
            "sigma_A": payload.get("sigma_A"),
            "lml": payload.get("gp_diagnostics", {})
            .get("optimizer", {})
            .get("log_marginal_likelihood"),
            "ls_opt": payload.get("gp_diagnostics", {})
            .get("length_scale", {})
            .get("optimized"),
            "const_opt": payload.get("gp_diagnostics", {})
            .get("constant", {})
            .get("optimized"),
        }
        for column, value in comparisons.items():
            row_value = float(row[column])
            if np.isfinite(row_value):
                matches = value is not None and np.isclose(
                    float(value), row_value, rtol=1e-12, atol=1e-12
                )
            else:
                matches = value is None or not np.isfinite(float(value))
            if not matches:
                raise ObservedContractError(
                    f"numbers JSON {column} mismatch at {mass_mev} MeV"
                )
        hashes[relative] = sha256(path)
    return hashes, tuple(missing)


def validate_plan(path: Path, freeze_sha: str, card_sha: str) -> Dict[str, Any]:
    path = path.expanduser().resolve()
    if path != REPAIR_PLAN.resolve():
        raise ObservedContractError(f"repair plan must be {REPAIR_PLAN}")
    plan = load_json(path)
    expected = {
        "status": "observed_repair_plan_frozen",
        "study_id": STUDY_ID,
        "support_freeze_sha256": freeze_sha,
        "card_sha256": card_sha,
        "primary_results_sha256": sha256(PRIMARY / "results_single.csv"),
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise ObservedContractError(f"repair-plan binding drift: {key}")
    return plan


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    static = static_preflight()
    if args.mode == "preflight":
        status = "production_blocked_no_provisional_edge"
        print(
            json.dumps(
                {
                    "status": status,
                    "observed_data_evaluated": False,
                    "static": static,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    decision = validate_freeze(args.support_freeze)
    card_manifest = validate_card(args.card, args.card_manifest, decision)
    freeze_sha = sha256(FREEZE)
    card_sha = sha256(CARD)
    if args.mode == "primary":
        output = PRIMARY
        mass_mev = None
        mass_min, mass_max = 0.039, 0.180
        run_role = "primary_142_mass_scan"
        repair_plan_sha = None
    else:
        plan = validate_plan(args.repair_plan, freeze_sha, card_sha)
        mass_mev = int(args.mass_mev)
        flagged = {int(item) for item in plan.get("repeat_masses_MeV", [])}
        if mass_mev not in flagged:
            raise ObservedContractError(
                f"{mass_mev} MeV is not in the frozen repair plan"
            )
        output = (
            REPEAT_ROOT
            / f"m{mass_mev:03d}"
            / f"repeat{int(args.repeat_index)}"
        )
        mass_min = mass_max = mass_mev / 1000.0
        run_role = f"unchanged_card_repeat_{int(args.repeat_index)}"
        repair_plan_sha = sha256(REPAIR_PLAN)
    if output.exists():
        raise ObservedContractError(f"refusing to overwrite observed run: {output}")

    origins = execute_archived_cli(CARD, output, mass_min, mass_max)
    frame = validate_run_output(output, mass_mev)
    numbers_hashes, missing_numbers = validate_numbers_json(output, frame)
    manifest = {
        "schema_version": 1,
        "status": "pass",
        "study_id": STUDY_ID,
        "run_role": run_role,
        "support_freeze_sha256": freeze_sha,
        "study_spec_sha256": sha256(STUDY_SPEC),
        "card_sha256": card_sha,
        "card_manifest_sha256": sha256(CARD_MANIFEST),
        "selected_support_low_MeV": int(
            decision["selected_support_low_MeV"]
        ),
        "support_high_MeV": int(decision["support_high_MeV"]),
        "mass_min_GeV": mass_min,
        "mass_max_GeV": mass_max,
        "rows": int(len(frame)),
        "results_single_sha256": sha256(output / "results_single.csv"),
        "numbers_json_count": len(numbers_hashes),
        "numbers_json_sha256": numbers_hashes,
        "numbers_json_missing": list(missing_numbers),
        "runtime_origins": origins,
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "observed_scan_authorized": True,
        "unchanged_card": True,
        "card_builder_sha256": card_manifest["card_builder_sha256"],
    }
    if repair_plan_sha is not None:
        manifest["repair_plan_sha256"] = repair_plan_sha
    atomic_json(output / RUN_MANIFEST, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
