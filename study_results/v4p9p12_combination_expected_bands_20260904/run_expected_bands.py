#!/usr/bin/env python3
"""Run deterministic cumulative pointwise expected-limit bands for v4.9.12."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_key, "1")

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PARENT = REPO / "study_results" / "v4p9p12_final_dataset_combinations_20260902"
sys.path.insert(0, str(PARENT))

import run_final_combinations as final  # noqa: E402
from hps_gpr.config import load_config  # noqa: E402
from hps_gpr.conversion import A_from_epsilon2  # noqa: E402
from hps_gpr.dataset import make_datasets  # noqa: E402
from hps_gpr.evaluation import build_combined_components  # noqa: E402
from hps_gpr.statistics import _chol_with_jitter  # noqa: E402
from band_solver import (  # noqa: E402
    SOLVER_VERSION,
    CachedPiecewiseBoundedLimit,
)


DEFAULT_OUTPUT = HERE / "derived"
DEFAULT_CARD = PARENT / "inputs" / "analysis_card.yaml"
DEFAULT_STATES = PARENT / "inputs" / "reviewed_gp_states.csv"
DEFAULT_INPUT_PROVENANCE = PARENT / "inputs" / "analysis_input_provenance.json"
DEFAULT_OBSERVED = PARENT / "derived" / "final_dataset_result_curves.csv"
MASTER_SEED = 491204
ALLOWED_RELEASE_STAGES = (50, 100, 300)
SCHEMA_VERSION = 2
GENERATOR_VERSION = "v4p9p12_mass_toy_dataset_seed_cholesky_mvn_poisson_v1"
QUANTILE_PROBABILITIES = (0.025, 0.16, 0.50, 0.84, 0.975)
QUANTILE_COLUMNS = (
    "expected_q025",
    "expected_q16",
    "expected_median",
    "expected_q84",
    "expected_q975",
)
DATASET_INDEX = {key: index for index, key in enumerate(final.DATASET_ORDER)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-toys", type=int, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--mass-min-mev", type=int, default=19)
    parser.add_argument("--mass-max-mev", type=int, default=250)
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Report missing toys from valid checkpoints without executing them.",
    )
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array, dtype="<i8"))
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def scope_spec(scope_key: str) -> Tuple[str, str, Tuple[str, ...], int, int]:
    return next(item for item in final.SCOPES if item[0] == scope_key)


def active_scopes(mass_mev: int) -> List[Tuple[str, str, Tuple[str, ...], int, int]]:
    return [
        item for item in final.SCOPES if int(item[3]) <= mass_mev <= int(item[4])
    ]


def contract(
    *, card: Path, states: Path, provenance: Path, observed: Path, seed: int
) -> Tuple[str, Dict[str, object]]:
    paths = {
        "runner": Path(__file__).resolve(),
        "statistical_protocol": HERE / "STATISTICAL_PROTOCOL.md",
        "numerical_amendment": HERE / "NUMERICAL_AMENDMENT_PRE_PRODUCTION.md",
        "continuation_numerical_amendment": (
            HERE / "NUMERICAL_AMENDMENT_100TOY_CONTINUATION.md"
        ),
        "band_solver": HERE / "band_solver.py",
        "analysis_card": card,
        "reviewed_gp_states": states,
        "analysis_input_provenance": provenance,
        "observed_result_curves": observed,
        "parent_runner": PARENT / "run_final_combinations.py",
        "parent_solver": PARENT / "piecewise_cached_solver.py",
        "parent_tail_mapper": PARENT / "runtime" / "bounded_tildeq_cls.py",
    }
    hashes = {name: sha256(path) for name, path in paths.items()}
    payload: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "master_seed": int(seed),
        "quantile_probabilities": list(QUANTILE_PROBABILITIES),
        "quantile_method": "linear",
        "parent_paths_sha256": hashes,
        "runtime_import_origins": final.RUNTIME_IMPORT_ORIGINS,
        "gp_package_origin": final.GP_PACKAGE_ORIGIN,
        "solver_version": SOLVER_VERSION,
        "scope_specs": [list(item) for item in final.SCOPES],
        "dataset_index": DATASET_INDEX,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), payload


def load_checkpoint(path: Path, digest: str, mass_mev: int) -> Dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not (
        payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("contract_sha256") == digest
        and int(payload.get("mass_MeV", -1)) == mass_mev
    ):
        return None
    draw_rows = list(payload.get("draw_rows", []))
    limit_rows = list(payload.get("limit_rows", []))
    n_completed = int(payload.get("n_toys_completed", -1))
    scopes = active_scopes(mass_mev)
    expected_draws = n_completed * len(
        sorted({key for _, _, keys, _, _ in scopes for key in keys})
    )
    expected_limits = n_completed * len(scopes)
    if n_completed < 0 or len(draw_rows) != expected_draws or len(limit_rows) != expected_limits:
        return None
    for scope_key, _, _, _, _ in scopes:
        found = sorted(
            int(row["toy_id"])
            for row in limit_rows
            if str(row["scope_key"]) == scope_key
        )
        if found != list(range(n_completed)):
            return None
    return payload


def _draw_dataset_observation(
    mean: np.ndarray,
    effective_factor: np.ndarray,
    *,
    seed: int,
    mass_mev: int,
    toy_id: int,
    dataset_key: str,
) -> Tuple[np.ndarray, Dict[str, object]]:
    dataset_index = DATASET_INDEX[dataset_key]
    seed_descriptor = [int(seed), int(mass_mev), int(toy_id), dataset_index]
    rng = np.random.default_rng(np.random.SeedSequence(seed_descriptor))
    accepted = None
    attempts = 0
    minimum_before_fallback = float("nan")
    for attempts in range(1, 81):
        latent = mean + effective_factor @ rng.standard_normal(mean.size)
        minimum_before_fallback = float(np.min(latent))
        if np.all(latent >= 0.0):
            accepted = latent
            break
    clip_fallback = accepted is None
    if clip_fallback:
        accepted = np.clip(latent, 0.0, None)
    counts = rng.poisson(accepted).astype(np.int64)
    record = {
        "dataset": dataset_key,
        "mass_MeV": int(mass_mev),
        "mass_GeV": float(mass_mev / 1000.0),
        "toy_id": int(toy_id),
        "seed_descriptor": json.dumps(seed_descriptor, separators=(",", ":")),
        "observation_sha256": array_sha256(counts),
        "observation_sum": int(np.sum(counts)),
        "observation_min": int(np.min(counts)),
        "observation_max": int(np.max(counts)),
        "latent_draw_attempts": int(attempts),
        "latent_clip_fallback": bool(clip_fallback),
        "latent_minimum_before_fallback": minimum_before_fallback,
    }
    return counts, record


def build_scope_models(
    mass_mev: int,
    datasets: Mapping[str, object],
    config: object,
    predictions: Mapping[str, object],
    conditioned_covariances: Mapping[str, np.ndarray],
) -> Dict[str, Dict[str, object]]:
    models: Dict[str, Dict[str, object]] = {}
    mass_gev = mass_mev / 1000.0
    for scope_key, scope_label, keys, low, high in active_scopes(mass_mev):
        ds_here = [datasets[key] for key in keys]
        pred_here = [predictions[key] for key in keys]
        _observed, bkg, _raw_cov, s_unit = build_combined_components(
            mass_gev, ds_here, pred_here, config=config
        )
        covariance = final.block_diagonal(
            [conditioned_covariances[key] for key in keys]
        )
        solver = CachedPiecewiseBoundedLimit(
            bkg,
            covariance,
            s_unit,
            alpha=float(config.cls_alpha),
            combined_mode=str(config.combined_mode),
        )
        k_by_dataset = {
            dataset.key: float(
                A_from_epsilon2(
                    dataset,
                    mass_gev,
                    1.0,
                    prediction.integral_density,
                )
            )
            for dataset, prediction in zip(ds_here, pred_here)
        }
        models[scope_key] = {
            "scope_key": scope_key,
            "scope_label": scope_label,
            "keys": tuple(keys),
            "solver": solver,
            "full_signal_yield_per_eps2": float(sum(k_by_dataset.values())),
            "fitted_signal_yield_per_eps2": float(np.sum(s_unit)),
            "conditioned_combined_covariance_sha256": final.array_sha256(
                covariance
            ),
        }
    return models


def run_mass(
    mass_mev: int,
    *,
    target_toys: int,
    seed: int,
    digest: str,
    checkpoint_dir: Path,
    datasets: Mapping[str, object],
    config: object,
    states: Mapping[Tuple[str, int], Dict[str, object]],
) -> Path:
    path = checkpoint_dir / f"m{mass_mev:03d}.json"
    previous = load_checkpoint(path, digest, mass_mev)
    if previous is None:
        draw_rows: List[Dict[str, object]] = []
        limit_rows: List[Dict[str, object]] = []
        start_toy = 0
    else:
        draw_rows = list(previous["draw_rows"])
        limit_rows = list(previous["limit_rows"])
        start_toy = int(previous["n_toys_completed"])
    if start_toy >= target_toys:
        return path

    if final.assert_import_origins(final.REQUIRED_RUNTIME_MODULES) != final.RUNTIME_IMPORT_ORIGINS:
        raise RuntimeError("worker imported a different attested hps_gpr runtime")
    mass_gev = mass_mev / 1000.0
    predictions, conditioned_covariances, conditioning_records, prediction_records = (
        final.reconstruct_predictions(mass_gev, datasets, config, states)
    )
    models = build_scope_models(
        mass_mev, datasets, config, predictions, conditioned_covariances
    )
    dataset_keys = tuple(predictions)
    effective_factors = {
        key: _chol_with_jitter(conditioned_covariances[key]) for key in dataset_keys
    }

    with threadpool_limits(limits=1):
        for toy_id in range(start_toy, target_toys):
            observations: Dict[str, np.ndarray] = {}
            draw_records: Dict[str, Dict[str, object]] = {}
            for key in dataset_keys:
                observation, record = _draw_dataset_observation(
                    np.asarray(predictions[key].mu, dtype=float),
                    effective_factors[key],
                    seed=seed,
                    mass_mev=mass_mev,
                    toy_id=toy_id,
                    dataset_key=key,
                )
                observations[key] = observation
                draw_records[key] = record
                draw_rows.append(record)

            for scope_key, model in models.items():
                keys = tuple(model["keys"])
                counts = np.concatenate([observations[key] for key in keys])
                solver = model["solver"]
                counters_before = asdict(solver.counters)
                try:
                    result = solver.limit(counts)
                except Exception as exc:
                    raise RuntimeError(
                        "toy limit failed at "
                        f"mass={mass_mev} MeV, toy_id={toy_id}, "
                        f"scope={scope_key}: {exc}"
                    ) from exc
                counter_delta = {
                    key: int(result.counters[key] - counters_before[key])
                    for key in counters_before
                }
                full_signal_scale = float(model["full_signal_yield_per_eps2"])
                limit_rows.append(
                    {
                        "scope_key": scope_key,
                        "scope_label": str(model["scope_label"]),
                        "dataset_set": "+".join(keys),
                        "mass_MeV": int(mass_mev),
                        "mass_GeV": mass_gev,
                        "toy_id": int(toy_id),
                        "eps2_90": float(result.eps2_90),
                        "A90_full_template_events": float(
                            result.eps2_90 * full_signal_scale
                        ),
                        "A90_fitted_window_events": float(
                            result.eps2_90
                            * float(model["fitted_signal_yield_per_eps2"])
                        ),
                        "full_signal_yield_per_eps2": full_signal_scale,
                        "fitted_signal_yield_per_eps2": float(
                            model["fitted_signal_yield_per_eps2"]
                        ),
                        "cls_alpha": float(result.alpha),
                        "confidence_level": float(result.confidence_level),
                        "cls_at_limit": float(result.cls_at_limit),
                        "cl_sb_at_limit": float(result.cl_sb_at_limit),
                        "cl_b_at_limit": float(result.cl_b_at_limit),
                        "qmu_obs_at_limit": float(result.qmu_obs_at_limit),
                        "qmu_asimov_b_at_limit": float(
                            result.qmu_asimov_b_at_limit
                        ),
                        "tail_branch_at_limit": str(result.tail_branch_at_limit),
                        "qmu_profile_branch_at_limit": str(
                            result.observed_qmu_branch_at_limit
                        ),
                        "limit_fit_unconstrained_eps2": float(
                            result.observed_unconstrained_strength
                            / result.signal_scale_counts_per_eps2
                        ),
                        "optimizer_ok": bool(result.optimizer_ok),
                        "bisection_iterations": int(result.bisection_iterations),
                        "bracket_expansions": int(result.bracket_expansions),
                        "bracket_low_eps2": float(result.bracket_low_eps2),
                        "bracket_high_eps2": float(result.bracket_high_eps2),
                        "bracket_low_cls": float(result.bracket_low_cls),
                        "bracket_high_cls": float(result.bracket_high_cls),
                        "convergence_reason": str(result.convergence_reason),
                        "solver_counter_delta": json.dumps(
                            counter_delta, sort_keys=True, separators=(",", ":")
                        ),
                        "observation_sha256_by_dataset": json.dumps(
                            {
                                key: draw_records[key]["observation_sha256"]
                                for key in keys
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "conditioned_combined_covariance_sha256": str(
                            model["conditioned_combined_covariance_sha256"]
                        ),
                        "limit_solver": SOLVER_VERSION,
                        "combined_mode": str(result.combined_mode),
                        "conditional_on_frozen_gp": True,
                        "gp_refit_per_toy": False,
                    }
                )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_sha256": digest,
        "mass_MeV": int(mass_mev),
        "n_toys_completed": int(target_toys),
        "toy_id_min": 0,
        "toy_id_max": int(target_toys - 1),
        "prediction_rows": prediction_records,
        "conditioning_records": conditioning_records,
        "draw_rows": draw_rows,
        "limit_rows": limit_rows,
    }
    final.atomic_json(path, payload)
    return path


def checkpoint_completion(path: Path, digest: str, mass_mev: int) -> int:
    payload = load_checkpoint(path, digest, mass_mev)
    return int(payload["n_toys_completed"]) if payload is not None else 0


def collect(
    paths: Iterable[Path], digest: str, target_toys: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    limits: List[Dict[str, object]] = []
    draws: List[Dict[str, object]] = []
    predictions: List[Dict[str, object]] = []
    for path in sorted(paths):
        mass_mev = int(path.stem[1:])
        payload = load_checkpoint(path, digest, mass_mev)
        if payload is None or int(payload["n_toys_completed"]) < target_toys:
            raise RuntimeError(f"checkpoint is incomplete or stale: {path}")
        limits.extend(
            row for row in payload["limit_rows"] if int(row["toy_id"]) < target_toys
        )
        draws.extend(
            row for row in payload["draw_rows"] if int(row["toy_id"]) < target_toys
        )
        predictions.extend(payload["prediction_rows"])
    limit_frame = pd.DataFrame(limits).sort_values(
        ["scope_key", "mass_MeV", "toy_id"]
    ).reset_index(drop=True)
    draw_frame = pd.DataFrame(draws).sort_values(
        ["dataset", "mass_MeV", "toy_id"]
    ).reset_index(drop=True)
    prediction_frame = pd.DataFrame(predictions).sort_values(
        ["dataset", "mass_MeV"]
    ).reset_index(drop=True)
    return limit_frame, draw_frame, prediction_frame


def validate_raw_frames(
    limits: pd.DataFrame,
    draws: pd.DataFrame,
    predictions: pd.DataFrame,
    target_toys: int,
) -> None:
    expected_limits = target_toys * sum(final.EXPECTED_SCOPE_ROWS.values())
    expected_draws = target_toys * sum(
        len(grid) for grid in final.EXPECTED_DATASET_GRIDS.values()
    )
    if len(limits) != expected_limits:
        raise RuntimeError(
            f"expected {expected_limits} toy limits, found {len(limits)}"
        )
    if len(draws) != expected_draws:
        raise RuntimeError(f"expected {expected_draws} toy draws, found {len(draws)}")
    if len(predictions) != 415 or predictions.duplicated(["dataset", "mass_MeV"]).any():
        raise RuntimeError("prediction ledger is not the exact 415-state grid")
    if limits.duplicated(["scope_key", "mass_MeV", "toy_id"]).any():
        raise RuntimeError("duplicate toy limit coordinate")
    if draws.duplicated(["dataset", "mass_MeV", "toy_id"]).any():
        raise RuntimeError("duplicate toy observation coordinate")
    if not (
        np.isfinite(limits.eps2_90.to_numpy(float)).all()
        and (limits.eps2_90 > 0.0).all()
        and limits.optimizer_ok.astype(bool).all()
        and np.allclose(limits.cls_at_limit, 0.1, rtol=0.0, atol=2.0e-6)
        and (limits.bracket_low_cls > 0.1).all()
        and (limits.bracket_high_cls <= 0.1).all()
    ):
        raise RuntimeError("one or more toy limits failed numerical gates")
    if draws.latent_clip_fallback.astype(bool).any():
        raise RuntimeError("a latent GP draw required the forbidden clip fallback")
    if set(limits.scope_key) != {item[0] for item in final.SCOPES}:
        raise RuntimeError("toy ledger does not contain exactly the seven scopes")
    for scope_key, _label, _keys, low, high in final.SCOPES:
        here = limits[limits.scope_key == scope_key]
        counts = here.groupby("mass_MeV").toy_id.nunique()
        if not (
            np.array_equal(counts.index.to_numpy(int), np.arange(low, high + 1))
            and (counts.to_numpy(int) == target_toys).all()
        ):
            raise RuntimeError(f"incomplete toy grid for {scope_key}")


def summarize(
    limits: pd.DataFrame, observed: pd.DataFrame, target_toys: int
) -> pd.DataFrame:
    observed_columns = observed[
        [
            "scope_key",
            "scope_label",
            "dataset_set",
            "mass_MeV",
            "mass_GeV",
            "eps2_90",
        ]
    ].rename(columns={"eps2_90": "eps2_observed"})
    rows: List[Dict[str, object]] = []
    for (scope_key, mass_mev), group in limits.groupby(
        ["scope_key", "mass_MeV"], sort=True
    ):
        values = group.sort_values("toy_id").eps2_90.to_numpy(float)
        quantiles = np.quantile(values, QUANTILE_PROBABILITIES, method="linear")
        row: Dict[str, object] = {
            "scope_key": str(scope_key),
            "mass_MeV": int(mass_mev),
            "n_toys": int(values.size),
            "toy_id_min": int(group.toy_id.min()),
            "toy_id_max": int(group.toy_id.max()),
            "expected_mean": float(np.mean(values)),
            "expected_std": float(np.std(values, ddof=1)),
            "quantile_method": "linear",
            "conditional_on_frozen_gp": True,
            "gp_refit_per_toy": False,
        }
        row.update(
            {name: float(value) for name, value in zip(QUANTILE_COLUMNS, quantiles)}
        )
        rows.append(row)
    summary = pd.DataFrame(rows).merge(
        observed_columns,
        on=["scope_key", "mass_MeV"],
        how="left",
        validate="one_to_one",
    )
    if summary.eps2_observed.isna().any():
        raise RuntimeError("expected-band rows did not join to every observed curve")
    percentile_rows = []
    for row in summary.itertuples(index=False):
        values = limits.loc[
            (limits.scope_key == row.scope_key)
            & (limits.mass_MeV == row.mass_MeV),
            "eps2_90",
        ].to_numpy(float)
        percentile_rows.append(float(np.mean(values <= row.eps2_observed)))
    summary["observed_empirical_cdf"] = percentile_rows
    ordered = [
        "scope_key",
        "scope_label",
        "dataset_set",
        "mass_MeV",
        "mass_GeV",
        "n_toys",
        "toy_id_min",
        "toy_id_max",
        "eps2_observed",
        *QUANTILE_COLUMNS,
        "expected_mean",
        "expected_std",
        "observed_empirical_cdf",
        "quantile_method",
        "conditional_on_frozen_gp",
        "gp_refit_per_toy",
    ]
    summary = summary[ordered].sort_values(
        ["scope_key", "mass_MeV"]
    ).reset_index(drop=True)
    q = summary[list(QUANTILE_COLUMNS)].to_numpy(float)
    if not np.all(np.diff(q, axis=1) >= 0.0):
        raise RuntimeError("expected-band quantiles are not ordered")
    if not (summary.n_toys == target_toys).all():
        raise RuntimeError("summary toy count does not equal the requested stage")
    return summary


def stage_paths(output: Path, target_toys: int) -> Dict[str, Path]:
    return {
        "limits": output / f"toy_limits_{target_toys}toys.csv",
        "draws": output / f"toy_observations_{target_toys}toys.csv",
        "predictions": output / "prediction_state_ledger.csv",
        "summary": output / f"expected_band_summary_{target_toys}toys.csv",
        "summary_current": output / "expected_band_summary.csv",
        "manifest": output / f"run_manifest_{target_toys}toys.json",
        "manifest_current": output / "run_manifest.json",
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    target_toys = int(args.target_toys)
    if target_toys <= 0 or target_toys > max(ALLOWED_RELEASE_STAGES):
        raise SystemExit("--target-toys must be between 1 and 300")
    mass_min = int(args.mass_min_mev)
    mass_max = int(args.mass_max_mev)
    if mass_min < 19 or mass_max > 250 or mass_min > mass_max:
        raise SystemExit("mass range must lie within the final 19--250 MeV grid")
    if int(args.workers) < 1:
        raise SystemExit("--workers must be positive")

    output = args.output_dir.expanduser().resolve()
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    digest, contract_payload = contract(
        card=DEFAULT_CARD,
        states=DEFAULT_STATES,
        provenance=DEFAULT_INPUT_PROVENANCE,
        observed=DEFAULT_OBSERVED,
        seed=int(args.seed),
    )
    final.atomic_json(
        output / "contract.json",
        {"contract_sha256": digest, **contract_payload},
    )
    requested = list(range(mass_min, mass_max + 1))
    completion = {
        mass: checkpoint_completion(
            checkpoint_dir / f"m{mass:03d}.json", digest, mass
        )
        for mass in requested
    }
    missing_by_mass = {
        mass: max(0, target_toys - complete)
        for mass, complete in completion.items()
    }
    print(
        json.dumps(
            {
                "contract_sha256": digest,
                "target_toys": target_toys,
                "mass_min_MeV": mass_min,
                "mass_max_MeV": mass_max,
                "masses_with_missing_work": int(
                    sum(value > 0 for value in missing_by_mass.values())
                ),
                "missing_mass_local_toys": int(sum(missing_by_mass.values())),
            },
            sort_keys=True,
        )
    )
    if args.plan:
        return

    final.result_config = load_config(DEFAULT_CARD)
    final.validate_card(final.result_config)
    input_provenance = final.validate_input_provenance(
        DEFAULT_INPUT_PROVENANCE,
        DEFAULT_CARD,
        DEFAULT_STATES,
        final.result_config,
    )
    histogram_inputs = final.validate_histogram_inputs(final.result_config)
    datasets = make_datasets(final.result_config)
    states = final.state_map(
        final.load_states(DEFAULT_STATES, final.result_config)
    )
    start = time.time()
    missing_masses = [mass for mass in requested if missing_by_mass[mass] > 0]
    if missing_masses:
        joblib.Parallel(
            n_jobs=min(2, max(1, int(args.workers))),
            backend="threading",
            verbose=10,
        )(
            joblib.delayed(run_mass)(
                mass,
                target_toys=target_toys,
                seed=int(args.seed),
                digest=digest,
                checkpoint_dir=checkpoint_dir,
                datasets=datasets,
                config=final.result_config,
                states=states,
            )
            for mass in missing_masses
        )

    if requested != list(range(19, 251)):
        print("Partial diagnostic grid complete; full-stage aggregation skipped.")
        return
    if target_toys not in ALLOWED_RELEASE_STAGES:
        print("Diagnostic toy target complete; release-stage aggregation skipped.")
        return

    checkpoint_paths = [checkpoint_dir / f"m{mass:03d}.json" for mass in requested]
    limits, draws, predictions = collect(checkpoint_paths, digest, target_toys)
    validate_raw_frames(limits, draws, predictions, target_toys)
    observed = pd.read_csv(DEFAULT_OBSERVED)
    summary = summarize(limits, observed, target_toys)
    paths = stage_paths(output, target_toys)
    final.atomic_csv(paths["limits"], limits)
    final.atomic_csv(paths["draws"], draws)
    final.atomic_csv(paths["predictions"], predictions)
    final.atomic_csv(paths["summary"], summary)
    final.atomic_csv(paths["summary_current"], summary)
    solver_counter_totals: Dict[str, int] = {}
    for encoded in limits.solver_counter_delta.astype(str):
        for key, value in json.loads(encoded).items():
            solver_counter_totals[key] = (
                int(solver_counter_totals.get(key, 0)) + int(value)
            )
    manifest: Dict[str, object] = {
        "status": "pointwise_conditional_expected_bands_complete",
        "stage_toys_per_mass": target_toys,
        "toy_id_range": [0, target_toys - 1],
        "allowed_release_stages": list(ALLOWED_RELEASE_STAGES),
        "next_stage": next(
            (value for value in ALLOWED_RELEASE_STAGES if value > target_toys), None
        ),
        "contract_sha256": digest,
        "master_seed": int(args.seed),
        "generator_version": GENERATOR_VERSION,
        "solver_version": SOLVER_VERSION,
        "mass_grid_MeV": [19, 250],
        "scope_count": len(final.SCOPES),
        "scope_mass_rows": len(summary),
        "toy_limit_rows": len(limits),
        "toy_observation_rows": len(draws),
        "prediction_rows": len(predictions),
        "latent_clip_fallbacks": int(draws.latent_clip_fallback.sum()),
        "solver_counter_totals": dict(sorted(solver_counter_totals.items())),
        "elapsed_seconds_this_invocation": float(time.time() - start),
        "conditional_on_frozen_gp": True,
        "gp_refit_per_toy": False,
        "pointwise_not_scanwide": True,
        "claim_boundary": (
            "Conditional pointwise background-only expected-limit quantiles; "
            "not unconditional coverage, toy-calibrated CLs, or global significance."
        ),
        "input_provenance_status": input_provenance.get("status"),
        "histogram_inputs": histogram_inputs,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        "artifacts_sha256": {
            key: sha256(path)
            for key, path in paths.items()
            if key not in {"manifest", "manifest_current"} and path.is_file()
        },
    }
    final.atomic_json(paths["manifest"], manifest)
    final.atomic_json(paths["manifest_current"], manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
