#!/usr/bin/env python3
"""Build paired GP-mean replacement ensembles for two distinct window grids."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-mpl")

HERE = Path(__file__).resolve().parent
PARENT_STUDY = HERE.parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import scipy
import sklearn
import uproot
import yaml

from hps_gpr.config import load_config
from hps_gpr.dataset import make_datasets
from hps_gpr.gpr import (
    fit_gpr,
    make_fixed_kernel,
    predict_counts_mean_from_log_gpr,
)
from hps_gpr.io import _build_model


MASS_GEV = 0.065
REQUESTED_NSIGMA = (2.25, 2.5, 3.0)
MASTER_SEED = 2026080603
N_DRAWS = 10
SOURCE_ROOT = Path(
    "/Users/emryspeets/Desktop/gp_mods/10pct_2021/final_10pct_invM.root"
)
SOURCE_KEY = "preselection/h_invM_8000"
EXPECTED_SOURCE_SHA256 = (
    "3944d4c71a453c6c810061248c34d2fca9eceaad1de85c137afdc291c2195ac4"
)
EXPECTED_STATE_LEDGER_SHA256 = (
    "a962c01aa030429c04e2cc102253b6b8750eacc3c9e294a7a99f851a9870aea9"
)
EXPECTED_SELECTED_SOURCE_SHA256 = (
    "fb065cd988534049027c8e3c255b341b97f9ed630b9a27e698ba7452a7f67dcc"
)
PARENT_CONFIG = (
    REPO
    / "study_configs"
    / "v4p2_combined_2015full_2016full_2021_10pct_300toy_20260805"
    / "config_obsUL90_combined_wide_support_v4p2_2016k12_combined300.yaml"
)
STATE_LEDGER = (
    REPO
    / "study_results"
    / "v4p1_2016_ls_upper_optimization_20260804"
    / "derived"
    / "observed_gp_states_k12_reviewed.csv"
)
TEMPLATE_CONFIG = (
    PARENT_STUDY
    / "configs"
    / "config_obsUL90_2021_10pct_gpmean_replacement_v4p2.yaml"
)
PARENT_REPLACEMENT_ROOT = (
    PARENT_STUDY / "inputs" / "pseudo65_background_replacements.root"
)
PARENT_EXPECTATION_KEY = "expectations/gp_mean_m065"
OUT_ROOT = HERE / "inputs" / "gp_window_ensemble.root"
PROVENANCE_JSON = HERE / "derived" / "input_provenance.json"
CONFIG_MANIFEST_JSON = HERE / "derived" / "config_manifest.json"

WINDOWS = {
    "window_2p25eq2p5": {
        "requested_nsigma": [2.25, 2.5],
        "label": "+/-2.25 sigma and +/-2.5 sigma (same selected bins)",
    },
    "window_3p0": {
        "requested_nsigma": [3.0],
        "label": "+/-3 sigma",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    digest.update(arr.tobytes())
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def select_geometry(
    coarse_edges: np.ndarray,
    sigma: float,
    nsigma: float,
) -> dict[str, Any]:
    centers = 0.5 * (coarse_edges[:-1] + coarse_edges[1:])
    physical_lo = MASS_GEV - nsigma * sigma
    physical_hi = MASS_GEV + nsigma * sigma
    indices = np.where((centers >= physical_lo) & (centers <= physical_hi))[0]
    if len(indices) == 0 or not np.all(np.diff(indices) == 1):
        raise RuntimeError(f"Invalid coarse-bin selection for {nsigma} sigma")
    return {
        "requested_nsigma": float(nsigma),
        "continuous_interval_GeV": [float(physical_lo), float(physical_hi)],
        "coarse_indices": indices,
        "coarse_center_interval_GeV": [
            float(centers[indices[0]]),
            float(centers[indices[-1]]),
        ],
        "complete_bin_interval_GeV": [
            float(coarse_edges[indices[0]]),
            float(coarse_edges[indices[-1] + 1]),
        ],
        "n_analysis_bins": int(len(indices)),
    }


def reconstruct_fixed_gp(
    source_values: np.ndarray,
    source_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[float, dict[str, Any]]]:
    cfg = load_config(str(PARENT_CONFIG))
    datasets = make_datasets(cfg)
    ds = datasets["2021"]
    states = pd.read_csv(STATE_LEDGER)
    rows = states[
        (states["dataset"].astype(str) == "2021")
        & np.isclose(states["mass_GeV"].to_numpy(float), MASS_GEV, atol=5.0e-10)
    ]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one accepted 2021 state at 65 MeV, found {len(rows)}")
    row = rows.iloc[0]

    sigma = float(ds.sigma(MASS_GEV))
    blind = (
        MASS_GEV - float(cfg.blind_nsigma) * sigma,
        MASS_GEV + float(cfg.blind_nsigma) * sigma,
    )
    model = _build_model(
        ds,
        blind,
        rebin=int(cfg.neighborhood_rebin),
        config=cfg,
        mass=MASS_GEV,
    )
    coarse_edges = np.asarray(model.histogram.axes[0].edges, float)
    coarse_centers = np.asarray(model.histogram.axes[0].centers, float)
    coarse_values = np.asarray(model.histogram.values(), float)
    train_mask = (coarse_centers < blind[0]) | (coarse_centers > blind[1])
    fixed_kernel = make_fixed_kernel(float(row["const_opt"]), float(row["ls_opt"]))
    gpr = fit_gpr(
        coarse_centers[train_mask],
        coarse_values[train_mask],
        cfg,
        restarts=0,
        kernel=fixed_kernel,
        optimize=False,
    )
    reconstructed_lml = float(gpr.log_marginal_likelihood_value_)
    if not np.isclose(
        reconstructed_lml, float(row["lml"]), atol=3.0e-5, rtol=0.0
    ):
        raise RuntimeError(
            f"Fixed-state LML mismatch: {reconstructed_lml} vs {float(row['lml'])}"
        )
    coarse_mean = predict_counts_mean_from_log_gpr(gpr, coarse_centers, cfg)

    geometries = {
        nsigma: select_geometry(coarse_edges, sigma, nsigma)
        for nsigma in REQUESTED_NSIGMA
    }
    if not np.array_equal(
        geometries[2.25]["coarse_indices"],
        geometries[2.5]["coarse_indices"],
    ):
        raise RuntimeError("Expected 2.25 and 2.5 sigma grids to be identical")
    if np.array_equal(
        geometries[2.5]["coarse_indices"],
        geometries[3.0]["coarse_indices"],
    ):
        raise RuntimeError("Expected 3 sigma grid to be distinct")

    native_centers = 0.5 * (source_edges[:-1] + source_edges[1:])
    expectation = np.zeros_like(source_values, dtype=float)
    group_records = []
    wide_indices = geometries[3.0]["coarse_indices"]
    for coarse_index in wide_indices:
        coarse_lo = float(coarse_edges[coarse_index])
        coarse_hi = float(coarse_edges[coarse_index + 1])
        group = np.where(
            (native_centers >= coarse_lo - 1.0e-14)
            & (native_centers < coarse_hi - 1.0e-14)
        )[0]
        if len(group) != int(cfg.neighborhood_rebin):
            raise RuntimeError(
                f"Expected five native bins in [{coarse_lo},{coarse_hi}), found {len(group)}"
            )
        relative = np.asarray(
            predict_counts_mean_from_log_gpr(gpr, native_centers[group], cfg),
            float,
        )
        native_mean = float(coarse_mean[coarse_index]) * relative / np.sum(relative)
        expectation[group] = native_mean
        group_records.append(
            {
                "coarse_index": int(coarse_index),
                "coarse_interval_GeV": [coarse_lo, coarse_hi],
                "coarse_mean": float(coarse_mean[coarse_index]),
                "native_mean_sum": float(np.sum(native_mean)),
            }
        )

    parent_expectation, parent_edges = uproot.open(PARENT_REPLACEMENT_ROOT)[
        PARENT_EXPECTATION_KEY
    ].to_numpy(flow=False)
    if not np.array_equal(parent_edges, source_edges):
        raise RuntimeError("Parent expectation edges differ from source")
    narrow_lo, narrow_hi = geometries[2.5]["complete_bin_interval_GeV"]
    narrow_mask = (native_centers >= narrow_lo) & (native_centers < narrow_hi)
    if not np.allclose(
        expectation[narrow_mask],
        np.asarray(parent_expectation, float)[narrow_mask],
        atol=1.0e-9,
        rtol=2.0e-13,
    ):
        raise RuntimeError("Narrow expectation does not reproduce frozen pseudo65 input")

    gp_info = {
        "source": "exact accepted v4.2 2021 fixed GP state at 65 MeV",
        "parent_config": repo_relative(PARENT_CONFIG),
        "parent_config_sha256": sha256_file(PARENT_CONFIG),
        "state_ledger": repo_relative(STATE_LEDGER),
        "state_ledger_sha256": sha256_file(STATE_LEDGER),
        "expected_state_ledger_sha256": EXPECTED_STATE_LEDGER_SHA256,
        "mass_GeV": MASS_GEV,
        "sigma_GeV": sigma,
        "v42_blind_interval_GeV": [float(blind[0]), float(blind[1])],
        "const_opt": float(row["const_opt"]),
        "ls_opt": float(row["ls_opt"]),
        "reviewed_lml": float(row["lml"]),
        "reconstructed_lml": reconstructed_lml,
        "selected_source": str(row["selected_source"]),
        "selected_source_sha256": str(row["selected_source_sha256"]),
        "expected_selected_source_sha256": EXPECTED_SELECTED_SOURCE_SHA256,
        "selected_review_status": str(row["review_status"]),
        "n_coarse_training_bins": int(np.count_nonzero(train_mask)),
        "wide_group_checks": group_records,
        "parent_narrow_expectation_reproduced": True,
        "wide_added_bins_relation_to_fixed_state": (
            "The four 0.625 MeV bins added by the 3-sigma replacement are "
            "training-sideband bins in the accepted 2.25-sigma-exclusion GP "
            "state; no 3-sigma-exclusion generating GP is refitted."
        ),
    }
    return expectation, native_centers, gp_info, geometries


def build_configs(root_keys: dict[str, list[str]]) -> dict[str, Any]:
    template = yaml.safe_load(TEMPLATE_CONFIG.read_text())
    records = []
    (HERE / "configs").mkdir(parents=True, exist_ok=True)
    for window, keys in root_keys.items():
        for draw_index, hist_key in enumerate(keys):
            cfg = deepcopy(template)
            cfg["path_2021"] = repo_relative(OUT_ROOT)
            cfg["hist_2021"] = hist_key
            cfg["scan_n_workers"] = 5
            output_dir = HERE / "runs" / window / f"draw_{draw_index:02d}" / "attempt_01"
            cfg["output_dir"] = repo_relative(output_dir)
            config_path = HERE / "configs" / f"config_{window}_draw_{draw_index:02d}.yaml"
            config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            records.append(
                {
                    "window": window,
                    "draw_index": draw_index,
                    "hist_key": hist_key,
                    "config": repo_relative(config_path),
                    "config_sha256": sha256_file(config_path),
                    "output_dir": repo_relative(output_dir),
                    "allowed_template_changes": [
                        "path_2021",
                        "hist_2021",
                        "output_dir",
                        "scan_n_workers",
                    ],
                }
            )
    manifest = {
        "schema_version": 1,
        "template_config": repo_relative(TEMPLATE_CONFIG),
        "template_config_sha256": sha256_file(TEMPLATE_CONFIG),
        "generated_config_count": len(records),
        "records": records,
    }
    CONFIG_MANIFEST_JSON.write_text(
        json.dumps(json_safe(manifest), indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    for directory in ("inputs", "configs", "derived", "runs", "plots"):
        (HERE / directory).mkdir(parents=True, exist_ok=True)

    source_sha = sha256_file(SOURCE_ROOT)
    if source_sha != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            f"2021 input SHA256 mismatch: {source_sha} != {EXPECTED_SOURCE_SHA256}"
        )
    source_hist = uproot.open(SOURCE_ROOT)[SOURCE_KEY]
    source_values, source_edges = source_hist.to_numpy(flow=False)
    source_values = np.asarray(source_values, float)
    source_edges = np.asarray(source_edges, float)
    if (
        len(source_values) != 8000
        or not np.all(source_values == np.rint(source_values))
        or not np.allclose(
            np.diff(source_edges), 0.000125, atol=2.0e-16, rtol=0.0
        )
    ):
        raise RuntimeError("Unexpected source histogram geometry or contents")

    expectation, native_centers, gp_info, geometries = reconstruct_fixed_gp(
        source_values, source_edges
    )
    window_geometry = {
        "window_2p25eq2p5": geometries[2.5],
        "window_3p0": geometries[3.0],
    }
    window_masks = {}
    expectation_keys = {}
    for window, geometry in window_geometry.items():
        lo, hi = geometry["complete_bin_interval_GeV"]
        mask = (native_centers >= lo) & (native_centers < hi)
        window_masks[window] = mask
        expectation_keys[window] = f"expectations/{window}/gp_mean_m065"
        geometry["n_native_bins"] = int(np.count_nonzero(mask))

    seed_sequence = np.random.SeedSequence(MASTER_SEED)
    children = seed_sequence.spawn(N_DRAWS)
    root_keys = {window: [] for window in WINDOWS}
    draw_records = []
    root_payload: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "source/preselection/h_invM_8000": (source_values, source_edges)
    }
    for window, expectation_key in expectation_keys.items():
        exp_values = np.zeros_like(expectation)
        exp_values[window_masks[window]] = expectation[window_masks[window]]
        root_payload[expectation_key] = (exp_values, source_edges)

    for draw_index, child in enumerate(children):
        rng = np.random.Generator(np.random.PCG64(child))
        wide_mask = window_masks["window_3p0"]
        wide_poisson = rng.poisson(expectation[wide_mask]).astype(np.int64)
        wide_values = source_values.copy()
        wide_values[wide_mask] = wide_poisson

        narrow_mask = window_masks["window_2p25eq2p5"]
        narrow_values = source_values.copy()
        narrow_values[narrow_mask] = wide_values[narrow_mask]
        if not np.array_equal(
            narrow_values[narrow_mask], wide_values[narrow_mask]
        ):
            raise RuntimeError("Paired common counts failed in overlapping bins")

        values_by_window = {
            "window_2p25eq2p5": narrow_values,
            "window_3p0": wide_values,
        }
        for window, values in values_by_window.items():
            key = (
                f"gp/{window}/draw_{draw_index:02d}/"
                "preselection/h_invM_8000"
            )
            root_keys[window].append(key)
            root_payload[key] = (values, source_edges)
            mask = window_masks[window]
            if not np.array_equal(values[~mask], source_values[~mask]):
                raise RuntimeError(f"{window} draw {draw_index} changed outside bins")
            draw_records.append(
                {
                    "window": window,
                    "draw_index": draw_index,
                    "root_key": key,
                    "child_seed_state": child.state,
                    "replacement_draw_sha256": sha256_array(values[mask]),
                    "replacement_draw_sum": int(np.sum(values[mask])),
                    "full_histogram_sha256": sha256_array(values),
                    "paired_overlap_sha256": sha256_array(values[narrow_mask]),
                }
            )

    metadata = {
        "study": "v4p2 conditional GP-mean replacement-window ensemble",
        "Ainj": 0.0,
        "master_seed": MASTER_SEED,
        "n_draws_per_unique_geometry": N_DRAWS,
        "paired_common_random_numbers": True,
        "windows": {
            window: {
                **{
                    key: value
                    for key, value in geometry.items()
                    if key
                    not in (
                        "coarse_indices",
                        "requested_nsigma",
                        "continuous_interval_GeV",
                    )
                },
                **WINDOWS[window],
                "requested_continuous_intervals_GeV": {
                    str(nsigma): geometries[float(nsigma)][
                        "continuous_interval_GeV"
                    ]
                    for nsigma in WINDOWS[window]["requested_nsigma"]
                },
                "representative_grid_geometry_nsigma": (
                    2.5 if window == "window_2p25eq2p5" else 3.0
                ),
                "expectation_key": expectation_keys[window],
                "draw_keys": root_keys[window],
            }
            for window, geometry in window_geometry.items()
        },
    }
    with uproot.recreate(OUT_ROOT) as root_file:
        for key, payload in root_payload.items():
            root_file[key] = payload
        root_file["metadata/json"] = json.dumps(
            json_safe(metadata), sort_keys=True, separators=(",", ":")
        )

    config_manifest = build_configs(root_keys)
    provenance = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_commit": git_commit(),
        "parent_study": repo_relative(PARENT_STUDY),
        "interpretation": (
            "conditional background-only GP-mean replacements; outside-window "
            "observations are retained; descriptive ten-draw summaries only"
        ),
        "source": {
            "path": str(SOURCE_ROOT),
            "key": SOURCE_KEY,
            "sha256": source_sha,
            "n_bins": int(len(source_values)),
            "native_bin_width_GeV": float(np.median(np.diff(source_edges))),
        },
        "gp_fixed_state": gp_info,
        "randomization": {
            "distribution": "independent binwise Poisson",
            "master_seed": MASTER_SEED,
            "bit_generator": "PCG64",
            "n_independent_child_streams": N_DRAWS,
            "paired_window_construction": (
                "For each draw index, generate the full 3-sigma window once and "
                "reuse its [60,70) MeV counts in the shared 2.25/2.5-sigma window."
            ),
        },
        "windows": metadata["windows"],
        "draws": draw_records,
        "output": {
            "root_path": repo_relative(OUT_ROOT),
            "root_sha256": sha256_file(OUT_ROOT),
            "source_key": "source/preselection/h_invM_8000",
            "root_key_count": len(root_payload),
        },
        "config_manifest": {
            "path": repo_relative(CONFIG_MANIFEST_JSON),
            "sha256": sha256_file(CONFIG_MANIFEST_JSON),
            "generated_config_count": config_manifest["generated_config_count"],
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "uproot": uproot.__version__,
            "pyyaml": yaml.__version__,
        },
        "interpretation_boundary": (
            "This is not an independent global-null ensemble and supplies no "
            "expected-limit band, coverage statement, or scan-calibrated global p-value."
        ),
    }
    PROVENANCE_JSON.write_text(
        json.dumps(json_safe(provenance), indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote {OUT_ROOT}")
    print(f"Wrote {PROVENANCE_JSON}")
    print(f"Wrote {CONFIG_MANIFEST_JSON}")
    for window, geometry in window_geometry.items():
        lo, hi = geometry["complete_bin_interval_GeV"]
        print(
            f"{window}: {geometry['n_analysis_bins']} analysis bins, "
            f"{geometry['n_native_bins']} native bins, [{lo:.5f},{hi:.5f}) GeV"
        )


if __name__ == "__main__":
    main()
