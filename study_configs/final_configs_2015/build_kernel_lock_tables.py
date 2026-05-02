#!/usr/bin/env python3
"""Build cross-fit fixed-kernel lock tables for 2015 refit diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_OUTDIR = Path("/sdf/data/hps/users/epeets/run/gpr_out/2015_closure/funcform_studies/kernel_lock_tables")


def _read_inputs(paths: list[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        frames.append(pd.read_csv(p))
    if not frames:
        raise ValueError("At least one input CSV is required")
    return pd.concat(frames, ignore_index=True, sort=False)


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(f"Missing required column; looked for {candidates}")


def _prepare(df: pd.DataFrame, *, require_zero_injection: bool = True) -> pd.DataFrame:
    if require_zero_injection and "inj_nsigma" in df.columns:
        inj = pd.to_numeric(df["inj_nsigma"], errors="coerce").to_numpy(float)
        zero = np.isfinite(inj) & (np.abs(inj) <= 1e-9)
        if np.any(zero):
            df = df.loc[zero].copy()

    dataset_col = _pick_column(df, ("dataset",))
    mass_col = _pick_column(df, ("mass_GeV",))
    toy_col = _pick_column(df, ("toy_index", "toy"))
    const_col = _pick_column(df, ("initial_const_opt", "const_opt", "refit_const_opt"))
    ls_col = _pick_column(df, ("initial_ls_opt", "ls_opt", "refit_ls_opt"))

    work = pd.DataFrame(
        {
            "dataset": df[dataset_col].astype(str),
            "mass_GeV": pd.to_numeric(df[mass_col], errors="coerce"),
            "toy_index": pd.to_numeric(df[toy_col], errors="coerce"),
            "const_opt": pd.to_numeric(df[const_col], errors="coerce"),
            "ls_opt": pd.to_numeric(df[ls_col], errors="coerce"),
        }
    )
    if "refit_ok" in df.columns:
        work["refit_ok"] = pd.to_numeric(df["refit_ok"], errors="coerce")
    elif "success" in df.columns:
        work["refit_ok"] = pd.to_numeric(df["success"], errors="coerce")
    else:
        work["refit_ok"] = 1.0

    ok = (
        np.isfinite(work["mass_GeV"].to_numpy(float))
        & np.isfinite(work["toy_index"].to_numpy(float))
        & np.isfinite(work["const_opt"].to_numpy(float))
        & np.isfinite(work["ls_opt"].to_numpy(float))
        & (work["const_opt"].to_numpy(float) > 0.0)
        & (work["ls_opt"].to_numpy(float) > 0.0)
        & (pd.to_numeric(work["refit_ok"], errors="coerce").fillna(0.0).to_numpy(float) > 0.0)
    )
    work = work[ok].copy()
    if work.empty:
        raise ValueError("No finite successful rows remained after filtering")
    work["mass_GeV"] = work["mass_GeV"].round(12)
    work["toy_index"] = work["toy_index"].round().astype(int)
    return work.drop_duplicates(["dataset", "mass_GeV", "toy_index"], keep="first")


def _quantile(values: pd.Series, q: float) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if arr.size else float("nan")


def _build_table(work: pd.DataFrame, *, n_folds: int, ls_quantile: float, label: str) -> pd.DataFrame:
    rows = []
    n_folds = int(max(2, n_folds))
    for (dataset, mass), sub in work.groupby(["dataset", "mass_GeV"], dropna=False):
        toys = sorted(int(t) for t in sub["toy_index"].unique())
        for toy in toys:
            fold = int(toy) % n_folds
            train = sub[(sub["toy_index"].astype(int) % n_folds) != fold]
            if train.empty:
                train = sub[sub["toy_index"].astype(int) != int(toy)]
            if train.empty:
                train = sub
            const = _quantile(train["const_opt"], 0.50)
            ls = _quantile(train["ls_opt"], float(ls_quantile))
            if not (np.isfinite(const) and const > 0.0 and np.isfinite(ls) and ls > 0.0):
                continue
            rows.append(
                {
                    "dataset": str(dataset),
                    "mass_GeV": float(mass),
                    "toy_index": int(toy),
                    "const_opt": float(const),
                    "ls_opt": float(ls),
                    "lock_quantile": str(label),
                    "n_train_lock_rows": int(len(train)),
                    "n_folds": int(n_folds),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f"No rows produced for lock table {label}")
    return out.sort_values(["dataset", "mass_GeV", "toy_index"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Input toy or summary CSVs containing B-only initial GP hyperparameters.")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="Directory for generated lock CSVs.")
    parser.add_argument("--n-folds", type=int, default=5, help="Toy-index folds for cross-fit lock values.")
    parser.add_argument(
        "--allow-nonzero-injection-rows",
        action="store_true",
        help="Do not filter to inj_nsigma==0 when that column is present.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    work = _prepare(
        _read_inputs(args.inputs),
        require_zero_injection=not bool(args.allow_nonzero_injection_rows),
    )

    outputs = {
        "kernel_lock_p50_crossfit.csv": _build_table(work, n_folds=args.n_folds, ls_quantile=0.50, label="p50"),
        "kernel_lock_p75ls_crossfit.csv": _build_table(work, n_folds=args.n_folds, ls_quantile=0.75, label="p75ls"),
        "kernel_lock_p25ls_crossfit.csv": _build_table(work, n_folds=args.n_folds, ls_quantile=0.25, label="p25ls"),
    }
    for name, table in outputs.items():
        path = outdir / name
        table.to_csv(path, index=False)
        print(f"Wrote {path} ({len(table)} rows)")


if __name__ == "__main__":
    main()
