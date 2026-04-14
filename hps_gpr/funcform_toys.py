"""Functional-form toy discovery, loading, scan orchestration, and merging."""

from __future__ import annotations

import copy
import fnmatch
import json
import os
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import List, Optional, Sequence, TYPE_CHECKING

import hist
import numpy as np
import pandas as pd
import uproot

from .plotting import ensure_dir

if TYPE_CHECKING:
    from .config import Config
    from .dataset import DatasetConfig


@dataclass(frozen=True)
class FuncFormToySpec:
    """A single functional-form toy histogram to scan."""

    source_root: str
    container: str
    function_tag: str
    toy_name: str
    toy_index: int

    @property
    def output_tag(self) -> str:
        return f"toy_{int(self.toy_index):04d}"


def _clean_key_name(key: str) -> str:
    """Strip ROOT cycle suffix from a key name."""
    return str(key).split(";", 1)[0]


def _natural_sort_key(name: str) -> tuple:
    """Provide a stable natural sort for toy names."""
    parts = re.split(r"(\d+)", str(name))
    out = []
    for part in parts:
        if part.isdigit():
            out.append((0, int(part)))
        else:
            out.append((1, part))
    return tuple(out)


def _extract_toy_index(toy_name: str) -> Optional[int]:
    """Parse the trailing toy index from a histogram name, if present."""
    m = re.search(r"_toy_(\d+)$", str(toy_name))
    if m is None:
        return None
    return int(m.group(1))


def _infer_function_tag(container: Optional[str], toy_name: str) -> str:
    """Infer the function tag from the container or toy name."""
    if container:
        return os.path.basename(str(container).rstrip("/"))
    m = re.match(r"(.+?)_toy_\d+$", str(toy_name))
    if m:
        return str(m.group(1))
    return "funcform"


def _get_container(file_or_dir, container: Optional[str]):
    """Return the requested ROOT container object."""
    if container:
        return file_or_dir[str(container)]
    return file_or_dir


def _iter_hist_names(root_path: str, container: Optional[str] = None) -> List[str]:
    """Return histogram-like object names within a ROOT container."""
    with uproot.open(root_path) as f:
        obj = _get_container(f, container)
        try:
            keys = obj.keys(cycle=False)
        except TypeError:
            keys = obj.keys()
    return [_clean_key_name(k) for k in keys]


def discover_funcform_toys(
    root_path: str,
    *,
    container: Optional[str] = None,
    toy_pattern: str = "*",
    toy_name_fmt: Optional[str] = None,
    toy_indices: Optional[Sequence[int]] = None,
) -> List[FuncFormToySpec]:
    """Discover toy histograms inside a ROOT file."""
    root_path = str(root_path)
    container = str(container or "").strip()

    if toy_name_fmt is not None:
        idxs = [int(i) for i in (toy_indices or [])]
        specs: List[FuncFormToySpec] = []
        seen = set()
        for idx in idxs:
            name = str(toy_name_fmt).format(i=int(idx))
            if name in seen:
                continue
            seen.add(name)
            specs.append(
                FuncFormToySpec(
                    source_root=root_path,
                    container=container,
                    function_tag=_infer_function_tag(container, name),
                    toy_name=str(name),
                    toy_index=int(idx),
                )
            )
        return specs

    names = _iter_hist_names(root_path, container=container or None)
    names = [n for n in names if fnmatch.fnmatch(n, str(toy_pattern))]
    names = sorted(dict.fromkeys(names), key=_natural_sort_key)
    specs = []
    for fallback_index, name in enumerate(names):
        toy_index = _extract_toy_index(name)
        if toy_index is None:
            toy_index = int(fallback_index)
        specs.append(
            FuncFormToySpec(
                source_root=root_path,
                container=container,
                function_tag=_infer_function_tag(container, name),
                toy_name=str(name),
                toy_index=int(toy_index),
            )
        )
    return specs


def load_funcform_toy_hist(
    root_path: str,
    *,
    container: Optional[str] = None,
    toy_name: str,
) -> hist.Hist:
    """Load a single toy histogram from a ROOT file as a ``hist.Hist``."""
    with uproot.open(root_path) as f:
        obj = _get_container(f, container)
        hobj = obj[str(toy_name)]
        if hasattr(hobj, "to_hist"):
            hout = hobj.to_hist()
            try:
                hout.axes[0].label = "Mass / GeV"
            except Exception:
                pass
            return hout

        vals, edges = hobj.to_numpy()
        vals = np.asarray(vals, dtype=float)
        edges = np.asarray(edges, dtype=float)

    hout = hist.Hist(
        hist.axis.Variable(edges, label="Mass / GeV"),
        storage=hist.storage.Weight(),
    )
    view = hout.view()
    view.value[...] = vals
    view.variance[...] = np.clip(vals, 0.0, None)
    return hout


def build_funcform_toy_dataset(ds: "DatasetConfig", toy_hist: hist.Hist) -> "DatasetConfig":
    """Clone a dataset config and swap in an in-memory toy histogram."""
    return replace(ds, hist_override=toy_hist)


def _augment_scan_table(df: pd.DataFrame, spec: FuncFormToySpec) -> pd.DataFrame:
    """Add toy-identity columns to a scan result table."""
    if df is None:
        return pd.DataFrame()
    df = df.copy()
    df.insert(0, "toy_index", int(spec.toy_index))
    df.insert(1, "toy_hist", str(spec.toy_name))
    df.insert(2, "function_tag", str(spec.function_tag))
    df.insert(3, "source_root", str(spec.source_root))
    df.insert(4, "container", str(spec.container))
    return df


def _toy_output_dir(base_output_dir: str, dataset_key: str, spec: FuncFormToySpec) -> str:
    """Return the output directory for one toy scan."""
    return os.path.join(str(base_output_dir), "toy_scans", str(dataset_key), spec.output_tag)


def _write_toy_metadata(outdir: str, dataset_key: str, spec: FuncFormToySpec) -> str:
    """Write a JSON sidecar describing the toy scan."""
    ensure_dir(outdir)
    payload = asdict(spec)
    payload["dataset"] = str(dataset_key)
    payload["output_dir"] = str(outdir)
    path = os.path.join(outdir, "toy_metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


def run_funcform_toy_scans(
    ds: "DatasetConfig",
    config: "Config",
    specs: Sequence[FuncFormToySpec],
    *,
    base_output_dir: str,
    mass_min: Optional[float] = None,
    mass_max: Optional[float] = None,
    save_plots: Optional[bool] = None,
    save_fit_json: Optional[bool] = None,
    save_per_mass_folders: Optional[bool] = None,
    scan_parallel: Optional[bool] = None,
    scan_n_workers: Optional[int] = None,
    scan_parallel_backend: Optional[str] = None,
    scan_threads_per_worker: Optional[int] = None,
) -> List[str]:
    """Run the existing scan engine once per toy histogram."""
    from .scan import run_scan

    written: List[str] = []
    for spec in specs:
        toy_hist = load_funcform_toy_hist(
            spec.source_root,
            container=(spec.container or None),
            toy_name=spec.toy_name,
        )
        toy_ds = build_funcform_toy_dataset(ds, toy_hist)

        toy_cfg = copy.deepcopy(config)
        toy_cfg.output_dir = _toy_output_dir(base_output_dir, ds.key, spec)
        toy_cfg.save_plots = bool(
            getattr(config, "toy_scan_save_plots", False)
            if save_plots is None else save_plots
        )
        toy_cfg.save_fit_json = bool(
            getattr(config, "toy_scan_save_fit_json", False)
            if save_fit_json is None else save_fit_json
        )
        toy_cfg.save_per_mass_folders = bool(
            getattr(config, "toy_scan_save_per_mass_folders", False)
            if save_per_mass_folders is None else save_per_mass_folders
        )
        toy_cfg.scan_parallel = bool(
            getattr(config, "toy_scan_parallel", False)
            if scan_parallel is None else scan_parallel
        )
        toy_cfg.scan_n_workers = max(
            1,
            int(
                getattr(config, "toy_scan_n_workers", 1)
                if scan_n_workers is None else scan_n_workers
            ),
        )
        toy_cfg.scan_parallel_backend = str(
            getattr(config, "toy_scan_parallel_backend", "threading")
            if scan_parallel_backend is None else scan_parallel_backend
        )
        toy_cfg.scan_threads_per_worker = max(
            1,
            int(
                getattr(config, "toy_scan_threads_per_worker", 1)
                if scan_threads_per_worker is None else scan_threads_per_worker
            ),
        )
        toy_cfg.ensure_output_dir()

        df_single, df_comb = run_scan(
            {str(ds.key): toy_ds},
            toy_cfg,
            mass_min=mass_min,
            mass_max=mass_max,
        )
        df_single = _augment_scan_table(df_single, spec)
        df_comb = _augment_scan_table(df_comb, spec)

        single_path = os.path.join(toy_cfg.output_dir, "results_single.csv")
        comb_path = os.path.join(toy_cfg.output_dir, "results_combined.csv")
        alias_path = os.path.join(toy_cfg.output_dir, "combined.csv")

        df_single.to_csv(single_path, index=False)
        df_comb.to_csv(comb_path, index=False)
        df_comb.to_csv(alias_path, index=False)
        _write_toy_metadata(toy_cfg.output_dir, ds.key, spec)
        written.append(toy_cfg.output_dir)

    return written


def _load_toy_scan_frames(input_dir: str) -> List[pd.DataFrame]:
    """Load toy scan CSVs from a batch-output directory."""
    frames: List[pd.DataFrame] = []
    for meta_path in sorted(Path(input_dir).glob("**/toy_metadata.json")):
        outdir = meta_path.parent
        single_path = outdir / "results_single.csv"
        if not single_path.exists():
            continue
        try:
            df = pd.read_csv(single_path)
        except Exception:
            continue
        if df.empty:
            continue

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        for col, key in [
            ("toy_index", "toy_index"),
            ("toy_hist", "toy_name"),
            ("function_tag", "function_tag"),
            ("source_root", "source_root"),
            ("container", "container"),
            ("dataset", "dataset"),
        ]:
            if col not in df.columns:
                df[col] = meta.get(key, "")
        frames.append(df)
    return frames


def _toy_scan_inventory(input_dir: str) -> dict:
    """Collect simple counts describing a toy-scan output tree."""
    base = Path(input_dir)
    toy_dirs = [
        path for path in base.glob("**/toy_scans/*/toy_*")
        if path.is_dir()
    ]
    meta_paths = [path for path in base.glob("**/toy_metadata.json") if path.is_file()]
    single_paths = [path for path in base.glob("**/results_single.csv") if path.is_file()]
    comb_paths = [path for path in base.glob("**/results_combined.csv") if path.is_file()]
    return {
        "toy_dirs": int(len(toy_dirs)),
        "metadata_files": int(len(meta_paths)),
        "results_single_files": int(len(single_paths)),
        "results_combined_files": int(len(comb_paths)),
    }


def describe_toy_scan_inventory(input_dir: str, inventory: Optional[dict] = None) -> str:
    """Return a compact textual summary of toy-scan artifacts."""
    inv = inventory if inventory is not None else _toy_scan_inventory(input_dir)
    return (
        f"toy_dirs={inv['toy_dirs']}, "
        f"toy_metadata.json={inv['metadata_files']}, "
        f"results_single.csv={inv['results_single_files']}, "
        f"results_combined.csv={inv['results_combined_files']}"
    )


def _summarize_one_toy(group: pd.DataFrame) -> dict:
    """Build one compact per-toy summary row."""
    grp = group.copy()
    z = pd.to_numeric(grp.get("Z_analytic"), errors="coerce")
    p0 = pd.to_numeric(grp.get("p0_analytic"), errors="coerce")
    mass = pd.to_numeric(grp.get("mass_GeV"), errors="coerce")

    fail_mask = pd.Series(False, index=grp.index)
    if "extract_success" in grp.columns:
        fail_mask = fail_mask | (~grp["extract_success"].astype(bool))
    if "error" in grp.columns:
        fail_mask = fail_mask | grp["error"].astype(str).str.len().gt(0)
    fail_mask = fail_mask | (~np.isfinite(z)) | (~np.isfinite(p0))

    if np.isfinite(z).any():
        idx = int(np.nanargmax(z.to_numpy(float)))
        max_z = float(z.iloc[idx])
        mass_at_max = float(mass.iloc[idx]) if np.isfinite(mass.iloc[idx]) else float("nan")
    else:
        max_z = float("nan")
        mass_at_max = float("nan")

    min_p0 = float(np.nanmin(p0.to_numpy(float))) if np.isfinite(p0).any() else float("nan")

    return {
        "dataset": str(grp["dataset"].iloc[0]),
        "toy_index": int(grp["toy_index"].iloc[0]),
        "toy_hist": str(grp["toy_hist"].iloc[0]),
        "function_tag": str(grp["function_tag"].iloc[0]),
        "source_root": str(grp["source_root"].iloc[0]),
        "container": str(grp["container"].iloc[0]) if "container" in grp.columns else "",
        "n_fail": int(fail_mask.sum()),
        "max_Z_analytic": max_z,
        "mass_at_max_Z": mass_at_max,
        "min_p0_analytic": min_p0,
    }


def _toy_scan_slug(parts: Sequence[object]) -> str:
    """Build a filesystem-safe slug from a sequence of labels."""
    toks = []
    for part in parts:
        text = str(part).strip()
        if not text:
            continue
        clean = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
        if clean:
            toks.append(clean)
    return "_".join(toks)


def _toy_scan_row_quantile(arr: np.ndarray, q: float) -> np.ndarray:
    """Compute a row-wise quantile while tolerating all-NaN rows."""
    arr = np.asarray(arr, dtype=float)
    out = np.full(arr.shape[0], np.nan, dtype=float)
    for i in range(arr.shape[0]):
        row = arr[i]
        finite = row[np.isfinite(row)]
        if finite.size:
            out[i] = float(np.percentile(finite, float(q)))
    return out


def _toy_scan_describe_sample(toy_names: Sequence[str]) -> str:
    """Condense a toy-name list into a reviewer-friendly sample label."""
    names = [str(name) for name in toy_names if str(name).strip()]
    if not names:
        return "unknown"
    if len(names) <= 3:
        return ", ".join(names)
    return f"{names[0]} ... {names[-1]}"


def _toy_scan_group_metadata(merged_grp: pd.DataFrame, summary_grp: pd.DataFrame) -> dict:
    """Collect common metadata for one merged toy-scan group."""
    dataset = str(merged_grp["dataset"].iloc[0]) if "dataset" in merged_grp.columns else "dataset"
    function_tag = str(merged_grp["function_tag"].iloc[0]) if "function_tag" in merged_grp.columns else "funcform"
    container = str(merged_grp["container"].iloc[0]) if "container" in merged_grp.columns else ""
    source_root = str(merged_grp["source_root"].iloc[0]) if "source_root" in merged_grp.columns else ""
    source_name = Path(source_root).name if source_root else "unknown"

    toy_names = summary_grp["toy_hist"].astype(str).tolist() if "toy_hist" in summary_grp.columns else []
    toy_indices = (
        sorted(int(v) for v in summary_grp["toy_index"].dropna().astype(int).tolist())
        if "toy_index" in summary_grp.columns
        else []
    )

    mass_vals = merged_grp["mass_GeV"].dropna().to_numpy(float) if "mass_GeV" in merged_grp.columns else np.array([])
    mass_lo_mev = 1000.0 * float(np.nanmin(mass_vals)) if mass_vals.size else float("nan")
    mass_hi_mev = 1000.0 * float(np.nanmax(mass_vals)) if mass_vals.size else float("nan")

    if {"toy_index", "mass_GeV"}.issubset(merged_grp.columns):
        per_toy_counts = (
            merged_grp.groupby("toy_index", dropna=False)["mass_GeV"]
            .nunique()
            .to_numpy(dtype=float)
        )
        n_mass = int(np.nanmedian(per_toy_counts)) if per_toy_counts.size else int(merged_grp["mass_GeV"].nunique())
    else:
        n_mass = int(merged_grp["mass_GeV"].nunique()) if "mass_GeV" in merged_grp.columns else 0

    fail_total = (
        int(summary_grp["n_fail"].fillna(0).astype(int).sum())
        if "n_fail" in summary_grp.columns
        else int((~merged_grp.get("extract_success", pd.Series(dtype=bool)).fillna(False)).sum())
    )

    med_max_z = (
        float(np.nanmedian(summary_grp["max_Z_analytic"].to_numpy(float)))
        if "max_Z_analytic" in summary_grp.columns and len(summary_grp)
        else float("nan")
    )
    med_peak_mass_mev = (
        1000.0 * float(np.nanmedian(summary_grp["mass_at_max_Z"].to_numpy(float)))
        if "mass_at_max_Z" in summary_grp.columns and len(summary_grp)
        else float("nan")
    )
    med_min_p0 = (
        float(np.nanmedian(summary_grp["min_p0_analytic"].to_numpy(float)))
        if "min_p0_analytic" in summary_grp.columns and len(summary_grp)
        else float("nan")
    )

    return {
        "dataset": dataset,
        "function_tag": function_tag,
        "container": container,
        "source_root": source_root,
        "source_name": source_name,
        "toy_names": toy_names,
        "toy_indices": toy_indices,
        "toy_label": _toy_scan_describe_sample(toy_names),
        "n_toys": int(len(summary_grp)),
        "fail_total": fail_total,
        "n_mass": n_mass,
        "mass_lo_mev": mass_lo_mev,
        "mass_hi_mev": mass_hi_mev,
        "median_max_z": med_max_z,
        "median_peak_mass_mev": med_peak_mass_mev,
        "median_min_p0": med_min_p0,
    }


def _toy_scan_info_text(meta: dict) -> str:
    """Format an annotation box for the validation plots."""
    idxs = meta.get("toy_indices") or []
    if idxs:
        idx_text = f"{idxs[0]}-{idxs[-1]}" if len(idxs) > 1 else str(idxs[0])
    else:
        idx_text = "unknown"

    container = str(meta.get("container", "")).strip()
    source_label = str(meta.get("source_name", "unknown"))
    if container:
        source_label = f"{source_label} :: {container}"

    lines = [
        f"Dataset: {meta.get('dataset', 'dataset')}",
        f"Function family: {meta.get('function_tag', 'funcform')}",
        f"Toy sample: {meta.get('toy_label', 'unknown')}",
        f"Toy indices: {idx_text} ({int(meta.get('n_toys', 0))} toys)",
        (
            f"Mass grid: {meta.get('mass_lo_mev', float('nan')):.0f}-"
            f"{meta.get('mass_hi_mev', float('nan')):.0f} MeV "
            f"({int(meta.get('n_mass', 0))} hypotheses/toy)"
        ),
        f"Total failed mass fits: {int(meta.get('fail_total', 0))}",
        f"Source: {source_label}",
    ]

    med_max_z = float(meta.get("median_max_z", float("nan")))
    med_peak_mass_mev = float(meta.get("median_peak_mass_mev", float("nan")))
    med_min_p0 = float(meta.get("median_min_p0", float("nan")))
    if np.isfinite(med_max_z):
        lines.append(f"Median max Z: {med_max_z:.2f} @ {med_peak_mass_mev:.0f} MeV")
    if np.isfinite(med_min_p0):
        lines.append(f"Median min p0: {med_min_p0:.3g}")
    return "\n".join(lines)


def _save_toy_scan_plot(fig, stem: str) -> None:
    """Save a toy-scan validation figure as PNG and PDF."""
    fig.savefig(f"{stem}.png", dpi=220)
    fig.savefig(f"{stem}.pdf")


def write_toy_scan_validation_plots(
    merged: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: str,
    *,
    stem_prefix: str = "toy_scan_validation",
) -> list[str]:
    """Write reviewer-facing validation plots for merged toy-scan outputs."""
    import matplotlib as mpl

    mpl.use("Agg", force=True)

    import matplotlib.pyplot as plt

    from .plotting import set_plot_style

    ensure_dir(str(output_dir))
    if merged is None or summary is None or merged.empty or summary.empty:
        return []

    set_plot_style("paper")
    created_stems: list[str] = []

    group_cols = [c for c in ["dataset", "function_tag", "container", "source_root"] if c in merged.columns]
    if not group_cols:
        group_cols = [merged.columns[0]]

    for group_key, merged_grp in merged.groupby(group_cols, sort=True, dropna=False):
        key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
        summary_mask = np.ones(len(summary), dtype=bool)
        for col, val in zip(group_cols, key_tuple):
            if col not in summary.columns:
                continue
            if pd.isna(val):
                summary_mask &= summary[col].isna().to_numpy()
            else:
                summary_mask &= (summary[col].astype(str) == str(val)).to_numpy()
        summary_grp = summary.loc[summary_mask].copy()
        if summary_grp.empty:
            continue

        meta = _toy_scan_group_metadata(merged_grp, summary_grp)
        tag = _toy_scan_slug([meta["dataset"], meta["function_tag"]]) or "toy_scan"
        info_text = _toy_scan_info_text(meta)

        z_pivot = (
            merged_grp.pivot_table(index="mass_GeV", columns="toy_index", values="Z_analytic", aggfunc="first")
            .sort_index()
        )
        eps2_pivot = (
            merged_grp.pivot_table(index="mass_GeV", columns="toy_index", values="eps2_up", aggfunc="first")
            .sort_index()
        )

        if not z_pivot.empty:
            masses_mev = 1000.0 * z_pivot.index.to_numpy(float)
            z_vals = z_pivot.to_numpy(float)
            z_q16 = _toy_scan_row_quantile(z_vals, 16.0)
            z_q50 = _toy_scan_row_quantile(z_vals, 50.0)
            z_q84 = _toy_scan_row_quantile(z_vals, 84.0)

            fig, ax = plt.subplots(figsize=(9.4, 5.6), constrained_layout=True)
            for col in z_pivot.columns:
                ax.plot(masses_mev, z_pivot[col].to_numpy(float), color="0.72", alpha=0.70, lw=1.0)
            ax.fill_between(masses_mev, z_q16, z_q84, color="#9ecae1", alpha=0.60, label="central 68% band")
            ax.plot(masses_mev, z_q50, color="#08519c", lw=2.2, label="toy median")
            for zref in [1.0, 2.0, 3.0]:
                ax.axhline(zref, color="0.80", lw=0.9, ls=":")
            ax.set_xlabel("Mass hypothesis [MeV]")
            ax.set_ylabel(r"Local significance $Z$")
            ax.set_title(
                f"Toy-scan validation: {meta['dataset']} local-significance scans",
                pad=10.0,
            )
            ax.legend(loc="upper right", frameon=True)
            ax.text(
                0.02,
                0.98,
                info_text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.6,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75", alpha=0.95),
            )
            stem = os.path.join(str(output_dir), f"{stem_prefix}_{tag}_local_significance")
            _save_toy_scan_plot(fig, stem)
            plt.close(fig)
            created_stems.append(stem)

        if not eps2_pivot.empty:
            masses_mev = 1000.0 * eps2_pivot.index.to_numpy(float)
            eps2_vals = eps2_pivot.to_numpy(float)
            eps2_q16 = _toy_scan_row_quantile(eps2_vals, 16.0)
            eps2_q50 = _toy_scan_row_quantile(eps2_vals, 50.0)
            eps2_q84 = _toy_scan_row_quantile(eps2_vals, 84.0)

            fig, ax = plt.subplots(figsize=(9.4, 5.6), constrained_layout=True)
            for col in eps2_pivot.columns:
                ax.plot(masses_mev, eps2_pivot[col].to_numpy(float), color="0.72", alpha=0.70, lw=1.0)
            ax.fill_between(masses_mev, eps2_q16, eps2_q84, color="#fdd0a2", alpha=0.65, label="central 68% band")
            ax.plot(masses_mev, eps2_q50, color="#d94801", lw=2.2, label="toy median")
            ax.set_xlabel("Mass hypothesis [MeV]")
            ax.set_ylabel(r"Upper limit on $\epsilon^2$")
            ax.set_yscale("log")
            ax.set_title(
                f"Toy-scan validation: {meta['dataset']} upper-limit scans",
                pad=10.0,
            )
            ax.legend(loc="upper right", frameon=True)
            ax.text(
                0.02,
                0.98,
                info_text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.6,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75", alpha=0.95),
            )
            stem = os.path.join(str(output_dir), f"{stem_prefix}_{tag}_upper_limits")
            _save_toy_scan_plot(fig, stem)
            plt.close(fig)
            created_stems.append(stem)

        fig, axs = plt.subplots(2, 2, figsize=(10.4, 7.2), constrained_layout=True)
        ax_text, ax_scatter, ax_mass, ax_p0 = axs.flat
        ax_text.axis("off")
        ax_text.text(
            0.02,
            0.98,
            info_text,
            ha="left",
            va="top",
            fontsize=9.0,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.75", alpha=0.98),
            transform=ax_text.transAxes,
        )
        ax_text.text(
            0.02,
            0.12,
            "Reviewer-facing smoke-test summary:\n"
            "use this dashboard to confirm toy identity, scan coverage,\n"
            "and the size/location of the strongest local upward fluctuation.",
            ha="left",
            va="bottom",
            fontsize=8.6,
            transform=ax_text.transAxes,
        )

        scatter_x = summary_grp["toy_index"].to_numpy(int)
        scatter_y = summary_grp["max_Z_analytic"].to_numpy(float)
        scatter_c = 1000.0 * summary_grp["mass_at_max_Z"].to_numpy(float)
        sc = ax_scatter.scatter(
            scatter_x,
            scatter_y,
            c=scatter_c,
            cmap="viridis",
            s=78,
            edgecolors="black",
            linewidths=0.45,
        )
        offset = 0.04 * max(1.0, float(np.nanmax(scatter_y)) if np.isfinite(scatter_y).any() else 1.0)
        for row in summary_grp.itertuples(index=False):
            ax_scatter.text(
                int(row.toy_index),
                float(row.max_Z_analytic) + offset,
                str(int(row.toy_index)),
                ha="center",
                va="bottom",
                fontsize=8.0,
            )
        ax_scatter.set_xlabel("Toy index")
        ax_scatter.set_ylabel(r"Max local significance $Z_{\max}$")
        ax_scatter.set_title("Peak upward fluctuation by toy")
        if len(scatter_x):
            ax_scatter.set_xticks(sorted(np.unique(scatter_x)))
        cbar = fig.colorbar(sc, ax=ax_scatter)
        cbar.set_label("Mass at max Z [MeV]")

        peak_masses_mev = 1000.0 * summary_grp["mass_at_max_Z"].to_numpy(float)
        ax_mass.hist(
            peak_masses_mev[np.isfinite(peak_masses_mev)],
            bins=min(10, max(4, len(summary_grp))),
            color="#6baed6",
            edgecolor="white",
        )
        ax_mass.axvline(
            float(np.nanmedian(peak_masses_mev)),
            color="#08519c",
            lw=1.8,
            ls="--",
            label="median",
        )
        ax_mass.set_xlabel(r"Mass at max $Z$ [MeV]")
        ax_mass.set_ylabel("Toys")
        ax_mass.set_title("Where the largest fluctuation lands")
        ax_mass.legend(loc="upper right", frameon=True)

        min_p0 = np.clip(summary_grp["min_p0_analytic"].to_numpy(float), 1.0e-12, 1.0)
        neglog_p0 = -np.log10(min_p0)
        ax_p0.hist(
            neglog_p0[np.isfinite(neglog_p0)],
            bins=min(10, max(4, len(summary_grp))),
            color="#fd8d3c",
            edgecolor="white",
        )
        ax_p0.axvline(
            float(np.nanmedian(neglog_p0)),
            color="#a63603",
            lw=1.8,
            ls="--",
            label="median",
        )
        ax_p0.set_xlabel(r"$-\log_{10}(\min p_0)$")
        ax_p0.set_ylabel("Toys")
        ax_p0.set_title("Smallest local p-value per toy")
        ax_p0.legend(loc="upper right", frameon=True)

        fig.suptitle(
            f"Toy-scan validation summary: {meta['dataset']} {meta['function_tag']}",
            y=1.01,
        )
        stem = os.path.join(str(output_dir), f"{stem_prefix}_{tag}_summary")
        _save_toy_scan_plot(fig, stem)
        plt.close(fig)
        created_stems.append(stem)

    return created_stems


def merge_toy_scan_results(
    input_dir: str,
    *,
    output_dir: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge per-toy scan outputs into long and compact summary CSVs."""
    input_dir = str(input_dir)
    outdir = str(output_dir or input_dir)
    ensure_dir(outdir)

    inventory = _toy_scan_inventory(input_dir)
    frames = _load_toy_scan_frames(input_dir)
    if not frames:
        raise FileNotFoundError(
            "No toy scan results found under "
            f"{input_dir} ({describe_toy_scan_inventory(input_dir, inventory)})"
        )

    merged = pd.concat(frames, ignore_index=True)
    sort_cols = [c for c in ["dataset", "function_tag", "toy_index", "mass_GeV"] if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols).reset_index(drop=True)

    summary_rows = []
    group_cols = ["dataset", "function_tag", "toy_index", "toy_hist", "source_root"]
    if "container" in merged.columns:
        group_cols.append("container")
    for _, grp in merged.groupby(group_cols, sort=True, dropna=False):
        summary_rows.append(_summarize_one_toy(grp))
    summary = pd.DataFrame(summary_rows).sort_values(
        ["dataset", "function_tag", "toy_index"]
    ).reset_index(drop=True)

    merged_path = os.path.join(outdir, "toy_scan_merged.csv")
    summary_path = os.path.join(outdir, "toy_scan_summary.csv")
    merged.to_csv(merged_path, index=False)
    summary.to_csv(summary_path, index=False)
    return merged, summary
