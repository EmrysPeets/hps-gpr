from __future__ import annotations

import argparse
import json
from pathlib import Path

import fitz
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import uproot
import yaml

from hps_gpr.funcform_toys import resolve_funcform_toy_root_path
from hps_gpr.statistics import bounded_two_sided_tail_pvalue


NOTE_DIR = Path(__file__).resolve().parents[1]
FUNCFORM_DATASETS = ["2015", "2016", "2021"]
FUNCFORM_TITLES = {
    "2015": "HPS 2015",
    "2016": "HPS 2016 10%",
    "2021": "HPS 2021 1%",
}
FUNCFORM_SCAN_RANGE_OVERRIDES_MEV = {
    "2016": (42.0, 210.0),
}
FUNCFORM_COLORS = {
    "data": "#222222",
    "fit": "#C44E52",
    "toy": "#4C72B0",
    "sideband": "#D9D9D9",
}
FUNCFORM_CANDIDATE_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#8172B3", "#937860"]
FUNCFORM_FORMULA_TEXT = {
    "fShiftSigPowTail": (
        r"$f(x)=A\,S(x;x_t,w)\,(x-x_0)^a \exp(-(x-x_0)/\theta)\,\exp(c_1 u + c_2 u^2)$",
        r"$u=(x-x_{\rm mid})/x_{\rm scale}$, with $f(x)=0$ for $x\leq x_0$.",
    ),
    "fSigPowExpQ": (
        r"$f(x)=A\,S(x;x_t,w)\,x^a \exp(-x/\theta)\,\exp(c_1 x + c_2 x^2)$",
        r"Threshold factor $S(x;x_t,w)=\left[1+\exp(-(x-x_t)/w)\right]^{-1}$.",
    ),
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, out_path: Path, *, dpi: int = 220) -> None:
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if out_path.suffix.lower() == ".png":
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")


def crop_pdf(
    pdf_path: Path,
    page_index: int,
    crop_fracs: tuple[float, float, float, float],
    out_path: Path,
    *,
    scale: float = 4.0,
) -> None:
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    page_rect = page.rect
    x0, y0, x1, y1 = crop_fracs
    clip = fitz.Rect(
        page_rect.x0 + x0 * page_rect.width,
        page_rect.y0 + y0 * page_rect.height,
        page_rect.x0 + x1 * page_rect.width,
        page_rect.y0 + y1 * page_rect.height,
    )
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False)
    ensure_dir(out_path.parent)
    pix.save(out_path)


def stack_images(paths: list[Path], out_path: Path, *, bg: str = "white", pad: int = 20) -> None:
    imgs = [Image.open(p).convert("RGB") for p in paths]
    width = max(img.width for img in imgs)
    normalized = []
    for img in imgs:
        if img.width != width:
            new_height = int(round(img.height * width / img.width))
            img = img.resize((width, new_height), Image.Resampling.LANCZOS)
        normalized.append(ImageOps.expand(img, border=2, fill="white"))
    total_height = sum(img.height for img in normalized) + pad * (len(normalized) - 1)
    canvas = Image.new("RGB", (width, total_height), color=bg)
    y = 0
    for img in normalized:
        canvas.paste(img, (0, y))
        y += img.height + pad
    ensure_dir(out_path.parent)
    canvas.save(out_path)


def tile_images_horizontal(paths: list[Path], out_path: Path, *, bg: str = "white", pad: int = 24) -> None:
    imgs = [Image.open(p).convert("RGB") for p in paths]
    height = max(img.height for img in imgs)
    normalized = []
    for img in imgs:
        if img.height != height:
            new_width = int(round(img.width * height / img.height))
            img = img.resize((new_width, height), Image.Resampling.LANCZOS)
        normalized.append(ImageOps.expand(img, border=2, fill="white"))
    total_width = sum(img.width for img in normalized) + pad * (len(normalized) - 1)
    canvas = Image.new("RGB", (total_width, height), color=bg)
    x = 0
    for img in normalized:
        canvas.paste(img, (x, 0))
        x += img.width + pad
    ensure_dir(out_path.parent)
    canvas.save(out_path)


def _read_named_json(root_path: Path, object_path: str) -> dict:
    with uproot.open(root_path) as fin:
        obj = fin[object_path]
        try:
            payload = obj.member("fTitle")
        except Exception:
            payload = str(obj)
        return json.loads(payload)


def _parse_funcform_root_overrides(raw_overrides: list[str] | None) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for raw in raw_overrides or []:
        text = str(raw or "").strip()
        if "=" not in text:
            raise ValueError(
                f"Invalid --funcform-root-override value '{text}'. "
                "Expected DATASET=/absolute/or/relative/path.root"
            )
        dataset, path_text = text.split("=", 1)
        dataset = str(dataset).strip()
        path = Path(path_text).expanduser()
        if not dataset or not str(path).strip():
            raise ValueError(
                f"Invalid --funcform-root-override value '{text}'. "
                "Expected DATASET=/absolute/or/relative/path.root"
            )
        out[dataset] = path.resolve()
    return out


def _resolve_funcform_root_path(
    dataset: str,
    *,
    root_dir: Path | None = None,
    root_overrides: dict[str, Path] | None = None,
) -> Path:
    override = None
    if root_overrides and dataset in root_overrides:
        override = str(root_overrides[dataset])
    resolved = resolve_funcform_toy_root_path(
        dataset,
        configured_root=override,
        root_dir=str(root_dir) if root_dir is not None else None,
    )
    return Path(resolved)


def _funcform_root_specs(
    *,
    root_dir: Path | None = None,
    root_overrides: dict[str, Path] | None = None,
) -> list[tuple[str, Path]]:
    return [
        (
            dataset,
            _resolve_funcform_root_path(
                dataset,
                root_dir=root_dir,
                root_overrides=root_overrides,
            ),
        )
        for dataset in FUNCFORM_DATASETS
    ]


def _funcform_scan_range_mev(dataset: str, scan_range_gev: tuple[float, float] | list[float]) -> tuple[float, float]:
    if dataset in FUNCFORM_SCAN_RANGE_OVERRIDES_MEV:
        return FUNCFORM_SCAN_RANGE_OVERRIDES_MEV[dataset]
    return float(scan_range_gev[0]) * 1.0e3, float(scan_range_gev[1]) * 1.0e3


def _funcform_step_xy(values: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.r_[edges[:-1], edges[-1]] * 1.0e3, np.r_[values, values[-1]]


def _funcform_positive_bounds(*arrays: np.ndarray) -> tuple[float, float]:
    positives = [np.asarray(arr, dtype=float)[np.asarray(arr, dtype=float) > 0.0] for arr in arrays]
    positives = [arr for arr in positives if arr.size]
    if not positives:
        return 0.08, 1.0
    merged = np.concatenate(positives)
    ymin = max(float(np.min(merged)) * 0.7, 0.08)
    ymax = max(float(np.max(merged)) * 1.5, ymin * 5.0)
    return ymin, ymax


def _tf1_param_payload(fit_obj) -> tuple[list[str], np.ndarray]:
    params = fit_obj.member("fParams")
    names = [str(name) for name in params.member("fParNames")]
    values = np.asarray([float(val) for val in params.member("fParameters")], dtype=float)
    return names, values


def _safe_exp(values: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(np.asarray(values, dtype=float), -700.0, 700.0))


def _sigmoid(x: np.ndarray, xt: float, w: float) -> np.ndarray:
    if float(w) <= 0.0:
        return np.zeros_like(np.asarray(x, dtype=float))
    return 1.0 / (1.0 + _safe_exp(-(np.asarray(x, dtype=float) - float(xt)) / float(w)))


def _evaluate_funcform_counts(tag: str, values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    x = 0.5 * (np.asarray(edges[:-1], dtype=float) + np.asarray(edges[1:], dtype=float))
    p = np.asarray(values, dtype=float)
    out = np.zeros_like(x, dtype=float)

    if tag == "fSigPowExpQ":
        A, a, theta, xt, w, c1, c2 = p
        positive = x > 0.0
        out[positive] = (
            float(A)
            * _sigmoid(x[positive], xt, w)
            * np.power(x[positive], float(a))
            * _safe_exp(-x[positive] / float(theta))
            * _safe_exp(float(c1) * x[positive] + float(c2) * np.square(x[positive]))
        )
        return out

    if tag == "fShiftSigPow":
        A, a, theta, x0, xt, w = p
        z = x - float(x0)
        valid = (x > 0.0) & (z > 0.0) & (float(theta) > 0.0) & (float(w) > 0.0)
        out[valid] = (
            float(A)
            * _sigmoid(x[valid], xt, w)
            * np.power(np.maximum(z[valid], 1.0e-9), float(a))
            * _safe_exp(-np.maximum(z[valid], 1.0e-9) / float(theta))
        )
        return out

    if tag == "fShiftSigPowTail":
        A, a, theta, x0, xt, w, c1, c2, xmid, xscale = p
        z = x - float(x0)
        valid = (
            (x > 0.0)
            & (z > 0.0)
            & (float(theta) > 0.0)
            & (float(w) > 0.0)
            & (float(xscale) > 0.0)
        )
        u = (x[valid] - float(xmid)) / float(xscale)
        safe_z = np.maximum(z[valid], 1.0e-9)
        out[valid] = (
            float(A)
            * _sigmoid(x[valid], xt, w)
            * np.power(safe_z, float(a))
            * _safe_exp(-safe_z / float(theta))
            * _safe_exp(float(c1) * u + float(c2) * np.square(u))
        )
        return out

    if tag == "fGenGammaShift":
        A, a, lam, power, x0 = p
        z = x - float(x0)
        valid = (z > 0.0) & (float(lam) > 0.0) & (float(power) > 0.0)
        safe_z = np.maximum(z[valid], 1.0e-9)
        out[valid] = (
            float(A)
            * np.power(safe_z, float(a))
            * _safe_exp(-np.power(safe_z / float(lam), float(power)))
        )
        return out

    if tag == "fGenGammaThresh":
        A, a, lam, power, x0, xt, w = p
        z = x - float(x0)
        valid = (z > 0.0) & (float(lam) > 0.0) & (float(power) > 0.0) & (float(w) > 0.0)
        safe_z = np.maximum(z[valid], 1.0e-9)
        out[valid] = (
            float(A)
            * _sigmoid(x[valid], xt, w)
            * np.power(safe_z, float(a))
            * _safe_exp(-np.power(safe_z / float(lam), float(power)))
        )
        return out

    raise KeyError(f"Unsupported functional-form tag for figure fallback: {tag}")


def _load_funcform_payload(dataset: str, root_path: Path) -> dict:
    meta = _read_named_json(root_path, "fit_metadata/fit_summary_json")
    primary_tag = meta["primary_function"]
    fits_by_tag = {fit["tag"]: fit for fit in meta["fits"]}

    with uproot.open(root_path) as fin:
        data_vals, edges = fin["input_hist"].to_numpy()
        fit_curves = {}
        for fit in meta["fits"]:
            tag = str(fit["tag"])
            validation_key = f"validation/{tag}_expected_counts"
            if validation_key in fin:
                fit_curves[tag], _ = fin[validation_key].to_numpy()
            else:
                fit_obj = fin[f"fit_functions/{tag}_fit"]
                _, par_values = _tf1_param_payload(fit_obj)
                fit_curves[tag] = _evaluate_funcform_counts(tag, par_values, edges)

        toy_mean_key = f"validation/{primary_tag}_toy_mean"
        if toy_mean_key in fin:
            toy_vals, _ = fin[toy_mean_key].to_numpy()
        else:
            toy_dir = fin[str(primary_tag)]
            toy_arrays = []
            for toy_name in toy_dir.keys(cycle=False):
                vals, _ = toy_dir[str(toy_name)].to_numpy()
                toy_arrays.append(np.asarray(vals, dtype=float))
            if not toy_arrays:
                raise RuntimeError(f"No toy histograms found under {primary_tag} in {root_path}")
            toy_vals = np.mean(np.stack(toy_arrays, axis=0), axis=0)

    support_range = meta.get("toy_support_range_GeV") or meta.get("input_histogram_range_GeV")
    if not support_range:
        support_range = [float(edges[0]), float(edges[-1])]
    scan_range = meta.get("scan_range_GeV") or [
        float(meta.get("fit_min_GeV", float(edges[0]))),
        float(meta.get("fit_max_GeV", float(edges[-1]))),
    ]
    support_lo, support_hi = support_range
    occupied = np.flatnonzero(data_vals > 0.0)
    display_hi = float(edges[occupied[-1] + 1]) if occupied.size else float(support_hi)
    return {
        "dataset": dataset,
        "title": FUNCFORM_TITLES.get(dataset, dataset),
        "meta": meta,
        "primary_tag": primary_tag,
        "primary_fit": fits_by_tag[primary_tag],
        "data_vals": np.asarray(data_vals, dtype=float),
        "edges": np.asarray(edges, dtype=float),
        "fit_curves": fit_curves,
        "toy_vals": np.asarray(toy_vals, dtype=float),
        "support_range_mev": (float(support_lo) * 1.0e3, float(support_hi) * 1.0e3),
        "display_xlim_mev": (float(support_lo) * 1.0e3, display_hi * 1.0e3),
        "scan_range_mev": _funcform_scan_range_mev(dataset, scan_range),
    }


def _draw_funcform_support_guides(ax: plt.Axes, payload: dict) -> None:
    x_lo, x_hi = payload["display_xlim_mev"]
    support_lo, support_hi = payload["support_range_mev"]
    scan_lo, scan_hi = payload["scan_range_mev"]
    if support_lo < scan_lo:
        ax.axvspan(support_lo, scan_lo, color=FUNCFORM_COLORS["sideband"], alpha=0.55, zorder=0)
    hi_sideband_hi = min(max(x_hi, scan_hi), support_hi)
    if scan_hi < hi_sideband_hi:
        ax.axvspan(scan_hi, hi_sideband_hi, color=FUNCFORM_COLORS["sideband"], alpha=0.55, zorder=0)
    ax.axvline(scan_lo, color="0.45", lw=1.0, ls=":")
    ax.axvline(scan_hi, color="0.45", lw=1.0, ls=":")


def _draw_funcform_candidate_panel(ax: plt.Axes, payload: dict, *, show_ylabel: bool = False) -> None:
    x_plot, y_data = _funcform_step_xy(payload["data_vals"], payload["edges"])
    ax.step(x_plot, y_data, where="post", color=FUNCFORM_COLORS["data"], lw=1.35, label="Input histogram")
    for color, fit in zip(FUNCFORM_CANDIDATE_COLORS, payload["meta"]["fits"]):
        fit_vals = payload["fit_curves"][fit["tag"]]
        _, y_fit = _funcform_step_xy(fit_vals, payload["edges"])
        label = f"{fit['label']} (chi2/ndof={fit['pearson_chi2ndf']:.2f})"
        lw = 1.9 if fit["tag"] == payload["primary_tag"] else 1.3
        ax.step(x_plot, y_fit, where="post", color=color, lw=lw, alpha=0.95, label=label)

    _draw_funcform_support_guides(ax, payload)
    ymin, ymax = _funcform_positive_bounds(payload["data_vals"], *payload["fit_curves"].values())
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(*payload["display_xlim_mev"])
    ax.grid(alpha=0.22, which="both")
    ax.set_xlabel(r"$m_{e^+e^-}$ [MeV]")
    if show_ylabel:
        ax.set_ylabel("Counts / bin")
    ax.set_title(f"{payload['title']} candidate-family overlay", fontsize=10.8)
    ax.legend(loc="upper right", fontsize=7.2, framealpha=0.96)


def _draw_funcform_primary_panel(
    ax: plt.Axes,
    payload: dict,
    *,
    show_ylabel: bool = False,
    show_legend: bool = False,
) -> None:
    x_plot, y_data = _funcform_step_xy(payload["data_vals"], payload["edges"])
    fit_vals = payload["fit_curves"][payload["primary_tag"]]
    _, y_fit = _funcform_step_xy(fit_vals, payload["edges"])
    _, y_toy = _funcform_step_xy(payload["toy_vals"], payload["edges"])

    _draw_funcform_support_guides(ax, payload)
    ax.step(x_plot, y_data, where="post", color=FUNCFORM_COLORS["data"], lw=1.5, label="Observed data")
    ax.step(x_plot, y_fit, where="post", color=FUNCFORM_COLORS["fit"], lw=1.75, label="Selected fit")
    ax.step(
        x_plot,
        y_toy,
        where="post",
        color=FUNCFORM_COLORS["toy"],
        lw=1.55,
        ls="--",
        label="Toy-ensemble mean",
    )

    ymin, ymax = _funcform_positive_bounds(payload["data_vals"], fit_vals, payload["toy_vals"])
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(*payload["display_xlim_mev"])
    ax.grid(alpha=0.22, which="both")
    ax.set_xlabel(r"$m_{e^+e^-}$ [MeV]")
    if show_ylabel:
        ax.set_ylabel("Counts / bin")

    scan_lo, scan_hi = payload["scan_range_mev"]
    ax.set_title(
        f"{payload['title']} / {payload['primary_fit']['label']}\n"
        f"Scan [{scan_lo:.0f}, {scan_hi:.0f}] MeV / N events = {float(payload['meta'].get('normalization_target_count', np.sum(payload['data_vals']))):.3e}",
        fontsize=9.0,
    )
    if show_legend:
        ax.legend(loc="upper left", fontsize=8.0, framealpha=0.96)


def make_funcform_primary_fit_summary(
    out_path: Path,
    *,
    root_dir: Path | None = None,
    root_overrides: dict[str, Path] | None = None,
) -> None:
    payloads = [
        _load_funcform_payload(dataset, root_path)
        for dataset, root_path in _funcform_root_specs(root_dir=root_dir, root_overrides=root_overrides)
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.8), constrained_layout=True)

    for iax, payload in enumerate(payloads):
        _draw_funcform_primary_panel(
            axes[iax],
            payload,
            show_ylabel=(iax == 0),
            show_legend=(iax == 0),
        )

    fig.text(
        0.5,
        0.01,
        "Shaded sidebands contribute to the toy normalization but lie outside the analysis scan range.",
        ha="center",
        fontsize=9.0,
    )
    save_figure(fig, out_path)
    plt.close(fig)


def make_funcform_publication_overview(
    out_path: Path,
    *,
    root_dir: Path | None = None,
    root_overrides: dict[str, Path] | None = None,
) -> None:
    payloads = [
        _load_funcform_payload(dataset, root_path)
        for dataset, root_path in _funcform_root_specs(root_dir=root_dir, root_overrides=root_overrides)
    ]
    fig = plt.figure(figsize=(13.6, 12.2))
    grid = fig.add_gridspec(3, 2, width_ratios=[1.08, 1.0], hspace=0.42, wspace=0.18)
    right_handles = None
    right_labels = None

    for row, payload in enumerate(payloads):
        ax_left = fig.add_subplot(grid[row, 0])
        ax_right = fig.add_subplot(grid[row, 1])
        _draw_funcform_candidate_panel(ax_left, payload, show_ylabel=True)
        _draw_funcform_primary_panel(
            ax_right,
            payload,
            show_ylabel=False,
            show_legend=(row == 0),
        )
        if row == 0:
            right_handles, right_labels = ax_right.get_legend_handles_labels()

    if right_handles and right_labels:
        fig.legend(
            right_handles,
            right_labels,
            loc="upper center",
            bbox_to_anchor=(0.73, 0.995),
            ncol=3,
            frameon=False,
            fontsize=9.0,
        )
    fig.text(
        0.5,
        0.02,
        "Shaded sidebands contribute to the toy normalization but lie outside the analysis scan range.",
        ha="center",
        fontsize=9.2,
    )
    save_figure(fig, out_path)
    plt.close(fig)


def _format_param_value(value: float) -> str:
    aval = abs(float(value))
    if aval >= 1.0e4 or (aval > 0.0 and aval < 1.0e-3):
        return f"{float(value):.3e}"
    if aval >= 100.0:
        return f"{float(value):.2f}"
    if aval >= 1.0:
        return f"{float(value):.4f}"
    return f"{float(value):.5f}"


def _load_funcform_parameter_payload(dataset: str, root_path: Path) -> dict:
    meta = _read_named_json(root_path, "fit_metadata/fit_summary_json")
    primary_tag = str(meta["primary_function"])
    primary_fit = next(fit for fit in meta["fits"] if str(fit["tag"]) == primary_tag)
    with uproot.open(root_path) as fin:
        fit_obj = fin[f"fit_functions/{primary_tag}_fit"]
        params = fit_obj.member("fParams")
        par_names = [str(name) for name in params.member("fParNames")]
        par_values = [float(val) for val in params.member("fParameters")]
        par_mins = [float(val) for val in fit_obj.member("fParMin")]
        par_maxs = [float(val) for val in fit_obj.member("fParMax")]

    param_rows = []
    for name, value, lo, hi in zip(par_names, par_values, par_mins, par_maxs):
        is_fixed = abs(float(hi) - float(lo)) <= 1.0e-12
        label = f"{name} = {_format_param_value(value)}"
        if is_fixed:
            label += " (fixed)"
        param_rows.append(label)

    return {
        "dataset": dataset,
        "title": FUNCFORM_TITLES.get(dataset, dataset),
        "root_path": root_path,
        "meta": meta,
        "primary_tag": primary_tag,
        "primary_fit": primary_fit,
        "formula_lines": FUNCFORM_FORMULA_TEXT.get(primary_tag, (primary_fit["label"], "")),
        "param_rows": param_rows,
        "scan_range_mev": _funcform_scan_range_mev(
            dataset,
            meta.get("scan_range_GeV")
            or [float(primary_fit["fit_min_GeV"]), float(primary_fit["fit_max_GeV"])],
        ),
    }


def make_funcform_parameterization_summary(
    out_path: Path,
    *,
    root_dir: Path | None = None,
    root_overrides: dict[str, Path] | None = None,
) -> None:
    payloads = [
        _load_funcform_parameter_payload(dataset, root_path)
        for dataset, root_path in _funcform_root_specs(root_dir=root_dir, root_overrides=root_overrides)
    ]
    fig, axes = plt.subplots(3, 1, figsize=(13.6, 10.5), constrained_layout=True)

    for ax, payload in zip(axes, payloads):
        ax.set_axis_off()
        scan_lo, scan_hi = payload["scan_range_mev"]
        primary_fit = payload["primary_fit"]
        formula_lines = [line for line in payload["formula_lines"] if str(line).strip()]
        params = payload["param_rows"]
        split = max(1, int(np.ceil(len(params) / 2.0)))
        left_params = params[:split]
        right_params = params[split:]

        header = (
            f"{payload['title']}  |  primary seed: {payload['primary_tag']}  |  "
            f"trial: {primary_fit['trial_label']}  |  "
            f"fit [{float(primary_fit['fit_min_GeV']) * 1e3:.0f}, {float(primary_fit['fit_max_GeV']) * 1e3:.0f}] MeV  |  "
            f"scan [{scan_lo:.0f}, {scan_hi:.0f}] MeV"
        )
        stats_line = (
            f"Pearson chi2/ndof = {float(primary_fit['pearson_chi2ndf']):.3f}  |  "
            f"Neyman chi2/ndof = {float(primary_fit['neyman_chi2ndf']):.3f}  |  "
            f"ROOT chi2/ndof = {float(primary_fit['root_chi2ndf']):.3f}"
        )
        root_line = f"Source: {payload['root_path'].name}"

        ax.text(
            0.01,
            0.93,
            header,
            ha="left",
            va="top",
            fontsize=11.6,
            weight="bold",
        )
        ax.text(0.01, 0.80, stats_line, ha="left", va="top", fontsize=10.0, color="#333333")
        ax.text(0.01, 0.72, root_line, ha="left", va="top", fontsize=9.2, color="#555555")

        formula_text = "\n".join(formula_lines)
        ax.text(
            0.01,
            0.57,
            formula_text,
            ha="left",
            va="top",
            fontsize=11.0,
            bbox=dict(boxstyle="round,pad=0.35", fc="#F6F6F6", ec="#D0D0D0", alpha=0.98),
        )
        ax.text(
            0.03,
            0.34,
            "\n".join(left_params),
            ha="left",
            va="top",
            fontsize=10.2,
            family="monospace",
        )
        if right_params:
            ax.text(
                0.52,
                0.34,
                "\n".join(right_params),
                ha="left",
                va="top",
                fontsize=10.2,
                family="monospace",
            )
        ax.axhline(0.02, color="0.82", lw=1.0)

    save_figure(fig, out_path)
    plt.close(fig)


def make_pvalue_schematic(out_path: Path) -> None:
    rng = np.random.default_rng(1729)
    toys = rng.lognormal(mean=np.log(1.0), sigma=0.18, size=5000)
    obs = float(np.quantile(toys, 0.32))
    p_strong = float(np.mean(toys <= obs))
    p_weak = float(np.mean(toys >= obs))
    p_two = bounded_two_sided_tail_pvalue(p_strong, p_weak)

    bins = np.linspace(np.quantile(toys, 0.002), np.quantile(toys, 0.998), 36)
    counts, edges = np.histogram(toys, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    xs = np.sort(toys)
    ecdf = np.arange(1, xs.size + 1) / xs.size

    fig, axs = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)

    ax = axs[0]
    colors = np.where(centers <= obs, "#4C72B0", "#DD8452")
    ax.bar(centers, counts, width=widths, color=colors, alpha=0.78, edgecolor="white", linewidth=0.8)
    ax.axvline(obs, color="black", lw=1.6, ls="--")
    ax.text(obs, 1.02 * np.max(counts), r"$\epsilon^2_{95,\rm obs}$", ha="center", va="bottom", fontsize=10)
    ax.text(
        0.03,
        0.95,
        r"$p_{\rm strong}=P(\epsilon^2_{95,t}\leq \epsilon^2_{95,\rm obs})$" "\n"
        r"$p_{\rm weak}=P(\epsilon^2_{95,t}\geq \epsilon^2_{95,\rm obs})$",
        transform=ax.transAxes,
        va="top",
        fontsize=9.4,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.94),
    )
    ax.set_xlabel(r"Toy upper limit $\epsilon^2_{95,t}$ at fixed mass")
    ax.set_ylabel("Density")
    ax.set_title("Background-only toy-limit ensemble", fontsize=11)
    ax.grid(alpha=0.25)

    ax = axs[1]
    ax.step(xs, ecdf, where="post", color="0.15", lw=2.0, label="Empirical CDF")
    ax.axvline(obs, color="black", lw=1.6, ls="--")
    ax.axhline(p_strong, color="#4C72B0", lw=1.2, ls=":")
    ax.axhline(1.0 - p_weak, color="#DD8452", lw=1.2, ls=":")
    ax.fill_between(xs, 0.0, ecdf, where=xs <= obs, color="#4C72B0", alpha=0.18)
    ax.fill_between(xs, ecdf, 1.0, where=xs >= obs, color="#DD8452", alpha=0.18)
    ax.text(
        0.04,
        0.93,
        rf"$p_{{\rm strong}}={p_strong:.2f}$" "\n"
        rf"$p_{{\rm weak}}={p_weak:.2f}$" "\n"
        rf"$p_{{\rm two}}={p_two:.2f}$",
        transform=ax.transAxes,
        va="top",
        fontsize=9.6,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.94),
    )
    ax.set_xlabel(r"Observed-limit position in toy ensemble")
    ax.set_ylabel("Empirical CDF")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Equivalent cumulative representation", fontsize=11)
    ax.grid(alpha=0.25)

    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _pvalue_tail_components(toys: np.ndarray, obs: float) -> tuple[float, float, float]:
    p_strong = float(np.mean(toys <= obs))
    p_weak = float(np.mean(toys >= obs))
    p_two = bounded_two_sided_tail_pvalue(p_strong, p_weak)
    return p_strong, p_weak, p_two


def _draw_pvalue_example_panel(ax: plt.Axes, toys: np.ndarray, obs: float, title: str) -> None:
    p_strong, p_weak, p_two = _pvalue_tail_components(toys, obs)
    bins = np.linspace(np.quantile(toys, 0.002), np.quantile(toys, 0.998), 34)
    counts, edges = np.histogram(toys, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    ax.bar(centers, counts, width=widths, color="0.86", edgecolor="white", linewidth=0.8)
    ax.bar(
        centers[centers <= obs],
        counts[centers <= obs],
        width=widths[centers <= obs],
        color="#4C72B0",
        alpha=0.88,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.bar(
        centers[centers >= obs],
        counts[centers >= obs],
        width=widths[centers >= obs],
        color="#DD8452",
        alpha=0.72,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.axvline(obs, color="black", lw=1.6, ls="--")
    ax.set_title(title, fontsize=11.0)
    ax.set_xlabel(r"Toy upper limit $\epsilon^2_{95,t}$ at fixed mass")
    ax.grid(alpha=0.22)
    ax.text(
        0.04,
        0.94,
        rf"$p_{{\rm strong}}={p_strong:.2f}$" "\n"
        rf"$p_{{\rm weak}}={p_weak:.2f}$" "\n"
        rf"$p_{{\rm two}}=\min(1,2\min)= {p_two:.2f}$",
        transform=ax.transAxes,
        va="top",
        fontsize=9.1,
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="0.75", alpha=0.95),
    )


def make_pvalue_tail_examples(out_path: Path) -> None:
    rng = np.random.default_rng(31415)
    toys = rng.lognormal(mean=np.log(1.0), sigma=0.20, size=6000)
    observations = [
        ("Strong observed limit", float(np.quantile(toys, 0.14))),
        ("Weak observed limit", float(np.quantile(toys, 0.86))),
        ("Median-case symmetric diagnostic", float(np.quantile(toys, 0.50))),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 3.9), constrained_layout=True, sharey=True)
    for ax, (title, obs) in zip(axes, observations):
        _draw_pvalue_example_panel(ax, toys, obs, title)
    axes[0].set_ylabel("Density")
    save_figure(fig, out_path)
    plt.close(fig)


def make_constraints_panel(out_path: Path) -> None:
    entries = [
        ("BaBar prompt visible", 20.0, 10000.0, "#8172B3"),
        ("NA48/2 $\\pi^0 \\to \\gamma A'$", 10.0, 125.0, "#CCB974"),
        ("NA64 visible / X17-inspired", 1.0, 17.0, "#64B5CD"),
        ("HPS 2015 prompt", 19.0, 81.0, "#4C72B0"),
        ("HPS 2016 prompt", 39.0, 179.0, "#DD8452"),
        ("Current HPS GPR scope", 20.0, 250.0, "#55A868"),
    ]

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ypos = np.arange(len(entries))[::-1]
    for y, (label, xmin, xmax, color) in zip(ypos, entries):
        ax.hlines(y, xmin, xmax, color=color, lw=9.0, alpha=0.9)
        ax.scatter([xmin, xmax], [y, y], color=color, s=36, zorder=3)
        ax.text(xmax * 1.06, y, label, va="center", fontsize=10)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 2.0e4)
    ax.set_ylim(-0.8, len(entries) - 0.2)
    ax.set_xlabel(r"Approximate prompt-visible mass reach $m_{A'}$ [MeV]")
    ax.set_yticks([])
    ax.grid(alpha=0.25, axis="x", which="both")
    ax.set_title("Representative prompt-visible dark-photon constraints", fontsize=11.5)
    fig.text(
        0.035,
        0.03,
        "Comparison panel shows approximate mass reach, not exclusion depth. "
        "NA64 coverage is model-dependent in visible/X17-motivated interpretations.",
        fontsize=8.6,
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.91, bottom=0.18)
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _dark_photon_loop_kernel(masses_mev: np.ndarray, m_lepton_mev: float) -> np.ndarray:
    z = np.linspace(0.0, 1.0, 4000)
    r2 = (np.asarray(masses_mev, dtype=float) / float(m_lepton_mev)) ** 2
    integrand = 2.0 * z * (1.0 - z) ** 2 / ((1.0 - z) ** 2 + r2[:, None] * z[None, :])
    return np.trapezoid(integrand, z, axis=1)


def _scaled_eps2_band(
    masses_mev: np.ndarray,
    *,
    m_lepton_mev: float,
    eps_at_17_mev: float,
    eps_unc_at_17_mev: float,
    nsigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f_ref = _dark_photon_loop_kernel(np.array([17.0]), m_lepton_mev)[0]
    f_mass = _dark_photon_loop_kernel(masses_mev, m_lepton_mev)
    scale = f_ref / f_mass

    eps_lo = max(eps_at_17_mev - nsigma * eps_unc_at_17_mev, 1.0e-9)
    eps_hi = eps_at_17_mev + nsigma * eps_unc_at_17_mev
    eps_c = eps_at_17_mev
    return (eps_lo**2) * scale, (eps_c**2) * scale, (eps_hi**2) * scale


def make_prompt_visible_eps2_placeholder(out_path: Path) -> None:
    masses = np.linspace(10.0, 300.0, 500)

    # Values at 17 MeV taken from Peets, arXiv:2601.05288.
    mu_lo, mu_c, mu_hi = _scaled_eps2_band(
        masses,
        m_lepton_mev=105.6583755,
        eps_at_17_mev=7.03e-4,
        eps_unc_at_17_mev=0.58e-4,
        nsigma=2.0,
    )
    rb2_lo, rb2_c, rb2_hi = _scaled_eps2_band(
        masses,
        m_lepton_mev=0.51099895,
        eps_at_17_mev=0.69e-3,
        eps_unc_at_17_mev=0.15e-3,
        nsigma=2.0,
    )
    rb1_lo, rb1_c, rb1_hi = _scaled_eps2_band(
        masses,
        m_lepton_mev=0.51099895,
        eps_at_17_mev=0.69e-3,
        eps_unc_at_17_mev=0.15e-3,
        nsigma=1.0,
    )
    _, cs_c, _ = _scaled_eps2_band(
        masses,
        m_lepton_mev=0.51099895,
        eps_at_17_mev=1.19e-3,
        eps_unc_at_17_mev=0.15e-3,
        nsigma=2.0,
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.0))

    ax.axvspan(19.0, 81.0, color="#4C72B0", alpha=0.08)
    ax.axvspan(39.0, 179.0, color="#DD8452", alpha=0.07)
    ax.axvspan(20.0, 250.0, color="#55A868", alpha=0.04)
    ax.axvspan(16.2, 17.2, color="#C44E52", alpha=0.28)

    ax.fill_between(masses, mu_lo, mu_hi, color="#8172B3", alpha=0.33, label=r"$(g-2)_\mu$ 2$\sigma$ band (WP25)")
    ax.plot(masses, mu_c, color="#6A3D9A", lw=2.1)
    ax.fill_between(masses, rb2_lo, rb2_hi, color="#64B5CD", alpha=0.25, label=r"$(g-2)_e$ 2$\sigma$ band (Rb 2020)")
    ax.fill_between(masses, rb1_lo, rb1_hi, color="#1F77B4", alpha=0.35, label=r"$(g-2)_e$ 1$\sigma$ band (Rb 2020)")
    ax.plot(masses, rb1_c, color="#1F77B4", lw=2.0)
    ax.plot(masses, cs_c, color="#FF8C42", lw=1.8, ls="--", label=r"$(g-2)_e$ central (Cs 2018)")

    ax.text(18.0, 2.2e-6, "X17 mass", color="#8B1E3F", rotation=90, va="bottom", fontsize=9)
    ax.text(50.0, 2.0e-4, "HPS 2015", color="#4C72B0", fontsize=8.8, ha="center")
    ax.text(109.0, 1.3e-4, "HPS 2016", color="#DD8452", fontsize=8.8, ha="center")
    ax.text(188.0, 8.0e-5, "Current GPR scope", color="#2F6E3F", fontsize=8.8, ha="center")
    ax.text(
        0.02,
        0.03,
        "Placeholder panel: add exclusion contours from APEX, A1/MAMI, KLOE/KLOE-2,\n"
        "BaBar, NA48/2, and NA64 in the final review figure.",
        transform=ax.transAxes,
        fontsize=8.7,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="0.75", alpha=0.95),
    )

    ax.set_yscale("log")
    ax.set_xlim(10.0, 300.0)
    ax.set_ylim(1.0e-8, 4.0e-4)
    ax.set_xlabel(r"$m_{A'}$ [MeV]")
    ax.set_ylabel(r"Kinetic mixing $\epsilon^2$")
    ax.set_title("Low-mass prompt-visible dark-photon parameter space (placeholder)", fontsize=12.0)
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="upper left", fontsize=8.6, framealpha=0.95)

    ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_2021_parameterization_fig(out_path: Path) -> None:
    cfg_path = NOTE_DIR.parent / "config_2015_2016_10pct_2021_1pct_10k_rpen7.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    m = np.linspace(0.03, 0.25, 300)
    sigma_coeffs = cfg["sigma_coeffs_2021"]
    frad = float(cfg["frad_coeffs_2021"][0])
    penalty = float(cfg["radiative_penalty_frac_2021"])
    sigma = sum(float(c) * m**i for i, c in enumerate(sigma_coeffs))

    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.4), constrained_layout=True)

    axs[0].plot(m * 1e3, sigma * 1e3, color="#009E73", lw=2.2)
    axs[0].set_xlabel(r"$m_{A'}$ [MeV]")
    axs[0].set_ylabel(r"$\sigma_m$ [MeV]")
    axs[0].set_title("2021 mass-resolution parameterization", fontsize=11)
    axs[0].grid(alpha=0.25)

    axs[1].plot(m * 1e3, np.full_like(m, frad), color="#0072B2", lw=2.2, label=r"Baseline $f_{\rm rad}$")
    axs[1].plot(m * 1e3, np.full_like(m, frad * (1.0 - penalty)), color="#D55E00", lw=2.0, ls="--",
                label=rf"Penalty-adjusted $(1-\delta_f)f_{{\rm rad}}$, $\delta_f={penalty:.2f}$")
    axs[1].set_xlabel(r"$m_{A'}$ [MeV]")
    axs[1].set_ylabel(r"$f_{\rm rad}$")
    axs[1].set_title("2021 radiative-fraction inputs", fontsize=11)
    axs[1].grid(alpha=0.25)
    axs[1].legend(loc="upper right", fontsize=8.5)

    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_2021_resolution_only_fig(out_path: Path) -> None:
    cfg_path = NOTE_DIR.parent / "config_2015_2016_10pct_2021_1pct_10k_rpen7.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    m = np.linspace(0.03, 0.25, 300)
    sigma_coeffs = cfg["sigma_coeffs_2021"]
    sigma = sum(float(c) * m**i for i, c in enumerate(sigma_coeffs))

    fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
    ax.plot(m * 1e3, sigma * 1e3, color="#009E73", lw=2.2)
    ax.set_xlabel(r"$m_{A'}$ [MeV]")
    ax.set_ylabel(r"$\sigma_m$ [MeV]")
    ax.set_title("2021 mass-resolution parameterization", fontsize=11)
    ax.grid(alpha=0.25)
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_2021_radiative_inputs_fig(out_path: Path) -> None:
    cfg_path = NOTE_DIR.parent / "config_2015_2016_10pct_2021_1pct_10k_rpen7.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    m = np.linspace(0.03, 0.25, 300)
    frad = float(cfg["frad_coeffs_2021"][0])
    penalty = float(cfg["radiative_penalty_frac_2021"])

    fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
    ax.plot(m * 1e3, np.full_like(m, frad), color="#0072B2", lw=2.2, label=r"Baseline $f_{\rm rad}$")
    ax.plot(
        m * 1e3,
        np.full_like(m, frad * (1.0 - penalty)),
        color="#D55E00",
        lw=2.0,
        ls="--",
        label=rf"Penalty-adjusted $(1-\delta_f)f_{{\rm rad}}$, $\delta_f={penalty:.2f}$",
    )
    ax.set_xlabel(r"$m_{A'}$ [MeV]")
    ax.set_ylabel(r"$f_{\rm rad}$")
    ax.set_title("2021 radiative-fraction inputs", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8.5)
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_projection_placeholder(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    masses = np.array([20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240], float)
    baseline = 1.8e-9 * np.exp(-0.007 * (masses - 20))
    proj = baseline / np.sqrt(np.where(masses < 130, 10.0, 100.0))
    ax.plot(masses, baseline, color="#4C72B0", lw=2.0, label="Current staged baseline")
    ax.plot(masses, proj, color="#D55E00", lw=2.2, ls="--", label="Projected full-luminosity reach")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_{A'}$ [MeV]")
    ax.set_ylabel(r"Projected 95\% CL upper limit on $\epsilon^2$")
    ax.set_title("Projected Unblinded Reach in $\epsilon^2$", fontsize=11.5)
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="upper right", fontsize=8.8)
    ax.text(
        0.02,
        0.03,
        "Placeholder figure built from illustrative sqrt(L) rescaling.\n"
        "Final panel should be regenerated from combined UL-band CSV inputs.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.6,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.92),
    )
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate note-local figures.")
    parser.add_argument(
        "--funcform-only",
        action="store_true",
        help="Only regenerate the functional-form toy-fit comparison figure.",
    )
    parser.add_argument(
        "--funcform-root-dir",
        type=Path,
        help="Directory containing the functional-form ROOT exports to use for the note figures.",
    )
    parser.add_argument(
        "--funcform-root-override",
        action="append",
        metavar="DATASET=PATH",
        help="Explicit per-dataset functional-form ROOT override. May be repeated.",
    )
    args = parser.parse_args(argv)
    root_dir = args.funcform_root_dir.expanduser().resolve() if args.funcform_root_dir else None
    root_overrides = _parse_funcform_root_overrides(args.funcform_root_override)

    if args.funcform_only:
        make_funcform_primary_fit_summary(
            NOTE_DIR / "toy_generation_figs" / "funcform_primary_fit_summary.png",
            root_dir=root_dir,
            root_overrides=root_overrides,
        )
        make_funcform_publication_overview(
            NOTE_DIR / "toy_generation_figs" / "funcform_publication_overview.png",
            root_dir=root_dir,
            root_overrides=root_overrides,
        )
        make_funcform_parameterization_summary(
            NOTE_DIR / "toy_generation_figs" / "funcform_parameterization_summary.png",
            root_dir=root_dir,
            root_overrides=root_overrides,
        )
        return

    crop_pdf(
        NOTE_DIR / "HPS_2016_Bump_Hunt_Internal_Note_note.pdf",
        4,
        (0.03, 0.02, 0.97, 0.35),
        NOTE_DIR / "apparatus_figs" / "hps_2015_2016_svt_baseline.png",
    )
    crop_pdf(
        NOTE_DIR / "HPS_Experiment_2022.pdf",
        7,
        (0.50, 0.62, 0.96, 0.74),
        NOTE_DIR / "apparatus_figs" / "hps_2019_2021_svt_upgrade.png",
    )
    crop_pdf(
        NOTE_DIR / "HPS_2015_PRL_2018.pdf",
        4,
        (0.04, 0.05, 0.49, 0.27),
        NOTE_DIR / "published_reference_figs" / "hps2015_published_limit.png",
    )
    crop_pdf(
        NOTE_DIR / "2016_HPS_Paper.pdf",
        16,
        (0.05, 0.40, 0.45, 0.58),
        NOTE_DIR / "published_reference_figs" / "hps2016_published_prompt_pvalue.png",
    )
    crop_pdf(
        NOTE_DIR / "2016_HPS_Paper.pdf",
        16,
        (0.54, 0.05, 0.97, 0.16),
        NOTE_DIR / "published_reference_figs" / "hps2016_published_prompt_limit.png",
    )
    crop_pdf(
        NOTE_DIR / "hps_2015_resonance_search_internal_note.pdf",
        29,
        (0.09, 0.13, 0.89, 0.74),
        NOTE_DIR / "resolution_figs" / "hps2015_mass_resolution_internal_fig24.png",
    )
    crop_pdf(
        NOTE_DIR / "hps_2015_resonance_search_internal_note.pdf",
        39,
        (0.09, 0.05, 0.90, 0.46),
        NOTE_DIR / "normalization_figs" / "hps2015_radiative_fraction_internal_fig31.png",
    )
    crop_pdf(
        NOTE_DIR / "2015_radiative_radiative_fraction.pdf",
        0,
        (0.04, 0.07, 0.97, 0.92),
        NOTE_DIR / "normalization_figs" / "hps2015_radiative_cross_section_review.png",
    )
    crop_pdf(
        NOTE_DIR / "2015_radiative_radiative_fraction.pdf",
        1,
        (0.08, 0.07, 0.96, 0.92),
        NOTE_DIR / "normalization_figs" / "hps2015_radiative_fraction_review_right.png",
    )
    tile_images_horizontal(
        [
            NOTE_DIR / "normalization_figs" / "hps2015_radiative_cross_section_review.png",
            NOTE_DIR / "normalization_figs" / "hps2015_radiative_fraction_review_right.png",
        ],
        NOTE_DIR / "normalization_figs" / "hps2015_radiative_fraction_review.png",
    )
    crop_pdf(
        NOTE_DIR / "HPS_2016_Bump_Hunt_Internal_Note_note.pdf",
        37,
        (0.07, 0.04, 0.94, 0.31),
        NOTE_DIR / "resolution_figs" / "hps2016_mass_resolution_internal_fig29.png",
    )
    crop_pdf(
        NOTE_DIR / "HPS_2016_Bump_Hunt_Internal_Note_note.pdf",
        40,
        (0.07, 0.04, 0.93, 0.50),
        NOTE_DIR / "normalization_figs" / "hps2016_radiative_fraction_internal_fig30.png",
    )
    make_pvalue_schematic(NOTE_DIR / "methodology_figs" / "pvalue_tail_schematic.png")
    make_pvalue_tail_examples(NOTE_DIR / "methodology_figs" / "pvalue_tail_examples.png")
    make_prompt_visible_eps2_placeholder(NOTE_DIR / "context_figs" / "prompt_visible_constraints_panel.png")
    make_prompt_visible_eps2_placeholder(NOTE_DIR / "context_figs" / "prompt_visible_eps2_placeholder.png")
    make_2021_parameterization_fig(NOTE_DIR / "resolution_figs" / "hps2021_resolution_and_frad.png")
    make_2021_resolution_only_fig(NOTE_DIR / "resolution_figs" / "hps2021_mass_resolution_parameterization.png")
    make_2021_radiative_inputs_fig(NOTE_DIR / "normalization_figs" / "hps2021_radiative_fraction_inputs.png")
    stack_images(
        [
            NOTE_DIR / "significance_figs" / "2015" / "p0_analytic_local_global.png",
            NOTE_DIR / "significance_figs" / "2016_10pct" / "p0_analytic_local_global.png",
            NOTE_DIR / "summary_combined_all_rad_penalty" / "2021_p0_local_global.png",
        ],
        NOTE_DIR / "methodology_figs" / "local_significances_across_datasets.png",
    )
    make_projection_placeholder(
        NOTE_DIR / "combined_search_figs" / "projected_unblinded_reach_eps2_placeholder.png"
    )
    make_funcform_primary_fit_summary(
        NOTE_DIR / "toy_generation_figs" / "funcform_primary_fit_summary.png"
    )
    make_funcform_publication_overview(
        NOTE_DIR / "toy_generation_figs" / "funcform_publication_overview.png"
    )
    make_funcform_parameterization_summary(
        NOTE_DIR / "toy_generation_figs" / "funcform_parameterization_summary.png"
    )


if __name__ == "__main__":
    main()
