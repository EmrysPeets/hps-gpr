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


NOTE_DIR = Path(__file__).resolve().parents[1]


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
        return json.loads(obj.member("fTitle"))


def make_funcform_primary_fit_summary(out_path: Path) -> None:
    root_specs = [
        ("2015", NOTE_DIR.parent / "outputs" / "funcform_toys" / "funcform_2015_dataset_mod_toys.root"),
        ("2016", NOTE_DIR.parent / "outputs" / "funcform_toys" / "funcform_2016_dataset_mod_toys.root"),
        ("2021", NOTE_DIR.parent / "outputs" / "funcform_toys" / "funcform_2021_dataset_mod_toys.root"),
    ]

    colors = {
        "data": "#222222",
        "fit": "#C44E52",
        "toy": "#4C72B0",
        "sideband": "#D9D9D9",
    }
    titles = {
        "2015": "HPS 2015",
        "2016": "HPS 2016 10%",
        "2021": "HPS 2021 1%",
    }

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.2), constrained_layout=True)

    for iax, (dataset, root_path) in enumerate(root_specs):
        meta = _read_named_json(root_path, "fit_metadata/fit_summary_json")
        primary = meta["primary_function"]
        fit_meta = next(item for item in meta["fits"] if item["tag"] == primary)

        with uproot.open(root_path) as fin:
            data_vals, edges = fin["input_hist"].to_numpy()
            fit_vals, _ = fin[f"validation/{primary}_expected_counts"].to_numpy()
            toy_vals, _ = fin[f"validation/{primary}_toy_mean"].to_numpy()

        ax = axes[iax]
        x_lo = edges[:-1] * 1.0e3
        x_hi = edges[1:] * 1.0e3
        x_plot = np.r_[x_lo, x_hi[-1]]

        support_lo, support_hi = meta["toy_support_range_GeV"]
        scan_lo, scan_hi = meta["scan_range_GeV"]
        occupied = np.flatnonzero(data_vals > 0.0)
        display_hi = edges[occupied[-1] + 1] if occupied.size else support_hi
        if support_lo < scan_lo:
            ax.axvspan(support_lo * 1.0e3, scan_lo * 1.0e3, color=colors["sideband"], alpha=0.55)
        if scan_hi < support_hi:
            ax.axvspan(scan_hi * 1.0e3, support_hi * 1.0e3, color=colors["sideband"], alpha=0.55)
        ax.axvline(scan_lo * 1.0e3, color="0.45", lw=1.0, ls=":")
        ax.axvline(scan_hi * 1.0e3, color="0.45", lw=1.0, ls=":")

        ax.step(x_plot, np.r_[data_vals, data_vals[-1]], where="post", color=colors["data"], lw=1.55,
                label="Observed data")
        ax.step(x_plot, np.r_[fit_vals, fit_vals[-1]], where="post", color=colors["fit"], lw=1.75,
                label="Selected fit")
        ax.step(x_plot, np.r_[toy_vals, toy_vals[-1]], where="post", color=colors["toy"], lw=1.55, ls="--",
                label="Toy-ensemble mean")

        positive = np.concatenate([data_vals[data_vals > 0], fit_vals[fit_vals > 0], toy_vals[toy_vals > 0]])
        ymin = max(np.min(positive) * 0.7 if positive.size else 0.5, 0.08)
        ymax = np.max([data_vals.max(), fit_vals.max(), toy_vals.max()]) * 1.5
        ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(support_lo * 1.0e3, display_hi * 1.0e3)
        ax.grid(alpha=0.22, which="both")
        ax.set_xlabel(r"$m_{e^+e^-}$ [MeV]")
        if iax == 0:
            ax.set_ylabel("Counts / bin")
        ax.set_title(titles.get(dataset, dataset), fontsize=11.2)
        ax.text(
            0.03,
            0.97,
            f"Primary: {fit_meta['label']}\n"
            f"scan: {scan_lo * 1.0e3:.0f}-{scan_hi * 1.0e3:.0f} MeV\n"
            f"target N: {meta['normalization_target_count']:.3e}",
            transform=ax.transAxes,
            va="top",
            fontsize=8.4,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="0.75", alpha=0.94),
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=3, frameon=False)
    fig.text(
        0.5,
        -0.01,
        "Shaded sidebands denote the visible support-only regions outside the scan range.",
        ha="center",
        fontsize=9.0,
    )
    save_figure(fig, out_path)
    plt.close(fig)


def make_pvalue_schematic(out_path: Path) -> None:
    rng = np.random.default_rng(1729)
    toys = rng.lognormal(mean=np.log(1.0), sigma=0.18, size=5000)
    obs = float(np.quantile(toys, 0.32))
    p_strong = float(np.mean(toys <= obs))
    p_weak = float(np.mean(toys >= obs))
    p_two = float(2.0 * min(p_strong, p_weak))

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
    args = parser.parse_args(argv)

    if args.funcform_only:
        make_funcform_primary_fit_summary(
            NOTE_DIR / "toy_generation_figs" / "funcform_primary_fit_summary.png"
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


if __name__ == "__main__":
    main()
