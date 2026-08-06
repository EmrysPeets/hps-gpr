#!/usr/bin/env python3
"""Generate reviewer-facing GPR hyperparameter explanation figures.

The figures are deterministic schematics.  They do not use observed HPS data and
must not be interpreted as fit results.  Their purpose is to separate the roles of
the ConstantKernel amplitude C, the RBF length scale ell, and the fixed diagonal
observation variance alpha, and to show what per-mass-hypothesis optimization means.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF


NOTE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = NOTE_DIR / "methodology_figs"

INK = "#263442"
MUTED = "#687784"
GRID = "#DCE4EA"
BLUE = "#356FA8"
ORANGE = "#D68A22"
GREEN = "#27845A"
RED = "#B74848"
PURPLE = "#7657A5"
PALE_ORANGE = "#F8E7C7"
PALE_BLUE = "#E8F0F7"
PALE_GREEN = "#E3F1E9"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.0,
            "axes.edgecolor": "#A9B5BF",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.7,
            "legend.frameon": False,
            "grid.color": GRID,
            "grid.linewidth": 0.65,
            "grid.alpha": 0.85,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
    )


def save_pair(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def fixed_gp_prediction(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    *,
    constant: float,
    length_scale: float,
    alpha: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    kernel = ConstantKernel(constant, constant_value_bounds="fixed") * RBF(
        length_scale, length_scale_bounds="fixed"
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=np.asarray(alpha, dtype=float),
        optimizer=None,
        normalize_y=False,
    )
    gp.fit(np.asarray(x_train)[:, None], np.asarray(y_train))
    mean, std = gp.predict(np.asarray(x_query)[:, None], return_std=True)
    return np.asarray(mean), np.asarray(std)


def make_hyperparameter_roles(output_dir: Path) -> None:
    """Show that ell, C, and alpha act on different parts of the model."""

    fig, axes = plt.subplots(1, 3, figsize=(7.35, 3.35))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.16, top=0.68, wspace=0.34)

    # Panel (a): ell changes normalized correlation reach.
    ax = axes[0]
    separation = np.linspace(0.0, 12.0, 500)
    ell_values = [1.5, 4.0, 9.0]
    colors = [BLUE, GREEN, RED]
    for ell, color in zip(ell_values, colors):
        corr = np.exp(-0.5 * (separation / ell) ** 2)
        ax.plot(
            separation,
            corr,
            color=color,
            lw=2.0,
            label=rf"$\ell/\sigma_x={ell:g}$",
        )
        ax.scatter([ell], [np.exp(-0.5)], s=18, color=color, zorder=4)
    ax.axhline(np.exp(-0.5), color=MUTED, ls=":", lw=1.0)
    ax.text(
        11.7,
        np.exp(-0.5) + 0.025,
        r"$e^{-1/2}=0.61$",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=MUTED,
    )
    ax.axvspan(0.0, 2.25, color=PALE_ORANGE, alpha=0.65, zorder=-3)
    ax.text(
        1.12,
        0.06,
        r"signal-scale" "\n" r"reference",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="#8B621F",
    )
    ax.set(
        xlim=(0.0, 12.0),
        ylim=(0.0, 1.03),
        xlabel=r"separation $|\Delta x|/\sigma_x(m_0)$",
        ylabel=r"correlation $k(\Delta x)/k(0)$",
        title=r"$\ell$: correlation reach",
    )
    ax.grid(True)
    ax.legend(loc="upper right", handlelength=1.7)
    ax.text(
        0.50,
        1.20,
        r"optimized: bounded $\ell$ domain",
        transform=ax.transAxes,
        fontsize=7.3,
        color=GREEN,
        fontweight="bold",
        ha="center",
    )
    panel_label(ax, "(a)")

    # Panel (b): C changes covariance amplitude, not correlation reach.
    ax = axes[1]
    fixed_ell = 4.0
    constants = [0.25, 1.0, 4.0]
    for constant, color in zip(constants, colors):
        covariance = constant * np.exp(-0.5 * (separation / fixed_ell) ** 2)
        ax.plot(
            separation,
            covariance,
            color=color,
            lw=2.0,
            label=rf"$C={constant:g}$",
        )
        ax.scatter([0.0], [constant], s=18, color=color, zorder=4)
    ax.axvline(fixed_ell, color=MUTED, ls=":", lw=1.0)
    ax.text(
        fixed_ell + 0.15,
        3.8,
        r"same $\ell$",
        ha="left",
        va="top",
        fontsize=7.5,
        color=MUTED,
    )
    ax.set(
        xlim=(0.0, 12.0),
        ylim=(0.0, 4.2),
        xlabel=r"separation $|\Delta x|/\sigma_x(m_0)$",
        ylabel=r"covariance $k(\Delta x)$ [log-count$^2$]",
        title=r"$C$: covariance amplitude",
    )
    ax.grid(True)
    ax.legend(loc="upper right", handlelength=1.7)
    ax.text(
        0.50,
        1.20,
        r"optimized: broad $C$ domain",
        transform=ax.transAxes,
        fontsize=7.3,
        color=GREEN,
        fontweight="bold",
        ha="center",
    )
    panel_label(ax, "(b)")

    # Panel (c): alpha changes per-bin leverage on the covariance diagonal.
    ax = axes[2]
    x_train = np.linspace(-4.0, 4.0, 17)
    latent = 0.08 * x_train + 0.13 * np.sin(1.15 * x_train)
    deterministic_noise = 0.035 * np.sin(3.7 * x_train + 0.4)
    y_train = latent + deterministic_noise
    outlier_index = int(np.argmin(np.abs(x_train - 1.0)))
    y_train[outlier_index] += 0.34
    x_query = np.linspace(-4.2, 4.2, 500)

    alpha_base = np.full_like(x_train, 0.012)
    alpha_small = alpha_base.copy()
    alpha_small[outlier_index] = 0.002
    alpha_large = alpha_base.copy()
    alpha_large[outlier_index] = 0.22

    mean_small, _ = fixed_gp_prediction(
        x_train,
        y_train,
        x_query,
        constant=0.16,
        length_scale=1.4,
        alpha=alpha_small,
    )
    mean_large, _ = fixed_gp_prediction(
        x_train,
        y_train,
        x_query,
        constant=0.16,
        length_scale=1.4,
        alpha=alpha_large,
    )
    ax.plot(
        x_query,
        mean_small,
        color=RED,
        lw=2.0,
        label="_nolegend_",
    )
    ax.plot(
        x_query,
        mean_large,
        color=BLUE,
        lw=2.0,
        label="_nolegend_",
    )
    keep = np.arange(x_train.size) != outlier_index
    ax.scatter(
        x_train[keep],
        y_train[keep],
        s=18,
        facecolor="white",
        edgecolor=INK,
        lw=0.8,
        zorder=4,
        label="_nolegend_",
    )
    ax.scatter(
        [x_train[outlier_index]],
        [y_train[outlier_index]],
        s=38,
        marker="D",
        color=ORANGE,
        edgecolor="white",
        lw=0.6,
        zorder=5,
        label="_nolegend_",
    )
    ax.annotate(
        r"only diagonal entry $\alpha_j$ changes",
        xy=(x_train[outlier_index], y_train[outlier_index]),
        xytext=(-3.8, 0.42),
        arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 1.0},
        fontsize=7.4,
        color="#8B621F",
    )
    ax.set(
        xlim=(-4.2, 4.2),
        ylim=(-0.48, 0.58),
        xlabel=r"GP coordinate $x$",
        ylabel=r"latent log-rate",
        title=r"$\alpha_i$: per-bin trust",
    )
    ax.grid(True)
    ax.text(
        -3.85,
        -0.35,
        r"small $\alpha_j$: bin has leverage",
        color=RED,
        fontsize=7.2,
        ha="left",
        va="top",
    )
    ax.text(
        -3.85,
        -0.43,
        r"large $\alpha_j$: bin downweighted",
        color=BLUE,
        fontsize=7.2,
        ha="left",
        va="top",
    )
    ax.text(
        0.50,
        1.20,
        r"fixed: $\alpha_i=1/y_i$",
        transform=ax.transAxes,
        fontsize=7.3,
        color=GREEN,
        fontweight="bold",
        ha="center",
    )
    panel_label(ax, "(c)")

    fig.suptitle(
        r"$C\times{\rm RBF}$ plus observation noise: three different roles",
        y=0.975,
        fontsize=11.3,
        fontweight="bold",
        color=INK,
    )
    save_pair(fig, output_dir, "gpr_hyperparameter_roles")


def synthetic_sideband_problem() -> dict[str, np.ndarray]:
    u_all = np.linspace(-10.0, 10.0, 41)
    is_sideband = np.abs(u_all) > 2.25
    u_train = u_all[is_sideband]
    latent = (
        0.12 * np.sin((u_train + 1.0) / 4.0)
        + 0.055 * np.cos(u_train / 2.3)
        + 0.02 * u_train / 10.0
    )
    fluctuation = 0.018 * np.sin(3.1 * u_train) + 0.012 * np.cos(5.3 * u_train)
    z_train = latent + fluctuation
    alpha = 0.00055 + 0.0002 * (1.0 + u_train / 10.0) ** 2
    return {
        "u_all": u_all,
        "is_sideband": is_sideband,
        "u_train": u_train,
        "z_train": z_train,
        "alpha": alpha,
    }


def log_marginal_likelihood_grid(
    x: np.ndarray,
    y: np.ndarray,
    alpha: np.ndarray,
    constants: np.ndarray,
    length_scales: np.ndarray,
) -> np.ndarray:
    distance = x[:, None] - x[None, :]
    identity_noise = np.diag(alpha)
    output = np.full((constants.size, length_scales.size), np.nan, dtype=float)
    for i, constant in enumerate(constants):
        for j, length_scale in enumerate(length_scales):
            covariance = constant * np.exp(
                -0.5 * (distance / length_scale) ** 2
            )
            observed_covariance = covariance + identity_noise
            try:
                chol = np.linalg.cholesky(observed_covariance)
                weight = np.linalg.solve(
                    chol.T, np.linalg.solve(chol, y)
                )
                output[i, j] = (
                    -0.5 * y @ weight
                    - np.log(np.diag(chol)).sum()
                    - 0.5 * y.size * np.log(2.0 * np.pi)
                )
            except np.linalg.LinAlgError:
                continue
    return output


def make_mass_hypothesis_optimization(output_dir: Path) -> None:
    """Explain one bounded sideband optimization and its repetition in a scan."""

    problem = synthetic_sideband_problem()
    u_train = problem["u_train"]
    z_train = problem["z_train"]
    alpha = problem["alpha"]

    constants = np.geomspace(0.002, 0.2, 110)
    length_scales = np.geomspace(0.8, 12.0, 110)
    lml = log_marginal_likelihood_grid(
        u_train, z_train, alpha, constants, length_scales
    )
    optimum_index = np.unravel_index(np.nanargmax(lml), lml.shape)
    constant_opt = float(constants[optimum_index[0]])
    length_opt = float(length_scales[optimum_index[1]])

    # Conditional C optima for the short/interior/long examples in panel (a).
    candidate_length_scales = [1.1, length_opt, 9.0]
    candidate_constants: list[float] = []
    for length_scale in candidate_length_scales:
        j = int(np.argmin(np.abs(length_scales - length_scale)))
        i = int(np.nanargmax(lml[:, j]))
        candidate_constants.append(float(constants[i]))

    fig = plt.figure(figsize=(7.35, 6.65))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.075,
        right=0.985,
        bottom=0.08,
        top=0.89,
        wspace=0.30,
        hspace=0.58,
    )
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]

    # Panel (a): the same masked sidebands under candidate length scales.
    ax = axes[0]
    u_query = np.linspace(-10.0, 10.0, 700)
    ax.axvspan(-2.25, 2.25, color=PALE_ORANGE, alpha=0.82, zorder=-4)
    ax.axvline(0.0, color=ORANGE, lw=1.0, ls="--")
    ax.scatter(
        u_train,
        z_train,
        s=17,
        facecolor="white",
        edgecolor=INK,
        lw=0.8,
        zorder=5,
        label="sideband observations",
    )
    labels = [
        r"short $\ell$: local flexibility",
        rf"interior optimum $\ell/\sigma_x={length_opt:.1f}$",
        r"long $\ell$: broad smoothing",
    ]
    colors = [BLUE, GREEN, RED]
    for length_scale, constant, label, color in zip(
        candidate_length_scales, candidate_constants, labels, colors
    ):
        mean, std = fixed_gp_prediction(
            u_train,
            z_train,
            u_query,
            constant=constant,
            length_scale=length_scale,
            alpha=alpha,
        )
        ax.plot(u_query, mean, color=color, lw=1.9, label=label)
        if color == GREEN:
            ax.fill_between(
                u_query,
                mean - std,
                mean + std,
                color=GREEN,
                alpha=0.14,
                lw=0,
                zorder=-1,
            )
    ax.text(
        0.0,
        -0.20,
        "excluded from training",
        ha="center",
        va="bottom",
        fontsize=7.4,
        color="#8B621F",
    )
    ax.set(
        xlim=(-10.0, 10.0),
        ylim=(-0.22, 0.23),
        xlabel=r"local coordinate $u=(x-x_0)/\sigma_x(m_0)$",
        ylabel=r"schematic latent log-rate",
        title=r"Sidebands at one mass hypothesis",
    )
    ax.grid(True)
    ax.legend(loc="upper left", handlelength=1.6)
    panel_label(ax, "(a)")

    # Panel (b): joint evidence surface for C and ell.
    ax = axes[1]
    delta_lml = lml - np.nanmax(lml)
    cmap = LinearSegmentedColormap.from_list(
        "hps_blue", ["#F3F7FA", "#BFD3E4", "#648FB4", "#254D73"]
    )
    mesh = ax.pcolormesh(
        length_scales,
        constants,
        np.clip(delta_lml, -12.0, 0.0),
        shading="auto",
        cmap=cmap,
        vmin=-12.0,
        vmax=0.0,
    )
    contour = ax.contour(
        length_scales,
        constants,
        delta_lml,
        levels=[-8.0, -4.5, -2.0, -0.5],
        colors="white",
        linewidths=0.8,
    )
    ax.clabel(contour, fmt=lambda v: rf"$\Delta\log L={v:g}$", fontsize=6.2)
    ax.scatter(
        [length_opt],
        [constant_opt],
        marker="*",
        s=105,
        color=ORANGE,
        edgecolor="white",
        lw=0.7,
        zorder=5,
        label="_nolegend_",
    )
    ax.axvline(1.0, color=RED, lw=1.15, ls="--")
    ax.axvline(10.0, color=RED, lw=1.15, ls="--")
    ax.text(
        1.04,
        0.165,
        r"$\ell_{\min}$",
        color=RED,
        fontsize=7.4,
        ha="left",
    )
    ax.text(
        9.75,
        0.165,
        r"$\ell_{\max}$",
        color=RED,
        fontsize=7.4,
        ha="right",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(
        xlim=(0.8, 12.0),
        ylim=(0.002, 0.2),
        xlabel=r"RBF length scale $\ell/\sigma_x(m_0)$",
        ylabel=r"ConstantKernel value $C$",
        title=r"Joint optimization of $C$ and $\ell$",
    )
    ax.annotate(
        "bounded maximum",
        xy=(length_opt, constant_opt),
        xytext=(1.45, 0.0031),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 0.9},
        fontsize=7.2,
        color="#8B621F",
        ha="left",
    )
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.05)
    colorbar.set_label(r"$\Delta\log p(\mathbf{z}_{\rm sb}|C,\ell)$", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    panel_label(ax, "(b)")

    # Panel (c): the window and local resolution change across mass hypotheses.
    ax = axes[2]
    masses = np.arange(55.0, 116.0, 3.0)
    rows = [(72.0, 1.25, 1.0), (96.0, 1.55, 0.0)]
    row_colors = [BLUE, PURPLE]
    for (mass_hypothesis, sigma_m, row), color in zip(rows, row_colors):
        blind_half_width = 2.25 * sigma_m
        is_train = np.abs(masses - mass_hypothesis) > blind_half_width
        ax.scatter(
            masses[is_train],
            np.full(np.count_nonzero(is_train), row),
            s=19,
            color=INK,
            zorder=3,
        )
        ax.scatter(
            masses[~is_train],
            np.full(np.count_nonzero(~is_train), row),
            s=26,
            facecolor="white",
            edgecolor=color,
            lw=1.2,
            zorder=4,
        )
        ax.axvspan(
            mass_hypothesis - blind_half_width,
            mass_hypothesis + blind_half_width,
            ymin=0.08 + 0.45 * row,
            ymax=0.44 + 0.45 * row,
            color=color,
            alpha=0.16,
        )
        ax.axvline(
            mass_hypothesis,
            ymin=0.10 + 0.45 * row,
            ymax=0.42 + 0.45 * row,
            color=color,
            lw=1.2,
        )
        ax.text(
            mass_hypothesis,
            row + 0.15,
            rf"$m_0={mass_hypothesis:.0f}$ MeV, "
            rf"$\sigma_m={sigma_m:.2f}$ MeV",
            ha="center",
            va="bottom",
            color=color,
            fontsize=7.5,
        )
    ax.text(
        57.0,
        0.50,
        "filled: training bin",
        color=INK,
        fontsize=7.3,
        va="center",
    )
    ax.text(
        57.0,
        -0.50,
        "open: excluded bin",
        color=MUTED,
        fontsize=7.3,
        va="center",
    )
    ax.set(
        xlim=(54.0, 117.0),
        ylim=(-0.72, 1.38),
        xlabel=r"invariant mass $m$ [MeV]",
        title=r"Moving $m_0$ changes the mask and local scale",
    )
    ax.set_yticks([])
    ax.grid(True, axis="x")
    panel_label(ax, "(c)")

    # Panel (d): interior and boundary optima have different interpretations.
    ax = axes[3]
    scan_mass = np.arange(50.0, 181.0, 5.0)
    lower = 1.0 + 0.08 * np.cos(scan_mass / 35.0)
    upper = np.full_like(scan_mass, 12.0)
    unconstrained_like = (
        5.3
        + 1.0 * np.sin((scan_mass - 45.0) / 18.0)
        + 0.35 * np.cos(scan_mass / 8.5)
    )
    unconstrained_like[-6:] = [8.7, 9.8, 11.3, 12.8, 13.7, 14.1]
    fitted = np.minimum(unconstrained_like, upper)
    at_upper = unconstrained_like >= upper
    ax.fill_between(
        scan_mass,
        lower,
        upper,
        color=PALE_BLUE,
        alpha=0.9,
        label="_nolegend_",
    )
    ax.plot(scan_mass, lower, color=MUTED, lw=1.0, ls=":")
    ax.plot(scan_mass, upper, color=RED, lw=1.2, ls="--")
    ax.plot(scan_mass, fitted, color=GREEN, lw=1.6, alpha=0.8)
    ax.scatter(
        scan_mass[~at_upper],
        fitted[~at_upper],
        s=22,
        color=GREEN,
        edgecolor="white",
        lw=0.4,
        label="_nolegend_",
        zorder=4,
    )
    ax.scatter(
        scan_mass[at_upper],
        fitted[at_upper],
        s=42,
        marker="^",
        color=RED,
        edgecolor="white",
        lw=0.5,
        label="_nolegend_",
        zorder=5,
    )
    ax.annotate(
        "domain is active;\nrange study required",
        xy=(scan_mass[-3], fitted[-3]),
        xytext=(112.0, 9.0),
        arrowprops={"arrowstyle": "->", "color": RED, "lw": 1.0},
        color=RED,
        fontsize=7.3,
        ha="left",
    )
    ax.text(
        54.0,
        1.5,
        r"not a measurement of a physical correlation length",
        color=MUTED,
        fontsize=7.2,
    )
    ax.set(
        xlim=(48.0, 183.0),
        ylim=(0.0, 13.2),
        xlabel=r"scanned mass hypothesis $m_0$ [MeV]",
        ylabel=r"$\ell_{\rm opt}/\sigma_x(m_0)$",
        title=r"Interior optimum versus boundary contact",
    )
    ax.grid(True)
    ax.text(
        54.0,
        12.35,
        "configured domain",
        color=INK,
        fontsize=7.3,
        va="center",
    )
    ax.text(
        58.0,
        6.7,
        "interior conditional optimum",
        color=GREEN,
        fontsize=7.3,
        va="center",
    )
    ax.text(
        139.0,
        12.55,
        "upper-bound contact",
        color=RED,
        fontsize=7.3,
        va="bottom",
    )
    panel_label(ax, "(d)")

    fig.suptitle(
        "What an optimized length scale means at a mass hypothesis",
        y=0.975,
        fontsize=11.5,
        fontweight="bold",
        color=INK,
    )
    save_pair(fig, output_dir, "gpr_mass_hypothesis_optimization")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate explanatory C*RBF hyperparameter diagrams."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory (default: note methodology_figs).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    make_hyperparameter_roles(args.output_dir)
    make_mass_hypothesis_optimization(args.output_dir)


if __name__ == "__main__":
    main()
