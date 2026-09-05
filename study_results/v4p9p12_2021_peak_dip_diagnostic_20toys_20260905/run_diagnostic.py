#!/usr/bin/env python3
"""Fast, isolated 2021 sideband-response pilot; exactly 20 toy spectra."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PARENT = REPO / "study_results/v4p9p12_final_dataset_combinations_20260902"
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["MPLCONFIGDIR"] = str(HERE / ".mplcache")
sys.dont_write_bytecode = True
sys.path.insert(0, str(PARENT))

import run_final_combinations as production
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
from hps_gpr.gpr import (fit_gpr, make_fixed_kernel,
                        predict_counts_from_log_gpr,
                        predict_counts_mean_var_from_log_gpr)
from hps_gpr.statistics import (fit_A_profiled_gaussian_details,
                               profiled_gaussian_fixed_poi_nll)
from hps_gpr.template import build_window_template_from_full

MASSES = np.arange(60, 81)
N_PAIRS = 10
SEED = 49126672
OUTPUT = HERE / "derived"
FIGURES = HERE / "figures"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    OUTPUT.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    cfg = production.load_config(production.DEFAULT_CARD)
    production.validate_card(cfg)
    sources = {
        "card": production.DEFAULT_CARD,
        "states": production.DEFAULT_STATES,
        "spectrum": PARENT / "derived/all_three_peak_extraction_plot_data.csv",
        "curves": PARENT / "derived/final_dataset_result_curves.csv",
        "peak_table": PARENT / "derived/all_three_peak_extraction_table.csv",
        "script": Path(__file__), "protocol": HERE / "PROTOCOL.md",
    }
    provenance = {k: {"path": str(p), "sha256": sha(p)} for k, p in sources.items()}
    frame = pd.read_csv(sources["spectrum"])
    frame = frame[frame.dataset.astype(str) == "2021"].sort_values("bin_center_GeV")
    x = frame.bin_center_GeV.to_numpy(float)
    y = frame.observed_events.to_numpy(float)
    widths = frame.bin_width_MeV.to_numpy(float) / 1000
    edges = np.r_[x - widths / 2, x[-1] + widths[-1] / 2]
    assert np.array_equal(y, np.rint(y)) and np.all(y > 0)
    assert len(x) > 300 and 0.035 < edges[0] < 0.037 and 0.299 < edges[-1] < 0.301
    states = production.state_map(pd.read_csv(sources["states"]))
    curves = pd.read_csv(sources["curves"])
    curves = curves[(curves.scope_key == "individual_2021_10pct") &
                    curves.mass_MeV.isin(MASSES)].sort_values("mass_MeV")
    peak = pd.read_csv(sources["peak_table"])
    peak = peak[peak.dataset.astype(str) == "2021"].iloc[0]
    injection = frame.independent_signed_signal_events.to_numpy(float)
    assert np.all(injection >= 0)
    assert np.isclose(injection.sum(), peak.independent_signed_full_template_yield)
    sigma = lambda m: float(np.polynomial.polynomial.polyval(m, cfg.sigma_coeffs_2021))

    def gp_fit(counts, mass, common=False):
        state = states[("2021", 66 if common else int(mass))]
        m = mass / 1000
        lo, hi = (0.060, 0.078) if common else (m - 2.25*sigma(m), m + 2.25*sigma(m))
        keep = (x < lo) | (x > hi)
        return fit_gpr(x[keep], counts[keep], cfg, restarts=0,
                       kernel=make_fixed_kernel(state["const_opt"], state["ls_opt"]),
                       optimize=False)

    def evaluate(counts, mass, model=None):
        m = mass / 1000
        mask = (x >= m - 2.25*sigma(m)) & (x <= m + 2.25*sigma(m))
        if model is None:
            model = gp_fit(counts, mass)
        mean, rawcov = predict_counts_from_log_gpr(model, x[mask], cfg)
        cov, conditioning = production.condition_covariance_block(rawcov, mean)
        template, _ = build_window_template_from_full(edges, mask, m, sigma(m), config=cfg)
        template = np.asarray(template, float)
        template /= template.sum()
        fit = fit_A_profiled_gaussian_details(counts[mask], mean, cov, template,
                                              allow_negative=True)
        null = profiled_gaussian_fixed_poi_nll(counts[mask], mean, cov, template, A_fixed=0)
        if not fit["success"] or not null["success"]:
            raise RuntimeError(f"Fit failed at {mass}: {fit.get('success')}, {null.get('success')}")
        q = 2*(float(null["nll"]) - float(fit["nll"]))
        if not np.isfinite(q):
            raise RuntimeError(f"Nonfinite likelihood difference at {mass}: {q}")
        # The null is a known feasible point in the signed alternative.
        # Match production's feasible-candidate safeguard, without retuning
        # the optimizer or hiding the raw discrepancy in the output ledger.
        raw_q = q
        null_fallback = q < 0
        if null_fallback:
            fit = dict(fit, A_hat=0.0, nll=float(null["nll"]))
            q = 0.0
        r = float(np.sign(fit["A_hat"]) * np.sqrt(max(q, 0)))
        return {"mass_MeV": int(mass), "signed_r": r,
                "A_hat_window": float(fit["A_hat"]),
                "sigma_A_window": float(fit["sigma_A"]),
                "raw_twice_delta_nll": raw_q,
                "null_feasible_fallback": bool(null_fallback),
                "diagonal_load_relative": conditioning["selected_diagonal_load_relative"]}

    print("Reconstructing the 21 saved observed fits before generating toys...", flush=True)
    observed = []
    for row in curves.to_dict("records"):
        result = evaluate(y, int(row["mass_MeV"]))
        base = json.loads(row["limit_profile_status"])["observed"]["base"]
        fit = base["fit_unbounded"]
        expected = np.sign(fit["A_hat"])*np.sqrt(max(2*(base["null"]["nll"]-fit["nll"]), 0))
        result["saved_signed_r"] = float(expected)
        result["reconstruction_delta_r"] = result["signed_r"] - expected
        if abs(result["reconstruction_delta_r"]) > 0.02:
            raise RuntimeError(f"Observed reconstruction mismatch: {result}")
        observed.append(result)
    pd.DataFrame(observed).to_csv(OUTPUT / "observed_scan.csv", index=False)
    print(f"Observed closure max |delta r| = {max(abs(r['reconstruction_delta_r']) for r in observed):.6g}", flush=True)

    common_gp = gp_fit(y, 66, common=True)
    truth, truth_var = predict_counts_mean_var_from_log_gpr(common_gp, x, cfg)
    common_results = [evaluate(y, m, common_gp) for m in (66, 71, 72)]
    pd.DataFrame(common_results).to_csv(OUTPUT / "common_mask_observed.csv", index=False)
    masks = {}
    plot_frame = pd.DataFrame({"mass_MeV": x*1000, "observed_counts": y,
                              "smooth_truth_counts": truth, "truth_gp_variance": truth_var,
                              "injected_signal_counts": injection})
    for m in (66, 71, 72):
        gm, _ = predict_counts_mean_var_from_log_gpr(gp_fit(y, m), x, cfg)
        plot_frame[f"background_m{m}"] = gm
        mask = (x < m/1000-2.25*sigma(m/1000)) | (x > m/1000+2.25*sigma(m/1000))
        masks[str(m)] = float(injection[mask].sum()/injection.sum())
    plot_frame.to_csv(OUTPUT / "spectrum_and_backgrounds.csv", index=False)

    toy_path = OUTPUT / "twenty_toy_spectra.npz"
    if toy_path.exists():
        with np.load(toy_path) as saved:
            assert np.array_equal(saved["x_GeV"], x)
            background_spectra = saved["background_only"].copy()
            injected_spectra = saved["signal_plus_background"].copy()
        print("Reusing the existing twenty toy spectra; no new random draws.", flush=True)
    else:
        rng = np.random.default_rng(SEED)
        background_spectra = rng.poisson(truth, size=(N_PAIRS, len(truth)))
        injected_spectra = background_spectra + rng.poisson(injection, size=(N_PAIRS, len(truth)))
        np.savez_compressed(toy_path, x_GeV=x, background_only=background_spectra,
                            signal_plus_background=injected_spectra)
    assert len(background_spectra) + len(injected_spectra) == 20
    records = []
    means = []
    for lane, counts in (("background", truth), ("injected", truth+injection)):
        means.extend(dict(evaluate(counts, int(m)), lane=lane) for m in MASSES)
    pd.DataFrame(means).to_csv(OUTPUT / "deterministic_mean_scans.csv", index=False)
    for pair in range(N_PAIRS):
        for lane, counts in (("background", background_spectra[pair]),
                             ("injected", injected_spectra[pair])):
            for m in MASSES:
                records.append(dict(evaluate(counts, int(m)), lane=lane, pair=pair))
        print(f"Completed pair {pair+1}/10 ({2*(pair+1)}/20 spectra); elapsed {time.monotonic()-started:.1f}s", flush=True)
    toys = pd.DataFrame(records)
    toys.to_csv(OUTPUT / "twenty_toy_scans.csv", index=False)
    bg = toys[toys.lane == "background"].pivot(index="pair", columns="mass_MeV", values="signed_r").to_numpy()
    sb = toys[toys.lane == "injected"].pivot(index="pair", columns="mass_MeV", values="signed_r").to_numpy()
    obs = np.array([r["signed_r"] for r in observed])
    delta = sb-bg
    at = lambda m: int(np.where(MASSES == m)[0][0])
    summary = {
        "status": "completed", "n_toy_spectra": 20, "n_paired_comparisons": 10,
        "seed": SEED, "mass_grid_MeV": MASSES.tolist(),
        "injected_full_template_events": float(injection.sum()),
        "injected_eps2": float(peak.independent_signed_eps2_hat),
        "signal_fraction_in_training": masks,
        "observed": {str(m): float(obs[at(m)]) for m in (66,71,72)},
        "common_mask_observed": {str(r["mass_MeV"]): r["signed_r"] for r in common_results},
        "median_injection_delta_r": {str(m): float(np.median(delta[:,at(m)])) for m in (66,71,72)},
        "median_toy_r_background": {str(m): float(np.median(bg[:,at(m)])) for m in (66,71,72)},
        "median_toy_r_injected": {str(m): float(np.median(sb[:,at(m)])) for m in (66,71,72)},
        "observed_closure_max_abs_delta_r": float(max(abs(r["reconstruction_delta_r"]) for r in observed)),
        "toy_null_feasible_fallbacks": int(toys.null_feasible_fallback.sum()),
        "deterministic_null_feasible_fallbacks": int(sum(r["null_feasible_fallback"] for r in means)),
        "claim_boundary": "Post-selection fixed-kernel sideband-response mechanism pilot; no calibrated significance or probability of signal. Common-mask truth is conditioned on these data. Not a full hyperparameter-reoptimized pipeline ensemble.",
        "sources": provenance, "runtime_manifest_sha256": production.RUNTIME_PROVENANCE["runtime_manifest_sha256"],
        "elapsed_seconds": time.monotonic()-started,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.grid": True,
                         "grid.alpha": .17, "legend.frameon": False,
                         "xtick.direction": "in", "ytick.direction": "in"})
    blue, red, purple = "#0072B2", "#D55E00", "#7A3E9D"
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.5), layout="constrained")
    select = (x*1000 >= 57) & (x*1000 <= 82)
    xx, tt = x[select]*1000, truth[select]
    ax = axes[0]
    ax.axvspan(60, 78, color="0.5", alpha=.055)
    err = 100*np.sqrt(truth_var[select])/tt
    ax.fill_between(xx, -err, err, color="0.7", alpha=.3, label="Common-mask GP uncertainty")
    ax.errorbar(xx, 100*(y[select]-tt)/tt, yerr=100*np.sqrt(y[select])/tt,
                fmt=".", color="black", ms=6, lw=.8, label="2021 observed counts")
    for m, color in ((66, blue), (72, red)):
        ax.plot(xx, 100*(plot_frame[f"background_m{m}"].to_numpy()[select]-tt)/tt,
                color=color, lw=2, label=f"Background for {m} MeV test")
    ax.axhline(0, color="0.4", lw=.8)
    ax.set(xlabel="Invariant mass (MeV)", ylabel="Difference from common-mask GP (%)",
           title="Same data, different background masks", xlim=(57,82))
    ax.legend(loc="upper center", bbox_to_anchor=(.5,-.19), fontsize=8, ncol=2)
    ax = axes[1]
    ax.plot(MASSES, obs, "o-", color="black", ms=3, lw=1.7, label="Usual moving-mask fit")
    ax.scatter([r["mass_MeV"] for r in common_results], [r["signed_r"] for r in common_results],
               marker="D", s=48, color=purple, zorder=5, label="Common 60–78 MeV mask")
    ax.axhline(0, color="0.5", lw=.8)
    ax.set(xlabel="Mass hypothesis (MeV)", ylabel="Signed local likelihood-ratio diagnostic r",
           title="Observed peak and dip: mask dependence", xlim=(60,80))
    ax.legend(loc="upper center", bbox_to_anchor=(.5,-.19), fontsize=8)
    fig.savefig(FIGURES / "observed_background_mask_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15.3, 5.6), layout="constrained")
    ax = axes[0]
    for matrix, color, label in ((bg, blue, "Background only (10 toys)"),
                                  (sb, red, "+ 66 MeV signal (10 toys)")):
        qlo, median, qhi = np.quantile(matrix, [.16,.5,.84], axis=0)
        ax.fill_between(MASSES, qlo, qhi, color=color, alpha=.18)
        ax.plot(MASSES, median, color=color, lw=2, label=label)
    ax.plot(MASSES, obs, color="black", lw=1.5, label="2021 observed")
    ax.axhline(0, color="0.5", lw=.8)
    ax.set(title="Sideband-refitted scans: 20 toys total", xlabel="Mass hypothesis (MeV)",
           ylabel="Signed local diagnostic r", xlim=(60,80))
    ax.legend(loc="upper center", bbox_to_anchor=(.5,-.20), fontsize=8)
    ax = axes[1]
    for pair in range(N_PAIRS):
        ax.plot(MASSES, delta[pair], color="0.65", alpha=.7, lw=.8)
    ax.plot(MASSES, np.median(delta, axis=0), color=red, lw=2, label="Median of 10 paired differences")
    md = pd.DataFrame(means).pivot(index="lane", columns="mass_MeV", values="signed_r")
    ax.plot(MASSES, md.loc["injected"]-md.loc["background"], color=blue, lw=1.5,
            ls="--", label="Deterministic mean spectra")
    ax.axhline(0, color="0.5", lw=.8)
    ax.set(title="Change caused by adding the signal", xlabel="Mass hypothesis (MeV)",
           ylabel="Paired change in r (injected − background)", xlim=(60,80))
    ax.legend(loc="upper center", bbox_to_anchor=(.5,-.20), fontsize=8)
    ax = axes[2]
    i, j = at(66), at(71)
    for pair in range(N_PAIRS):
        ax.plot([bg[pair,i], sb[pair,i]], [bg[pair,j], sb[pair,j]], color="0.7", lw=.8)
    ax.scatter(bg[:,i], bg[:,j], color=blue, s=26, label="Background only (10)")
    ax.scatter(sb[:,i], sb[:,j], color=red, s=26, label="Signal added (10)")
    ax.scatter([obs[i]], [obs[j]], color="black", marker="*", s=150, label="2021 observed", zorder=5)
    ax.axhline(0, color="0.6", lw=.7)
    ax.axvline(0, color="0.6", lw=.7)
    ax.set(title="Paired response at the peak and dip", xlabel="r at 66 MeV", ylabel="r at 71 MeV")
    ax.legend(loc="upper center", bbox_to_anchor=(.5,-.20), fontsize=8)
    fig.savefig(FIGURES / "twenty_toy_peak_dip_response.png", dpi=180)
    plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k not in ("sources",)}, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
