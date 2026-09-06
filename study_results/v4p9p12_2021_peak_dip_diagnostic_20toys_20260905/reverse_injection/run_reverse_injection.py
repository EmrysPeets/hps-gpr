#!/usr/bin/env python3
"""Compare deterministic 66/78/80 MeV injections; generate zero new toys."""
from pathlib import Path
import hashlib
import json
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PILOT = HERE.parent
sys.path.insert(0, str(PILOT))
import run_diagnostic as pilot

np, pd, plt = pilot.np, pilot.pd, pilot.plt
production = pilot.production
MASSES = np.arange(60, 89)
INJECTIONS = (66, 78, 80)
FOCAL = (66, 71, 72, 78, 80, 85)
OUTPUT, FIGURES = HERE / "derived", HERE / "figures"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    OUTPUT.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    protected_paths = sorted((PILOT / "derived").glob("*")) + sorted((PILOT / "figures").glob("*"))
    protected_paths += [PILOT / "run_diagnostic.py", PILOT / "PROTOCOL.md"]
    protected = {str(p): sha(p) for p in protected_paths if p.is_file()}
    cfg = production.load_config(production.DEFAULT_CARD)
    production.validate_card(cfg)
    sources = {
        "card": production.DEFAULT_CARD, "states": production.DEFAULT_STATES,
        "spectrum": pilot.PARENT / "derived/all_three_peak_extraction_plot_data.csv",
        "curves": pilot.PARENT / "derived/final_dataset_result_curves.csv",
        "script": Path(__file__), "protocol": HERE / "PROTOCOL.md",
        "pilot_script": PILOT / "run_diagnostic.py",
    }
    source_hashes = {k: {"path": str(p), "sha256": sha(p)} for k, p in sources.items()}
    frame = pd.read_csv(sources["spectrum"])
    frame = frame[frame.dataset.astype(str) == "2021"].sort_values("bin_center_GeV")
    x = frame.bin_center_GeV.to_numpy(float)
    y = frame.observed_events.to_numpy(float)
    widths = frame.bin_width_MeV.to_numpy(float)/1000
    edges = np.r_[x-widths/2, x[-1]+widths[-1]/2]
    assert np.array_equal(y, np.rint(y)) and np.all(y > 0)
    assert 0.035 < edges[0] < 0.037 and 0.299 < edges[-1] < 0.301
    states = production.state_map(pd.read_csv(sources["states"]))
    curves = pd.read_csv(sources["curves"])
    curves = curves[(curves.scope_key == "individual_2021_10pct") & curves.mass_MeV.isin(MASSES)]
    curves = curves.sort_values("mass_MeV").set_index("mass_MeV")
    assert len(curves) == len(MASSES)
    sigma = lambda m: float(np.polynomial.polynomial.polyval(m, cfg.sigma_coeffs_2021))

    def fit_background(counts, mass, common=False):
        state = states[("2021", 66 if common else int(mass))]
        m = mass/1000
        lo, hi = (0.060, 0.086) if common else (m-2.25*sigma(m), m+2.25*sigma(m))
        keep = (x < lo) | (x > hi)
        return pilot.fit_gpr(x[keep], counts[keep], cfg, restarts=0,
                             kernel=pilot.make_fixed_kernel(state["const_opt"], state["ls_opt"]),
                             optimize=False)

    def evaluate(counts, mass):
        m = mass/1000
        mask = (x >= m-2.25*sigma(m)) & (x <= m+2.25*sigma(m))
        model = fit_background(counts, mass)
        mean, rawcov = pilot.predict_counts_from_log_gpr(model, x[mask], cfg)
        cov, conditioning = production.condition_covariance_block(rawcov, mean)
        template, _ = pilot.build_window_template_from_full(edges, mask, m, sigma(m), config=cfg)
        template = np.asarray(template, float)
        template /= template.sum()
        fit = pilot.fit_A_profiled_gaussian_details(counts[mask], mean, cov, template, allow_negative=True)
        null = pilot.profiled_gaussian_fixed_poi_nll(counts[mask], mean, cov, template, A_fixed=0)
        if not fit["success"] or not null["success"]:
            raise RuntimeError(f"Failed profile at {mass} MeV")
        raw_q = 2*(float(null["nll"])-float(fit["nll"]))
        if not np.isfinite(raw_q):
            raise RuntimeError(f"Nonfinite likelihood difference at {mass}")
        fallback = raw_q < 0
        amp = 0.0 if fallback else float(fit["A_hat"])
        r = float(np.sign(amp)*np.sqrt(max(raw_q, 0)))
        return {"mass_MeV": int(mass), "signed_r": r, "A_hat_window": amp,
                "sigma_A_window": float(fit["sigma_A"]), "raw_twice_delta_nll": raw_q,
                "null_feasible_fallback": bool(fallback),
                "diagonal_load_relative": conditioning["selected_diagonal_load_relative"]}

    observed = []
    for mass, row in curves.iterrows():
        result = evaluate(y, int(mass))
        base = json.loads(row.limit_profile_status)["observed"]["base"]
        fit = base["fit_unbounded"]
        saved = float(np.sign(fit["A_hat"])*np.sqrt(max(2*(base["null"]["nll"]-fit["nll"]), 0)))
        result.update(saved_signed_r=saved, reconstruction_delta_r=result["signed_r"]-saved)
        if abs(result["reconstruction_delta_r"]) > .02:
            raise RuntimeError(f"Observed reconstruction mismatch: {result}")
        observed.append(result)
    observed = pd.DataFrame(observed)
    observed.to_csv(OUTPUT / "observed_reconstruction.csv", index=False)
    print(f"Observed closure: max |delta r|={observed.reconstruction_delta_r.abs().max():.3g}", flush=True)

    truth, variance = pilot.predict_counts_mean_var_from_log_gpr(fit_background(y, 66, common=True), x, cfg)
    assert np.all(np.isfinite(truth)) and np.all(truth > 0)
    spectrum = pd.DataFrame({"mass_MeV": x*1000, "observed_counts": y,
                             "smooth_truth_counts": truth, "truth_gp_variance": variance})
    templates, metadata = {}, {}
    for mass in INJECTIONS:
        row = curves.loc[mass]
        m = mass/1000
        mask = (x >= m-2.25*sigma(m)) & (x <= m+2.25*sigma(m))
        _, full = pilot.build_window_template_from_full(edges, mask, m, sigma(m), config=cfg)
        full = np.asarray(full, float)
        assert np.isclose(full.sum(), 1.0, rtol=1e-12) and np.all(full >= 0)
        eps2 = float(row.limit_fit_unconstrained_eps2)
        total = eps2*float(row.signal_yield_per_eps2_total)
        assert eps2 > 0 and total > 0
        injection = total*full
        if mass == 66:
            assert np.allclose(injection, frame.independent_signed_signal_events, rtol=1e-8, atol=1e-9)
        templates[mass] = injection
        spectrum[f"signal_m{mass}_counts"] = injection
        training_fraction = {}
        for test_mass in FOCAL:
            tm = test_mass/1000
            keep = (x < tm-2.25*sigma(tm)) | (x > tm+2.25*sigma(tm))
            training_fraction[str(test_mass)] = float(injection[keep].sum()/injection.sum())
        metadata[str(mass)] = {"injected_eps2": eps2, "full_template_events": total,
                               "resolution_sigma_MeV": sigma(m)*1000,
                               "signal_fraction_in_training": training_fraction}
    spectrum.to_csv(OUTPUT / "common_truth_and_signals.csv", index=False)

    records = []
    for lane, counts in [("background", truth)]+[(f"inject_{mass}", truth+templates[mass]) for mass in INJECTIONS]:
        for mass in MASSES:
            records.append(dict(evaluate(counts, int(mass)), lane=lane))
        print(f"Finished {lane}: 29 deterministic fits, 0 new toys", flush=True)
    scans = pd.DataFrame(records)
    scans.to_csv(OUTPUT / "deterministic_scans.csv", index=False)
    grid = scans.pivot(index="mass_MeV", columns="lane", values="signed_r").sort_index()
    changes = pd.DataFrame({f"inject_{mass}": grid[f"inject_{mass}"]-grid.background for mass in INJECTIONS})
    changes.index.name = "mass_MeV"
    changes.to_csv(OUTPUT / "injection_induced_delta_r.csv")
    for p, digest in protected.items():
        if sha(p) != digest:
            raise RuntimeError(f"Original pilot artifact changed: {p}")
    summary = {
        "status": "completed", "new_toy_spectra": 0, "original_toy_spectra": 20,
        "n_deterministic_scenarios": 4, "n_deterministic_fits": len(scans),
        "mass_grid_MeV": MASSES.tolist(), "truth_excluded_window_MeV": [60,86],
        "truth_kernel_anchor_MeV": 66, "injections": metadata,
        "observed_r": {str(int(r.mass_MeV)): float(r.signed_r) for r in observed.itertuples() if r.mass_MeV in FOCAL},
        "deterministic_r": {lane: {str(m): float(grid.loc[m,lane]) for m in FOCAL} for lane in grid.columns},
        "injection_induced_delta_r": {lane: {str(m): float(changes.loc[m,lane]) for m in FOCAL} for lane in changes.columns},
        "null_feasible_fallbacks": int(scans.null_feasible_fallback.sum()),
        "observed_closure_max_abs_delta_r": float(observed.reconstruction_delta_r.abs().max()),
        "original_pilot_artifacts_unchanged": True, "protected_sha256": protected,
        "source_hashes": source_hashes,
        "runtime_manifest_sha256": production.RUNTIME_PROVENANCE["runtime_manifest_sha256"],
        "elapsed_seconds": time.monotonic()-started,
        "claim_boundary": "Data-selected, deterministic, fixed-kernel response study on one smooth generating truth. No calibrated p-values, hypothesis ranking, or physical signal identification.",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")

    colors = {66: "#7A3E9D", 78: "#0072B2", 80: "#D55E00"}
    styles = {66: "--", 78: "-", 80: "-."}
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.grid": True,
                         "grid.alpha": .17, "legend.frameon": False,
                         "xtick.direction": "in", "ytick.direction": "in"})
    fig, axes = plt.subplots(1,2,figsize=(13.4,5.8),layout="constrained")
    axes[0].plot(MASSES, observed.signed_r, "k.-", lw=1.7, label="2021 observed")
    axes[0].plot(MASSES, grid.background, color="0.5", ls=":", lw=1.6, label="Smooth background only")
    for mass in INJECTIONS:
        label=f"+ {mass} MeV signal"
        axes[0].plot(MASSES, grid[f"inject_{mass}"], color=colors[mass], ls=styles[mass], lw=1.7, label=label)
        axes[1].plot(MASSES, changes[f"inject_{mass}"], color=colors[mass], ls=styles[mass], lw=2, label=label)
    for ax in axes:
        ax.axhline(0,color="0.5",lw=.8)
        ax.set(xlabel="Mass hypothesis (MeV)",xlim=(60,88))
        ax.legend(loc="upper center",bbox_to_anchor=(.5,-.20),fontsize=8,ncol=2)
    axes[0].set(title="Deterministic scans on one common background",ylabel="Signed local diagnostic r")
    axes[1].set(title="Response caused by each positive-only injection",ylabel="Change in r relative to background only")
    fig.savefig(FIGURES / "reverse_injection_scans.png",dpi=180)
    plt.close(fig)

    matrix = changes.loc[list(FOCAL)].to_numpy()
    scale = float(np.max(np.abs(matrix)))
    fig, ax = plt.subplots(figsize=(6.2,5.2),layout="constrained")
    ax.grid(False)
    mesh=ax.imshow(matrix,cmap="RdBu_r",vmin=-scale,vmax=scale,aspect="auto")
    ax.set_xticks(range(3),[f"{m} MeV" for m in INJECTIONS])
    ax.set_yticks(range(len(FOCAL)),[f"{m} MeV" for m in FOCAL])
    ax.set(xlabel="Injected signal location",ylabel="Tested mass hypothesis",
           title="Injection-induced response: 0 new toys")
    for i in range(len(FOCAL)):
        for j in range(3):
            ax.text(j,i,f"{matrix[i,j]:+.2f}",ha="center",va="center",
                    color="white" if abs(matrix[i,j]) > .55*scale else "black",fontsize=12)
    fig.colorbar(mesh,ax=ax,label="Change in signed local diagnostic r",shrink=.9)
    fig.savefig(FIGURES / "reverse_injection_response_matrix.png",dpi=180)
    plt.close(fig)
    print(json.dumps({k:summary[k] for k in ["injections","observed_r","deterministic_r","injection_induced_delta_r","null_feasible_fallbacks","elapsed_seconds"]},indent=2),flush=True)


if __name__ == "__main__":
    with pilot.threadpool_limits(limits=1):
        main()
