# HPS-GPR v5.3: MC resolution and upward broadening

Standalone observed-data study dated 10 September 2026. The report compares unscaled MC and progressively corrected mass resolutions for full 2015, full 2016, and native 2021 10% data, both individually and in their simultaneous available-dataset combination.

Read `report.pdf`. The authoritative numerical results are `derived/scans.csv`; `derived/grid_envelope.csv` records the additional upward variation relative to MC. No expected-limit bands or toys are generated.

The interpolation is sigma(t) = sigma_MC + t (sigma_scaled - sigma_MC), for t = 0, 0.4, 0.6, 1. For 2021 these are 1, 1.10, 1.15, 1.25 times its baseline. The 2015 endpoint also includes the existing 3% target allowance; its MC curve is reconstructed from the displayed, rounded calibration ratio. For 2016 the endpoints are distinct raw and track-smeared polynomials, so their ratio depends on mass. See `resolution_audit.md` and the source-page PDF.

`reviewed_seed_refit` is the main lane: scenario-specific GP bounds and masks, 12 seeded optimizer restarts plus a start from the reviewed kernel projected into the same bounds, full covariance propagation, bin-integrated signal shapes, and recalculated observed prompt densities. The choice among the random-restart result, reviewed seed, and its local optimization uses only the sideband log marginal likelihood. No choice uses an observed limit or p-value. `refit` retains the initial random-restart results as an optimizer diagnostic. `fixed_kernel` holds the reviewed kernel hyperparameters fixed but recomputes the GP conditioning with the new masks. The comparison helps separate refitting effects from the changed signal/extraction window. All lanes use the same safeguarded Poisson/Gaussian profile solver and asymptotic 90% CLs construction.

The envelope is the maximum of four scenario limits. It is neither a nuisance-profiled interval nor a calibrated one-sigma error. A data/MC discrepancy alone does not supply the calibration likelihood needed for that interpretation. The p-values are conditional local asymptotic values at fixed mass and scenario; minima across the mass/width scan have no look-elsewhere calibration. The main combination follows a coherent interpolation across datasets, not an exhaustive independent-nuisance envelope.

The raw `epsilon2_90` column uses the electron-channel convention. `display_epsilon2_90` applies the previously released dimuon branching factor once at each mass to match the prior note's minimal-visible plots. P-values and event yields are unaffected.

## Rebuild figures and report

Use Python with numpy, scipy, pandas, matplotlib, scikit-learn, reportlab and PyMuPDF. From this folder:

```sh
python3 scripts/aggregate.py
python3 scripts/make_report.py
```

All observations required for refitting are bundled as native and rebinned histogram arrays; the historical ROOT paths in the input card are provenance only. Relevant GP routines are vendored under `scripts/vendor_hps`, and the profile solver is bundled. To recompute in a separate directory without replacing the delivered scan, run:

```sh
python3 scripts/reproduce_fits.py /absolute/path/to/new-v53-rerun
```

This copies the source, inputs and protocol into a new directory and sets a fresh 25-minute fit budget. The refit controller uses two processes, each restricted to one linear-algebra thread. During the delivered run a third single-thread process performed the uniform reviewed-seed check as checkpoints arrived. A full rerun is dependent on machine speed; missing rows stay missing and are enumerated by the aggregation QA.

## Numerical and source checks

`qa/scan_validation.json` states completed/missing coordinates and solver checks. `qa/statistics_numerical_review.md` explains comparisons to legacy optimizers. One fixed-kernel covariance repair at unscaled 2016, 73 MeV is documented, with original checkpoint preserved and a doubled-loading stability check. Recomputed fully scaled controls can differ slightly from the archived release because of numerical likelihood minimization and covariance conditioning; they are not asserted to be bitwise reproductions.

Timing benchmark checkpoints are retained only under `qa/benchmark_chunks` and do not enter the reported curves. Each initial refit scenario runs in a separate process; the reviewed-seed check clears the resolution-dependent GP cache at every context.

`inputs/manifest.json` pins copied inputs. `MANIFEST.sha256` identifies the delivered package. The immutable parent studies and shared analysis code were not edited.
