# HPS-GPR v5.8.2: nominal GP reference local and global significance

Read `source/report.pdf`. The primary figures are `figures/local_global_Z_overview.pdf` and `figures/local_global_p_overview.pdf`. Separate two-panel figures are provided for full2015, full2016, native2021/10%, and the shared-coupling search with all active datasets over19–250MeV. An additional explicitly labeled50–100MeV overlap figure contains all three datasets everywhere. All figures have PNG counterparts.

Every probability is conditional on the corresponding fixed nominal GP mean. The source kernels are fixed at76MeV for each dataset; the analysis retains its reviewed mass-dependent kernels. Source-estimation uncertainty and model selection are not calibrated here. The dashed gray curve shows the unshifted local asymptotic result; black/red show reference-local and scope-global results. Global intervals are MonteCarlo sampling intervals only.

The primary grid is0.5MeV. Individual and combined fields contain163,283,401,463 coordinates. Each dataset has256 complete Poisson spectra, shared across tested masses; matching independent dataset draws form256 combined scans. These same inputs are shared with individual scopes, so the four validation ensembles are not mutually independent. The Gaussian global estimate uses200,000 fields per scope. The method is a differential count-response approximation checked against direct perturbations and complete Poisson scans. The global correction is for each stated grid/domain, not for selecting among the reported scopes.

## Rebuild from saved results

```sh
python3 scripts/make_figures.py
tectonic source/report.tex
```

`python3 scripts/check_portable_build.py` verifies an isolated saved-result rebuild against the supplied PDF text and all rendered pixels. NumPy, SciPy, pandas, Matplotlib, pypdf, Pillow, Tectonic and pdftoppm are needed for that check.

## Replay numerical work

The copied fitting engine and all required spectra are bundled. Python requires NumPy, SciPy and pandas. `prepare.py` is the exact executed source-builder snapshot; its prior2016 check refers to the originating checkout. `prepare_portable.py` makes that optional comparison only when the prior study exists and otherwise uses the identical source-generation algorithm. The saved inputs permit replay without rebuilding them.

```sh
python3 scripts/prepare_portable.py
python3 scripts/validate_response.py
python3 scripts/check_full_response.py
python3 scripts/run_scan.py 0 3
python3 scripts/run_scan.py 1 3
python3 scripts/run_scan.py 2 3
python3 scripts/analyze.py
python3 scripts/validate_outputs.py
python3 scripts/make_figures.py
```

The three `run_scan` workers own disjoint mass coordinates and can run concurrently, each with one BLAS thread. Existing checkpoints are reused; use a fresh copy with its checkpoint files removed to rerun fits. Subsequent source or model changes require new checkpoints, not reuse of old ones. `analyze.py` regenerates the deterministic-seed Gaussian ensembles. `validate_outputs.py` optionally compares the observed roots with the prior checkout if available.

`fields/*.npz` holds response matrices, correlations, raw observed roots, coherent validation roots and simulated maxima. `results/significance_curves.csv` is the full plotting ledger, with raw counts, exact intervals, atom conventions and scope ranges. `results/combined_overlap_significance.csv` records the separately declared overlap diagnostic. Protocols, input identities, numerical and rendered QA are retained. No prior release is modified.
