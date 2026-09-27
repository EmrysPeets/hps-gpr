# Completed bounded study: v5.8.4

Started 2026-09-18 21:57:15 UTC. Hard stop 22:27:15 UTC (30 minutes).

User asks: continue v5.8.3 with 1 MeV spacing; independent-experiment Fisher or similar combinations versus shared coupling; blind half-widths 2.4, 2.5, 2.6 sigma versus 2.25; compare local/global p, Z and 90% CLs reach; put plots in a standalone report. Preserve old releases.

Completed an isolated v5.8.4 study with pinned v5.8.2 spectra, GP nulls, 256 coherent Poisson spectra per dataset and 100,000 Gaussian fields per width/scope. Width changes affect the GP held-out mask and likelihood window together, following existing implementation. Sources and kernel hyperparameters remain fixed. Fisher uses exact Gaussian-reference atom-mixture calibration, accounting for nonpositive-fit p=1 atoms. Equal-weight signed Stouffer is a secondary comparison. All previous studies and production files are unchanged.

All three numerical workers completed. There are 928 checkpoint pairs, 2,628 fitted scope/mass configurations, 5,096 displayed rows and 36 peak summaries. Width 2.25 reuses frozen significance responses but recomputes observed/model-Asimov CLs limits. Other widths refit all 256 paired spectra. All 124 numerical/provenance checks and 264 derivative checks pass. Fisher algebra and atom checks pass. The isolated saved-results rebuild reproduces every CSV, the PDF text and all rendered pixels.

## Results and interpretation

At the baseline half-width 2.25 sigma and 1 MeV spacing, full-domain shared-coupling local/global Z is 2.829/0.812 at 66 MeV. Fisher is 3.723/2.280 at 92 MeV; signed Stouffer is 3.934/2.580 at 92 MeV. The common-overlap 50-100 MeV results are separate, narrower-domain tests and cannot replace these full-domain results.

The 92 MeV feature is stable across the requested 2.4, 2.5 and 2.6 sigma widths. At 2.6 sigma the median observed coupling-limit ratios to baseline are 1.031, 1.022, 1.017 and 1.020 for 2015, 2016, 2021 and shared coupling: modestly weaker reach. The paired spacing test shows that a coarser grid reduces the penalty at fixed threshold but can miss a peak; observed global significance declines for 2016 and 2021 on the integer grid.

One statistical tail check is flagged: 2016 at width 2.6 has Gaussian global p=0.09547 versus 35/256 direct exceedances, with a direct 95% interval of 0.0971-0.1850. Its separate 95%-maximum threshold test passes within available precision. Independent-combination tails are sparsely sampled directly. All estimates remain conditional on fixed observed-data-derived GP sources; uncertainty from source estimation and selection among methods, widths and scopes is not calibrated. Fisher alone has no unique common-coupling reach.

## Delivered work and remaining work

The standalone report is `source/report.pdf` (11 pages), with 12 PDF/PNG figure pairs in `figures/`, reproducible code, fixed inputs, checkpoints, machine-readable result ledgers and QA records. Delivery packages are under `output/pdf/v5p8p4_independent_combinations_windows_20260918` in the checkout. Hashes and completion timing are recorded in the package manifests and `qa/final_validation.json`.

No requested analysis remains. Future work, outside this bounded study, would be additional independent direct-Poisson tail validation, propagation of source-estimation uncertainty, and calibration of any predeclared selection among methods or widths. These are limitations rather than unfinished computations promised by this study.

Science Python `/opt/homebrew/bin/python3`; pypdf/Pillow are in bundled runtime Python `/Users/emryspeets/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3`. Tectonic and pdftoppm are on PATH. See README.md for replay and saved-results rebuild commands.
