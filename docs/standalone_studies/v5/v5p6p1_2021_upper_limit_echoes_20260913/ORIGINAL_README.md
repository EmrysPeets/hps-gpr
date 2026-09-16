# HPS GPR v5.6.1: projected upper limits and GP echoes

Extension of the frozen v5.6.0 peak catalogue. Reuses its exact 15 scenarios and 300 Poisson spectra; no new toy or observed 100% 2021 data is generated or read. The report follows the existing v5.6 LaTeX design.

Every spectrum is scanned at 1 MeV steps. The historical 1% lane uses 53–250 MeV, omitting the documented 50–52 MeV edge; native 10% uses 50–250 MeV. Source-specific kernel coordinates and scaled detector resolution remain fixed. The GP is reconditioned on each spectrum and each moving exclusion, and the signal yield and GP background nuisance are profiled with the inherited bounded 90% asymptotic CLs solver.

The main plot is the pseudo-observed epsilon-squared upper limit with per-spectrum count-density normalization and the inherited electron/dimuon branching convention. Limits on the full Gaussian yield and a fixed-continuum-density conversion are also saved. Echo landmarks use the ratio of the injected-model yield limit to the background-only Asimov reference, which isolates the fit response from density normalization. They are conditional model predictions, not guaranteed future structures or an unblinding authorization.

## Products

- `report.pdf`: extension report.
- `derived/echo_catalogue.csv`: primary left/right echo landmarks and depths; integer-grid 10% depression extents; toy location spread and fixed-location ratios.
- `derived/all_Asimov_dips.csv`: additional full-scan local depressions, separated from the primary 2–8 sigma-flank catalogue.
- `derived/toy_echo_locations.csv`: per-toy flank minima and ratios at the fixed Asimov landmark.
- `derived/pointwise_summary.csv`: descriptive 20-toy quantiles.
- `derived/scans/`: all 345 scanned spectra and per-file dependency fingerprints.
- `inputs/toys/`: unchanged parent 300 spectra.
- `inputs/parent/`: parent provenance and original catalogue PDF.
- `qa/`: numerical, rendering, provenance and portable-build checks.

## Reproduction

Use Python3 with NumPy, SciPy, pandas, Matplotlib, uproot, PyMuPDF and Pillow, plus Tectonic. From this package directory:

```sh
python3 scripts/scan_limits.py run
python3 scripts/summarize_echoes.py
python3 scripts/validate_echoes.py
python3 scripts/make_report.py
cd source
tectonic --keep-logs main.tex
```

The scan uses one process and forces one BLAS/OMP thread. Completed per-spectrum checkpoints are reused only after input/source fingerprints and output hashes match. For a fresh numerical run, copy the package and clear `derived/scans/` in that copy first. The numerical engine and all needed inputs are shipped locally; original provenance paths are informational.

The two source lanes have different selections and unverified exact relative exposure. Twenty toys give descriptive spreads; they do not establish coverage or the probability that future data will reproduce a chosen dip. Comparison after unblinding should retain the saved positions, windows, amplitude assumptions and analysis settings.
