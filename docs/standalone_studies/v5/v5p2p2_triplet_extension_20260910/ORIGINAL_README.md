# v5.2.2: weak middle-component extension

Open `report.pdf` for the new triplet study. The delivered complete-study PDF appends this extension to the preserved 24-page parent report. This is an observed-motivated deterministic study with **zero new toys**, completed within the requested 20-minute work limit.

## Reproduce

Use Python 3.9+ with NumPy, SciPy, pandas, matplotlib, scikit-learn, PyMuPDF and Pillow; Tectonic builds the PDF. In the original checkout the interpreter is `venv/bin/python`. From this study folder:

```bash
python scripts/triplet_fits.py
python scripts/triplet_matched_domain.py
python scripts/triplet_response.py
python scripts/triplet_response.py --fit-scans
python scripts/plot_triplets.py
python scripts/make_triplet_report.py
bash scripts/build.sh
python scripts/render_verify.py
python scripts/portable_check.py
python scripts/package.py
```

Response scenarios are checkpointed in `derived/triplet_response_chunks`. Existing chunks are reused. Move old chunks aside before changing the generator or inputs; do not mix configurations. `response_scan.py` is a pinned parent helper module imported by the triplet code, not a reproduction entrypoint for this smaller package.

The primary injection holds the parent continuum, outer masses and outer amplitudes fixed. Its middle strength is a fraction of the weaker outer **epsilon-squared** amplitude. Both reference and effective fractions, epsilon ratios and full-support event yields are saved, including the fixed-middle-yield controls. The frozen-training comparison starts from the parent **pair** spectrum. It isolates the added middle component's effect on training while still profiling background nuisances.

The fitted alternatives use identical parent count bins and GP constraints. A separate matched-domain control accounts for different reference years: all-three reference masses 75--80 MeV map to 78--83 MeV in 2016, while the initial 2016+2021 scan uses 75--80 MeV directly in 2016. The matched control covers 78--83 MeV in the latter fit without changing its experiment or replacing the primary results.

## Contents

- `triplet_fit_scan.csv`, `triplet_fit_best.csv`: primary count fits and selected weak/unrestricted triplets.
- `triplet_matched_domain_*`: explicitly separate reference-domain comparison.
- `triplet_fit_exports.json`: full-support fitted-mean manifest, with invalid extensions flagged rather than clipped.
- `triplet_fit_*_full.npz`: counts, components, amplitudes, positive continuum and full fitted mean.
- `triplet_limits.csv.gz`, `triplet_metrics.csv`: all observed/deterministic half-MeV CLs curves and descriptive response metrics.
- `triplet_yields_*.csv`: actual coupling and event-yield ledgers for every injected spectrum.
- `PROTOCOL.md`, `qa`: design, independent fit/statistics reviews, numerical replays and rendered PDF checks.

Current file identity is pinned by `MANIFEST.json`. `inputs/manifest.json` is an unchanged historical ancestry ledger whose original source paths need not exist in the portable package. The exported NPZ histograms and local code contain the inputs used here. The parent report and selected parent tables are preserved under `inputs/parent`.

A local limit uplift is not evidence for a third particle. A positive parent signed root must not be described as a deficit being filled. The data-selected baselines, mass-domain boundaries, year-rate assumptions, frozen resolutions/conversions, inherited 2016 qualification exception and 2015 range extension remain part of the interpretation.
