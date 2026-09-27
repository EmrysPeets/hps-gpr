# HPS-GPR v5.8.0: local significance, stress response and scan spacing

Standalone follow-up to v5.0.5 Sections 4.20 and 6.11, using v5.1.0–1 as the prior diagnostic baseline. Created 17 September 2026 within the requested 40-minute budget. The final delivery record reports actual elapsed time.

Read `source/report.pdf`. New numerical results remain conditional on the backgrounds and fitting policies recorded in their protocols. The study does not replace the released upper-limit scan or certify a new particle significance.

The study separates:

- Finer tested-mass spacing from histogram binning, signal-window width and search-domain selection.
- The signed likelihood root from a stress-centered coordinate and a scan-global probability.
- Actual historical 2016 10% counts from a same-shape exposure counterfactual.
- A shorter kernel's bias reduction from its larger uncertainty and its signal recovery.
- Shared-coupling information weighting from a change in physical local significance calibration.

`gpr/` holds coherent response fields and nested-grid tests. `local/` holds new direct local-tail experiments. `historical10/` contains the saved subset histogram, matching engine and actual-subset tests. `statistics/` provides independent coupling, stiffness and probability checks. `reviews/` contains the paper and prior-test audits. `qa/` records numerical and PDF checks. JSON manifests and the final `SHA256SUMS.txt` identify all saved artifacts.

## Rebuild the note and figures

From this study directory:

```sh
python3 scripts/make_artifacts.py
python3 gpr/summarize_grid.py
cd source
tectonic report.tex
```

The report figures/tables use only saved results and require Python with NumPy, SciPy, pandas and Matplotlib, plus Tectonic. `scripts/make_artifacts.py` does not rerun fits or draw random numbers. The GP summary regenerates inexpensive Gaussian fields with its saved deterministic seed; it does not regenerate Poisson data.

The full fitting scripts record legacy HPS runtime dependencies in their manifests. They are intentionally not described as a completely portable analysis environment. Full-support truth arrays, signed roots, source spectra, exact input hashes and deterministic seeds are included. Historical-subset refitting is self-contained inside its directory.

The PDF was checked semantically and by rendering every final page. Numerical QA verifies full tables, source identity, probability counts/intervals, solver checks and preserved shared coordinates. The final archive excludes external legacy data files and prior long analysis-note PDFs; their source hashes identify them.
