# HPS-GPR v5.8.1 — alternative coherent background references

A bounded follow-up to v5.8.0, comparing six frozen 2016 generating spectra with the analysis likelihood held fixed. Read `source/report.pdf`. The nominal all-data GP substantially reduces deterministic stress bias, but direct source-injection rebuilding demonstrates substantial signal absorption. The blocked and regional constructions tested here are unsuccessful source models; their failure records are retained.

These are conditional diagnostics, not a released significance replacement. Background truths estimated from the same data are not independent controls. Numerical failure at a reference/anchor excludes the whole corresponding Monte Carlo probability; failed or partial trials are never silently dropped from a denominator.

`truths/backgrounds.npz` preserves the five initial means; `truths/supplementary.npz` contains the explicit local-kernel/edge-fallback amendment. `results` contains scans, extraction injections, and local root samples. `truths/source_absorption.csv` instead tests injections before constructing the generating mean. These are different questions.

Rebuild from saved results:

```sh
python3 scripts/make_artifacts.py
cd source
tectonic report.tex
```

Python needs NumPy, pandas, and Matplotlib for that build. Full fitting scripts additionally require the archived HPS runtime paths in the input manifests. SHA256SUMS.txt identifies the final contents. QA records preserve unsuccessful candidates rather than claiming all scientific tests passed.
