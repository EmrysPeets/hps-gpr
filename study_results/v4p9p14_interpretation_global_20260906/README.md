# v4.9.14: understanding calibration and a 2015 global-significance pilot

This derivative preserves the v4.9.13 release and explains its consequences
in the same LaTeX style. The prior studies were verified against 2,693 committed
and local files and merged into main in [PR 66](https://github.com/EmrysPeets/hps-gpr/pull/66),
merge `49c7103c3b7827206e1e9bb2dc6ecda868cd63dd`. The shared checkout's branch,
index and unrelated edits were preserved.

The reader report is
`output/pdf/v4p9p14_interpretation_global_20260906/HPS_GPR_v4p9p14_Calibration_Explained_and_Global_Study.pdf`.
Its LaTeX source is `note/reader_report.tex`. The independent HEP review is
`review/HEP_STATISTICAL_REVIEW.md`.

## Interpretation

The combined local-p-value plateau around 66 MeV is dominated by an archived
stress truth producing a large positive null offset, mainly from 2016. The
upper-limit hump around 73–75 MeV is its negative-offset counterpart. The
underlying observed fits did not reverse sign because they were calibrated.
Profiling remains defensible, but the current raw asymptotic mapping has
measured failures in the specified toy families. A conservative envelope can
substantially lose power under one family member while protecting against
another. These are conditional diagnoses, not observed-data bias measurements
or unconditional final-analysis certification.

## Global-significance study

Following Ananiev and Read, arXiv:2206.12328v3, estimate a significance-field
covariance using one-bin Asimov perturbations. Extend its regular unbiased
premise explicitly by retaining deterministic offsets and response scales.
Use one common full 2015 stress spectrum across all 72 hypotheses. The local
and global curves preserve the bounded discovery atom and use a declared
minimum-local-p ordering. A separate raw-maximum ordering is retained in JSON.
Neither globally calibrates the old mass-dependent two-truth envelope.

- `global/2015/pilot10/`: ten complete Poisson scans, separate timing pilot.
- `global/2015/validation1000/`: 1,000 independent complete Poisson scans.
- `global/2015/asimov/`: one unfluctuated spectrum and 484 one-bin perturbations.
- `global/2015/analysis/`: covariance, 200,000 GP maxima, p-value CSVs, marginal
  diagnostics, finite-grid comparisons, and exact binomial tail intervals.
- `figures/`: separate local/global and validation figures plus the hump diagnosis.
- `provenance/`: publication verification, source/figure/PDF identities.
- `qa/`: numerical, semantic and rendered-page checks.

All saved spectra retain correlations across mass. Old pointwise toy IDs were
not joined or pooled. Numerical work uses one worker and one BLAS thread.
Frozen parent code and native input hashes are checked before generating new
toys; the new study consumes the canonical parent runtime and its documented
external native data. See `PROTOCOL.md` and `NEXT_STEPS.md` for exact definitions
and runnable continuation commands. Machine-readable results and their actual
qualifications are in `global/2015/analysis/summary.json`.
