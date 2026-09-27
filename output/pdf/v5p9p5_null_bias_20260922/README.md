# HPS-GPR v5.9.5: apparent null bias

Study of slides 14 and 50 of `unblind_meeting_RCmeet`, completed 22 September 2026. Read `HPS_GPR_v5.9.5_Null_Bias_Study.pdf` for the standalone report.

## Findings

- A positive scan maximum is expected under a centered null. It is not itself evidence of a biased signed estimator.
- A real mass-dependent signed-root offset also exists under the frozen generating source. At 78 MeV, the direct toy mean is 0.548; the source response is 0.577. The masked GP prediction leaves a residual whose signal-template projection reproduces that offset.
- The conventional local tail is 0.00249; the conditional marginal tail under that source is 0.01138. The source-conditional full-scan tail remains 0.25092. The observation and its ordering are unchanged.
- Slide 14's expected residual statistic is near one, but one is not guaranteed by its definition. At 78 MeV, exact replay gives mean Q/Nbin = 1.0217 versus observed 2.330. The exploratory residual-scan maximum is exceeded by 20/256 saved spectra (95% binomial interval 0.0484–0.1181).
- The 2016 response-width comparison leaves a localized discrepancy at 43.5–44 MeV. No blanket validation across all campaigns is claimed.

All these ensemble results are conditional on a frozen source estimated from observed data. The quoted intervals omit source-estimation uncertainty. Neither the residual-scan tail nor agreement with source response establishes physical-background adequacy, unconditional discovery calibration, or limit coverage. The production ±2.25σ mask is fixed. No presentation edits were made for this study.

## Contents

- `source/report.tex` and generated tables: report source.
- `scripts/ideal_null.py`: analytic centered-normal maximum illustration.
- `scan_maximum/`: copied archived fields, paired Gaussian controls, local moment audit, figures and findings.
- `residual_diagnostic/`: copied source/Poisson inputs, 201-mass exact replay, analytic decomposition, signal-projection bridge and findings.
- `review/`: independent definition and numerical audit.
- `qa/`: final PDF text/rendering checks and verification evidence.
- `MANIFEST.sha256`: hashes of the packaged files, excluding the manifest itself.

## Reproduce

Use Python 3 with NumPy, SciPy and Matplotlib, plus Tectonic for the PDF. The original numerical environment used Python 3.9.6, NumPy 2.0.2 and SciPy 1.13.1. Run:

```sh
PYTHON=/path/to/python3 bash reproduce.sh
```

Omit `PYTHON` to use `python3` on PATH. No parent checkout or network data source is required for the numerical calculation. Tectonic may fetch its TeX support files on its first run. All numerical scripts select one BLAS thread; the exact replay took approximately 67 seconds in the original environment. The child pipelines were also tested in a directory without the parent checkout, reproducing their numerical CSVs/arrays exactly. Timing metadata and PDF creation metadata can vary on rerun.

The replay reuses 256 existing complete Poisson spectra; it does not draw another Poisson ensemble or reoptimize archived kernel parameters. New Gaussian counterfactuals use a recorded fixed seed and 200,000 draws. These are null controls, not new recommended production p-values. Original analysis inputs are identified by full hashes in the component manifests.

Before rerunning, verify the delivered archive with `shasum -a 256 -c MANIFEST.sha256`. Rerunning updates derived files, so the original manifest describes the delivered snapshot, not arbitrary regenerated metadata.
