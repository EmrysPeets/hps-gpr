# v4.2 65 MeV extraction binning follow-up

Status: **PASS**

This directory is a self-contained follow-up to Figures 61 and 62 of the
authoritative HPS-GPR v4.2 note (study commit
`fb1295680bacdd5edbabff9546ee200e3c68b78a`). It does not edit or replace the
analysis note.

## Main finding

An exact 0.625 MeV rebin is not available from the supplied histogram-only
2015/2016 inputs. Their source bins are 0.05 MeV wide, so the required factor is
12.5. Splitting source-bin counts would create fractional pseudo-counts and
invalidate the integer-Poisson likelihood. The ROOT inputs contain no event
TTrees or RNTuples from which the mass histograms could be rebuilt. The study
therefore uses 0.5 MeV, the finer of the equidistant 0.5/0.75 MeV
source-compatible choices and the nearest one that retains all three full
supports, with integer factors 10/10/4. An exact 1.25 MeV coarsening stress
test uses factors 25/25/10.

## Fixed-mass results at 65 MeV

| Binning | Combined $\widehat{\epsilon^2}$ | Wald $\sigma$ | local $p_0$ | local $Z$ | $\Delta Z$ vs native |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native v4.2 | 6.427063e-06 | 1.610091e-06 | 3.259175e-05 | 3.993215 | 0 |
| Common 0.5 MeV | 6.207189e-06 | 1.611009e-06 | 5.803269e-05 | 3.854291 | -0.138923 |
| Common 1.25 MeV | 6.404938e-06 | 1.634577e-06 | 4.444672e-05 | 3.919064 | -0.074150 |

The 0.5 MeV result is a newly optimized local GP refit using the same physical
mass resolution, training exclusion, fit support, radiative conversion, and
profile likelihood as v4.2. The density normalization remains sourced from the
uncropped fine histogram and the exact $m\pm1.64\sigma_m$ window. The 1.25 MeV
stress histogram for 2015 ends at 134 MeV rather than 135 MeV because 121 MeV
of support is not divisible into uniform 1.25 MeV bins; this one-MeV far-side
trim is recorded in the tables and is why 0.5 MeV is the primary comparison.

This is a fixed-mass study. It tests the 65 MeV extraction and local asymptotic
significance only; it does not establish that 65 MeV remains the minimum of a
rebinned full scan and does not recompute the analytic Sidak reference or a
scan-toy global significance.

## Why the original 2016 error bar crossed zero

The native standalone 2016 result is
$\widehat{\epsilon^2}=1.519207e-06$ with symmetric local
Wald uncertainty $\sigma=2.557492e-06$. Its lower plotted
endpoint was -1.038285e-06. The fit deliberately allowed a
signed signal-strength estimator and the code divided both the fitted event
amplitude and its positive uncertainty by the same positive conversion factor.
Thus the extension below zero was not a sign or normalization bug. It was a
symmetric curvature uncertainty on an unconstrained estimator. The new
Figure 62 composite preserves the three residual panels and replaces only its
coefficient panel with the physical $\epsilon^2\geq0$ profile set; the 2016
lower endpoint is zero. The displayed 68% sets are nominal/asymptotic and have
not been coverage calibrated.

## Outputs

- `reference_v4p2/`: bitwise copies of the authoritative native extraction
  tables, provenance, and Figures 61/62.
- `tables/extraction_comparison.csv`: standalone and combined fit results for
  all three binnings, including differences from native.
- `tables/profile_intervals68.csv`: physical-domain profile intervals.
- `tables/optimizer_repeats.csv`: two independent 12-restart fits per dataset
  and alternative binning, with the maximum-LML branch selected.
- `tables/bin_level_common_0p5MeV.csv`: plotted counts and profiled
  expectations.
- `figures/figure61_common_0p5MeV.*`: clean common-bin extraction.
- `figures/figure61_common_0p5MeV_profiled.*`: exact-window profiled version.
- `figures/figure62_profiled_residuals_physical68.*`: corrected native
  Figure 62 composite preserving all three residual panels.
- `figures/figure62_coefficients_physical68.*`: corrected physical interval
  comparison between native v4.2 and the common-bin refit.
- `validation.json`: machine-readable pass/fail checks.
- `provenance.json`: input, code, commit, histogram, and output hashes.
- `CAPTIONS.md`: publication-ready caption text and interpretation boundaries.
- `VISUAL_QA.md`: manual original-resolution PNG and one-page PDF inspection
  record.

## Reproduce

From the repository root:

```bash
MPLCONFIGDIR=/tmp/codex-mpl-v4p2-m065-followup \
python3 study_results/v4p2_followups_20260806/m065_extraction/run_m065_common_binning_study.py
```

The script refuses noninteger rebins, validates native reconstruction against
the accepted v4.2 table, runs optimizer-repeat checks, verifies nonnegative
physical interval endpoints, and writes only inside this directory.
