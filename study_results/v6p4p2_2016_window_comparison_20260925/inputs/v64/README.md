# HPS GPR v6.4: 2016 MC central locations and signal shapes

The standalone report is `pdf/HPS_GPR_v6p4_2016_MC_Shapes.pdf`. It analyzes all 29 supplied smeared, FEE-scaled prompt target-constrained mass histograms and includes a complete native-mass catalogue. The unsmeared histograms are not fitted.

## Findings

- In the primary 40-175 MeV domain, the local Gaussian-plus-pedestal core lies 0.08-1.02 MeV below the generated mass. At 60, 100 and 160 MeV, shifts are -0.188, -0.309 and -1.017 MeV. Conditional MC statistical errors are 0.028, 0.034 and 0.181 MeV; fit-definition spreads are 0.044, 0.084 and 0.396 MeV. These are separate diagnostics.
- The sign agrees with the saved 2021 shifts, but their size is smaller. The 2016 samples retain 89.2-93.7% of the full selected probability inside two fitted core widths. The native 2021 response and its shift law should not be transferred to 2016.
- Native 2016 histograms and interpolation between nearby masses are the preferred full-shape starting point. Omitted-mass CDF differences for the neighbor interpolation are 0.14-0.76 percentage points (median 0.24), versus a median 1.04 points for one common empirical shape, 3.23 for a core-matched Gaussian and 2.05 for a full-mean/RMS Gaussian. These are descriptive MC discrepancies, not inference calibration.
- The 30 MeV histogram has only 84 entries and an invalid primary core fit. The 35 MeV histogram has a definition-sensitive width and is reported separately. No 150 MeV file was supplied. Both low-mass samples remain visible in the report and data tables.

## Contents and reproduction

`inputs/root/` contains unchanged source files. `histograms/` contains portable counts and full-normalization probabilities. `results/centers_and_shapes.csv` gives all locations, widths and window fractions; `results/bootstrap_fits.csv` preserves every attempted replica fit, including failures. `figures/` contains the 11 figure pairs as PDF and PNG, and `source/report.md` is the readable report source. SHA-256 identities and the runtime are recorded in `provenance/`.

Use Python 3.9 or newer with the packages in `requirements.txt`. From this directory:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python3 scripts/analyze.py
python3 scripts/make_figures.py
python3 scripts/build_report.py
python3 scripts/validate.py
```

No repository parent or Downloads input is needed after extraction. Source paths in the provenance files are descriptive only. The prior-report audit checks external PDFs only when those paths are available and does not alter them. A concurrent v6.3.6 PDF change observed during authoring is preserved and documented separately from numerical validation.

To create an explicitly interpolated example at the missing 150 MeV point:

```sh
python3 scripts/template.py --mass 150 --output histograms/interpolated_example_m150.npz
```

The template API returns bin probabilities plus below- and above-support categories. It does not normalize to a fit window. Requests outside the qualified 40-175 MeV interpolation domain fail explicitly. This is an exploratory template provider; it makes no production configuration change.

## Interpretation

The core locator follows the v6.1 definition reused in v6.3.6. It uses bin-integrated Gaussian expectations, Poisson deviance and a nonnegative affine pedestal in a local mask. Means and quantiles use histogram-level conventions. Replica errors assume independent Poisson bins and do not include detector calibration or selection uncertainty. The common-shape comparison conditions on the measured center and width; the held-out neighbor morph predicts those as well.

Prompt target constraint and prior FEE scaling/smearing are supplied sample metadata. Selection equivalence to the intended 2016 data analysis is not established by these ROOT files. This study supplies shape evidence, not efficiency, rate, yield-recovery, coverage, exclusion or significance results. No observed data, unsmeared-resolution study, GP extraction or remote computation is run.
