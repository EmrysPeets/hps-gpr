# GP-only replacement-window ensemble

This directory is an additive follow-up to the frozen
`study_results/v4p2_followups_20260806/pseudo65` study.  It does not modify
that validated single-draw product.

The study constructs conditional, background-only GP-mean replacements of the
2021 10% histogram around 65 MeV and runs the frozen v4.2
observed/asymptotic scan on each replacement.  No signal is injected, and no
expected-limit bands, CLs-calibration toys, or limit-band toys are produced.

## Window geometry

At 65 MeV the v4.2 resolution is
`sigma_m = 2.121696875 MeV`.  Complete 0.625 MeV production bins are selected
by whether their centers lie inside the requested continuous interval.

- `window_2p25eq2p5`: the requested `+/-2.25 sigma_m` and
  `+/-2.5 sigma_m` choices select the same 16 bins, with exact edges
  `[60,70) MeV`.  They therefore share one ten-draw ensemble.
- `window_3p0`: `+/-3 sigma_m` selects 20 bins, with exact edges
  `[58.75,71.25) MeV`.

The two geometries use paired common random numbers.  For each draw index, the
Poisson counts in `[60,70) MeV` are identical in the narrow and wide
histograms; the wider histogram additionally replaces the four edge bins.
This isolates the window extension instead of confounding it with a second
Poisson fluctuation.  The ten draw indices are independent PCG64 child
streams of master seed `2026080603`.

The generating mean is the exact accepted v4.2 fixed-GP state at 65 MeV,
whose training exclusion remains `+/-2.25 sigma_m`.  The four analysis bins
added by the 3-sigma replacement were therefore training-sideband bins for
that fixed state.  The 3-sigma lane is a conditional smoothing/resampling
stress test around the accepted mean, not a newly trained 3-sigma-exclusion GP
truth model.

## Interpretation

Every histogram retains the observed spectrum outside its replacement window.
The ensemble is therefore conditional on that common outside observation and
is not a global-null ensemble.  Pointwise arithmetic means, medians, and
16--84% quantiles across the ten fixed draws are descriptive summaries only.
They do not establish expected sensitivity, coverage, or a scan-calibrated
global probability.

The scan card retains the v4.2 2021 physics and statistical settings:

- 50--250 MeV in 1 MeV steps;
- 0.625 MeV analysis bins;
- `blind_nsigma = gp_train_exclude_nsigma = 2.25`;
- 2021 resolution-scaled length-scale ceiling factor 15;
- 12 GP optimizer restarts per mass;
- profiled extraction with negative signed estimates allowed;
- observed/asymptotic 90% CLs and local one-sided asymptotic `p0`;
- no expected-limit bands, CLs-calibration toys, or limit-band toys.

`scan_n_workers` is reduced from six to five only as a runtime control so two
independent scans can use the ten available logical CPUs concurrently.  This
does not alter the inference.

## Descriptive results

At 65 MeV the fixed v4.2 observed-context point has
`eps2_up = 1.17183994214e-5` and local asymptotic
`p0 = 1.05702045272e-5` (`Z = 4.25249272061`).  The conditional
background-only replacements give:

| replacement geometry | median `eps2_up` | mean `eps2_up` | empirical 16--84% `eps2_up` | median local `p0` |
|---|---:|---:|---:|---:|
| `+/-2.25 sigma_m` and `+/-2.5 sigma_m` (same 16 bins) | `3.69709658522e-6` | `3.57685515667e-6` | `[2.70526168873e-6, 4.14839887304e-6]` | `0.4376862322` |
| `+/-3 sigma_m` (20 bins) | `4.27450275949e-6` | `4.19152068892e-6` | `[3.06746859302e-6, 5.12518180840e-6]` | `0.284312864526` |

The paired median ratio
`eps2_up(+/-3 sigma_m) / eps2_up(narrow) = 1.17956747173`, with
empirical 16--84% interval `[1.00971067185, 1.27080233353]`.  The two
lanes have exactly equal per-draw `integral_density` at 65 MeV because its
`+/-1.64 sigma_m` normalization window lies wholly inside their shared
counts.  The difference therefore comes from the four outer replacement bins
and the profiled GP/extraction response, not from a different epsilon-squared
normalization.

Across 55--75 MeV, none of the ten individual conditional draws reaches a
local `3 sigma` excess.  The smallest local `p0` is `0.00691677116`
(`Z = 2.46155698`) for the narrow geometry and `0.01117809527`
(`Z = 2.28426110`) for the wide geometry, both at 60 MeV in draw 04.  This is
a descriptive statement about these ten fixed conditional draws, not a
global probability or coverage result.

## Reproduction

Run from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/codex-mpl \
  python3 study_results/v4p2_followups_20260806/pseudo65/gp_window_ensemble_20260806/build_ensemble.py

PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/codex-mpl \
  python3 study_results/v4p2_followups_20260806/pseudo65/gp_window_ensemble_20260806/run_scans.py \
  --mode pilot --max-parallel 2

PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/codex-mpl \
  python3 study_results/v4p2_followups_20260806/pseudo65/gp_window_ensemble_20260806/run_scans.py \
  --mode full --max-parallel 2

PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/codex-mpl \
  python3 study_results/v4p2_followups_20260806/pseudo65/gp_window_ensemble_20260806/run_scans.py \
  --mode central_repeat --max-parallel 2

PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/codex-mpl \
  python3 study_results/v4p2_followups_20260806/pseudo65/gp_window_ensemble_20260806/review_central.py

# If review_central.py reports unreproduced selected states, repeat this pair
# with successive round numbers until the count is zero.
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/codex-mpl \
  python3 study_results/v4p2_followups_20260806/pseudo65/gp_window_ensemble_20260806/run_central_repairs.py \
  --round 3 --max-parallel 2
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/codex-mpl \
  python3 study_results/v4p2_followups_20260806/pseudo65/gp_window_ensemble_20260806/review_central.py

PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/codex-mpl \
  python3 study_results/v4p2_followups_20260806/pseudo65/gp_window_ensemble_20260806/bind_results.py

PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/codex-mpl \
  python3 study_results/v4p2_followups_20260806/pseudo65/gp_window_ensemble_20260806/postprocess.py

PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/codex-mpl \
  python3 study_results/v4p2_followups_20260806/pseudo65/gp_window_ensemble_20260806/validate_ensemble.py \
  --stage final

PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/codex-mpl \
  python3 -m pytest -q \
  study_results/v4p2_followups_20260806/pseudo65/gp_window_ensemble_20260806/tests
```

The input constructor requires the same external 2021 10% ROOT prerequisite
and exact SHA256 recorded by the parent pseudo65 bundle.  Seeds, ROOT keys,
array hashes, generated scan-card hashes, and the fixed-GP reconstruction
state are recorded under `derived/`.

Each full draw has one 12-restart scan attempt over 50--250 MeV.  Every
55--75 MeV row also has an unchanged-card second attempt, followed where
needed by targeted unchanged-card repeats until the selected maximum-finite-
LML state is reproduced.  All 420 central rows close this gate with no
selected kernel-bound or interpolated rows; 17 retain documented multi-branch
histories.  Rows outside 55--75 MeV remain single-attempt descriptive results,
so full-grid optimizer-repeat stability is not established.
