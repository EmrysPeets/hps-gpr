# v4.9.8 selected-results scope

## Purpose

This release assembles a compact selected-results section for the Harvard writing
sample.  It is a presentation release, not a new statistical analysis.  It adds
no expected-limit bands and makes no scan-global significance claim.

The requested all-three shorthand is resolved explicitly as **full 2015 + full
2016 + 2021 10%**.  The all-three curve is shown only on the common 50--90 MeV
mass grid where all three campaigns contribute.

## Included result states

| Curve | State used here | Mass range |
|---|---|---:|
| 2015 full | reviewed historical v4.2 | 19--90 MeV |
| 2016 10% | reviewed v4.1 standalone | 39--180 MeV |
| 2016 full | reviewed historical v4.2 | 39--180 MeV |
| 2021 1% | reviewed support-40 standalone | 50--250 MeV |
| 2021 10% | current v4.9.5 support-36 standalone | 50--250 MeV |
| all pairwise combinations | reviewed historical v4.2 | campaign overlap |
| all-three combination | reviewed historical v4.2 | 50--90 MeV |

The historical v4.2 combinations retain their original 2021 support of
40--300 MeV.  The newer support-36 2021 result is shown only as an individual
curve and is not substituted into the historical combinations.

## Full-2016 support boundary

The immediately preceding v4.9.7 study already implemented the statistically
correct form of scaling the 2016 10% source to full statistics: it fitted a smooth
source-conditioned mean, normalized that mean once to the full-data count, and
drew independent Poisson fluctuations at full scale.  It did not multiply realized
toy counts by ten.  No candidate support passed the predeclared qualification rule,
so v4.9.7 produced neither a selected support nor a new full-2016 observed curve.

This release therefore retains the reviewed historical v4.2 full-2016 and
combination results.  A new full-2016 result would require a separately frozen,
independently justified low-threshold generating mean and a successful support
qualification before observed data are opened.

## Interpretation boundary

- Every upper limit is an observed asymptotic 90% CLs value.
- Every p-value is a fixed-mass local asymptotic p0 value.
- No expected bands, scan-wide pseudoexperiment calibration, global significance,
  direct coverage statement, exclusion claim, or sensitivity claim is added.
- The 2021 1% points at 50--52 MeV are retained only as support-edge diagnostics
  and are excluded from candidate interpretation.
- The extracted 65 MeV display is the largest historical all-three local
  fluctuation.  It is a candidate diagnostic, not evidence for a physical signal.

