# Continuing the v4.9.15 global-significance study

This extension preserves the original 2015 study and applies the same
likelihood statistics to full 2016 (39–180 MeV, 142 masses) and native 2021
10% (50–250 MeV, 201 masses). Each grid has 1 MeV spacing. Each complete toy
uses identical spectrum counts at every mass. The generating spectrum is
the dataset's archived common stress mean, not a certified physical truth.

## Reproduce or resume this extension

Run sequentially from the repository root, one numerical process at a time:

```bash
nice -n 10 python3 -B study_results/v4p9p15_global_2016_2021_20260906/run_global_accelerated.py --dataset 2016
python3 -B study_results/v4p9p15_global_2016_2021_20260906/analyze_global.py --dataset 2016
```

The numerical runner evaluates all three ensembles for each mass, reusing the
per-coordinate accuracy gate. Accepted products are in `global_fast/`.
`global/` preserves the exact ten-scan pilots and partial exact 2016 validation;
they are paired references, not additional independent toys. Consult the
acceleration protocol and full-response gates before changing the backend.
The original `run_global.py --dataset 2016 --ensemble validation1000` resumes
the paused exact reference, if a full exact replay is wanted. It is expensive.

Replace `2016` with `2021` for the second dataset. Source/count contracts must
match to resume. Check every phase's `summary.json`, its per-mass `_qa.json`
files, and any `_FAILURE.json`; do not remove failed spectra or hypotheses.
Preserve a completed version and its manifest before any reanalysis.

Both numerical runners use deterministic seeds keyed by `v4p9p14-global`, dataset and
ensemble. **Changing only `--output` does not generate independent toys.**
The existing ten pilot scans and 1,000 validation scans are separate samples.
Keep the pilot out of the validation ensemble. Do not pool repeated files or
join independently generated mass-local toys.

## Additional independent direct scans

Create a new study directory, copy the accelerated runner there, and declare a new
ensemble name, for example `validation10000_v2`, with a count of 10,000.
Use that new name in the seed coordinates and output path; record the new
script/protocol hashes before generating any spectra. Update the analyzer's
validation input and sample-size labels together. Keep the existing 1,000
scans frozen as a separate comparison, or explicitly record the disjoint
toy IDs and union rule if pooling is scientifically justified.

Choose the sample size from the probability and precision to be tested.
If the true tail probability is p, n direct scans give about n*p exceedances;
approximately 100 exceedances give about 10% relative counting uncertainty
when p is small. Zero exceedances in n scans imply only the one-sided 95%
upper bound `1 - 0.05**(1/n)`. Thus 0/1,000 bounds p below approximately
0.00299 under the declared generating model; it cannot certify a much
smaller GP extrapolation. For importance sampling of a complete scan, verify every retained toy
against one common full-spectrum target and its actual proposal mixture,
then rescan that complete spectrum at all masses. Stored full-spectrum
stress toys may be reusable if their counts and proposal provenance survive;
recalculate or verify their spectrum-level weights and assess effective
sample size for the global maximum. Pointwise tail estimates and pointwise
effective sample sizes cannot be carried over to a global tail.

## More GP draws

More GP draws are inexpensive and reduce sampling error **within the assumed
Gaussian field**. They do not improve background-model validity or replace
direct tail checks. Copy the completed analysis to a new derivative before
increasing `--gp-samples`, for example to 2,000,000. The default field RNG
extends the same stream, so the larger sample is nested, not independent.
Report zero tails with limits, and keep Monte Carlo uncertainty separate
from approximation error and physical-model uncertainty.

## The 2016 scientific priority

Use predeclared predictive controls to assess the 2016 generating shape,
including its 75–85 MeV transition, and independently justified smooth
alternatives. Its source-development subset is not established as disjoint
from the observed sample; the archived continuation also retains the
restricted fit-status waiver. Small conditional p-values can reject this
particular construction without identifying a particle. A large raw-maximum
p-value is not a goodness-of-fit test.

Compare fixed and reestimated kernel policies on identical complete spectra,
with a declared selection rule. Resolve the separate inherited 2016
numerical exception before promoting a final result. Preserve the existing
failures and do not discard a background based on a favorable observed
p-value or upper limit. A sensitivity claim needs a separately declared
signal-injection or expected-limit study with adequate calibration.

## Finer grids and joint searches

The two 2 MeV subgrids only expose finite-grid dependence. A 0.5 or 0.25 MeV
extension needs an explicit kernel-state policy at intermediate masses,
fresh observed fits and the same rule in every full-spectrum toy. Do not
interpolate plotted p-values and call that a validated finer search.

A combined search needs one complete spectrum per constituent dataset,
coherent joint generating scenarios, the shared-coupling likelihood at each
mass, and all relevant nuisance correlations. Calibrate the same declared
decision rule on observed and toy scans. Neither the minimum common-truth
local p-value nor the raw maximum in this study calibrates the v4.9.13
two-truth envelope. Its mass-dependent local GP truths cannot be spliced
into one global experiment. State the joint family and any mixed-scenario
treatment before viewing the combined result.

## Report and quality checks

```bash
python3 -B study_results/v4p9p15_global_2016_2021_20260906/build_report.py
pdftoppm -scale-to 1200 -png output/pdf/v4p9p15_global_2016_2021_20260906/HPS_GPR_v4p9p15_Global_Study_2016_Full_2021_10pct.pdf study_results/v4p9p15_global_2016_2021_20260906/qa/pages/page
python3 -B study_results/v4p9p15_global_2016_2021_20260906/validate_products.py
```

Inspect every rendered page after rebuilding and regenerate the final
manifest only when numerical, semantic and visual checks are complete.
Numerical/product QA and scientific qualifications are reported separately.
