# 2021 exposure and length-scale ensemble pilot

This directory contains a resumable, review-facing pilot for testing whether
the resolution-scaled GP length-scale upper bound should change with 2021
statistics. It creates no expected limit bands.

## Completed review artifacts

- `FINAL_LENGTH_SCALE_BOUND_INTERPRETATION.md` and
  `derived/final_length_scale_bound_interpretation.json` are the deterministic
  human- and machine-readable decision records.
- `../../hps_gpr_analysis_note/HPS_GPR_Analysis_Note_v4p1_20260804.pdf` is
  the compiled v4.1 analysis note containing the optimized-length-scale
  interpretation, observed dataset comparisons, ten-toy exposure projections,
  fixed-amplitude response study, and combined observed/asymptotic 2016 update.
- `NOTE_BUILD_QA.md` records the patch, build, reference, and rendered-page
  checks for that PDF.
- `PUBLICATION_MANIFEST.md` records the curated GitHub scope and the excluded
  regenerable raw task directories.

The ten-toy projection selects factor 20 as the smallest nonbinding
projected-2021-100% ceiling. Factor 25 is the first universal tested ceiling
because the diagnostic native-1% lanes still contact factor 20. The
factor-20-to-25 likelihood and fitted-uncertainty plateau shows no material
sensitivity benefit from the larger value. The common roughly 2--3% paired
signal under-response persists at factors 15, 20, and 25 and remains a
finite-ensemble closure diagnostic rather than a calibrated bias or coverage
claim.

## Frozen inputs

The fit implementation is imported only from the pinned checkout
`/Users/emryspeets/Desktop/gp_mods/hps-gpr-analysis-note-v3-20260709` at
commit `df4d4562060b4cdde46994cc58128d580d289adb`. The driver verifies that
commit, requires a clean tracked `hps_gpr/` tree there, verifies every
fit-module SHA-256 in `study_spec.json`, and verifies the exact reviewed 2021
10% k15 observed-only card
`config_obsUL90_2021_10pct_fit040_300_k15_observed_only.yaml` (SHA-256
`3a4120ab520cca3352d281e06d4d0c5e4c05c83cec97c319b1aabe19e9c0b3f2`)
before any fit.

The exact-support analytic seeds retain the native 8000-bin, 0--1 GeV
histogram geometry, are normalized on 40--300 MeV, and are scanned only on
50--250 MeV. The primary comparison uses `fGenGammaThresh` for both the 1%
and 10% sources so that native 10% is not compared to a different analytic
family. `fSigPowExpQ` is a separately labeled alternate-truth lane.

Both functions have `selection_pass=true` in the generated metadata, but
their ROOT `fit_ok` flags are false. For `fGenGammaThresh`, the recorded
Pearson chi2/ndf values are 1.108 for the 1% source and 2.781 for the 10%
source. These are smooth stress truths, not claims that either function is
the physical background generator.

## Paired Poisson construction

For every analytic truth, native source family, bin, and toy index, the
driver makes independent increments:

```text
N_1pct       ~ Pois(lambda_1pct)
N_1pct_x10    = N_1pct + Pois(9 lambda_1pct)
N_1pct_x100   = N_1pct_x10 + Pois(90 lambda_1pct)

N_10pct      ~ Pois(lambda_10pct)
N_10pct_x10   = N_10pct + Pois(9 lambda_10pct)
```

Thus every marginal is Poisson with the requested mean and paired
comparisons share their lower-exposure realization. A realized histogram is
never multiplied by 10 or 100. The native 10% source is kept separate from
1%-to-10% scaling; their observed support-normalization ratio is about
11.296, not exactly ten.

`paired_exposure_toy_manifest.json` records the source hashes, seed words,
increment hashes, full-count hashes, totals, parent relationships, analytic
truth, scenario, and toy index. `validate-toys` checks the nesting algebra
and every stored count hash.

## Predeclared scan

The candidate upper factors are 6, 9, 12, 15, 20, and 25. All other
scientific settings come from the reviewed 2021 10% k15 observed-only card.
The default pilot grid has eleven masses:

```text
50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250 MeV
```

Each mass is run separately with a stable optimizer seed that depends on
truth, scenario, background-toy index, and mass, but not on the upper-bound
factor. This makes candidate comparisons reproducible and lets a restarted
task reuse completed mass parts. Use `--mass-step-mev 1` only for the later
full 201-point scan.

The scan records the complete `results_single.csv` schema, including
`ls_lo`, `ls_hi`, `ls_opt`, `sigma_x`, boundary flags, constant parameters,
LML, training counts, extraction status, asymptotic local p-value/Z, and
observed-equivalent upper-limit diagnostics. Study provenance and seeds are
appended to every row. Because the immutable `df4d456` `io.py` predates the
v4 uncropped fractional-bin density implementation, this driver also labels
its epsilon-squared conversion as
`immutable_df4d456_rebinned_whole_bin` and sets
`eps2_up_promotable=false`. Length-scale, LML, count-amplitude, and local
p-value diagnostics remain the purpose of this pilot; its epsilon-squared
column must not be promoted as the exact v4 conversion.

The ten-toy ensemble is a screening pilot. Even zero boundary hits in ten
toys gives a one-sided 95% upper bound of about 25.9% on the true hit rate.
It is not a coverage qualification and its per-toy limits must not be
presented as expected bands.

## Fixed-amplitude injection protocol

Injection comparisons use
`factor15_prefit_asimov_absolute_v1`. For every analytic truth, exposure
scenario, background-toy index, and targeted mass (60, 120, or 220 MeV), a
factor-15 background-only fit defines one prefit-Asimov
`sigmaA_ref`. The ledger converts the predeclared 0, 1, 3, and 5 anchor-sigma
tags to absolute count amplitudes once:

```text
A_anchor = anchor_nsigma * sigmaA_ref(factor 15, prefit Asimov)
```

Those exact absolute amplitudes are reused for factors 6, 9, 12, 15, 20,
and 25. The candidate factor's own `sigmaA_ref`, fitted `sigma_A`, pull, and
effective `inj_nsigma = A_anchor / sigmaA_ref(candidate)` remain in the raw
row. This separates a sensitivity change from the signal size and prevents
each candidate from silently receiving a different injected signal.

The shared candidate YAML files and `task_manifest.jsonl` are frozen and are
not regenerated. They retain their original `sigmaA` declaration for scan
provenance. Their six SHA-256 values are pinned in `study_spec.json` and are
rechecked by preflight and task execution. Injection execution changes the
loaded config to `absolute` only in memory, after validating the versioned
factor-15 ledger. Every injection row records:

```text
injection_protocol
injection_anchor_factor
injection_anchor_nsigma
injection_anchor_strength
injection_anchor_sigmaA_ref
injection_anchor_ledger_sha256
```

Anchor preparation also predicts the exact full signal-count vector using
the pinned injection code and stable point seed. During each candidate run,
the driver transparently captures the actual vector returned by that same
pinned helper and fails closed unless its SHA-256, `Nsig_win`, and
`Nsig_train` match the factor-15 reference. The output retains the realized
`signal_draw_sha256` and the boolean verification fields. Thus common random
numbers are checked from the realization, not inferred only from equal seed
labels. Injection collection repeats the check across every available
candidate factor in each matched truth/scenario/background-toy/mass/strength
group.

## Commands

Read-only preflight:

```bash
python3 study_results/v4p1_2021_ls_exposure_ensembles_20260804/run_ensemble.py preflight
```

Prepare deterministic paired toys, six configs, and 600 scan plus 600
injection tasks:

```bash
python3 study_results/v4p1_2021_ls_exposure_ensembles_20260804/run_ensemble.py prepare
```

Validate the paired ROOT file:

```bash
python3 study_results/v4p1_2021_ls_exposure_ensembles_20260804/run_ensemble.py validate-toys
```

Plan factor-15 anchor preparation without starting a fit:

```bash
python3 study_results/v4p1_2021_ls_exposure_ensembles_20260804/run_ensemble.py \
  prepare-injection-anchors
```

After review, prepare all 100 resumable truth/scenario/background-toy anchor
parts with bounded fresh processes and consolidate the deterministic
300-entry ledger:

```bash
python3 study_results/v4p1_2021_ls_exposure_ensembles_20260804/run_ensemble.py \
  prepare-injection-anchors --workers 8 --max-parts 100 --execute
```

Interrupted preparation is resumed by the same command. It skips validated
parts and writes
`derived/injection_anchors/factor15_prefit_asimov_absolute_v1.json` only
when all parts are present. `--force` recomputes selected valid parts; use it
only after review.

Show status for the primary-truth scan:

```bash
python3 study_results/v4p1_2021_ls_exposure_ensembles_20260804/run_ensemble.py status \
  --kind scan --truth gengamma
```

Dry-run one task, then execute it explicitly:

```bash
python3 study_results/v4p1_2021_ls_exposure_ensembles_20260804/run_ensemble.py \
  run-task scan__f15__gengamma__2021_1pct_x10__t0000

python3 study_results/v4p1_2021_ls_exposure_ensembles_20260804/run_ensemble.py \
  run-task scan__f15__gengamma__2021_1pct_x10__t0000 --execute
```

Run one pending primary task as a bounded smoke:

```bash
python3 study_results/v4p1_2021_ls_exposure_ensembles_20260804/run_ensemble.py \
  run-pending --kind scan --truth gengamma --max-tasks 1 --execute
```

After the single-task smoke is reviewed, launch a bounded batch with eight
fresh subprocesses. Each process still pins numerical libraries to one
thread:

```bash
python3 study_results/v4p1_2021_ls_exposure_ensembles_20260804/run_ensemble.py \
  run-pending --kind scan --truth gengamma --max-tasks 300 --workers 8 --execute
```

Run the later full 1 MeV grid for a reviewed task:

```bash
python3 study_results/v4p1_2021_ls_exposure_ensembles_20260804/run_ensemble.py \
  run-task scan__f15__gengamma__2021_1pct_x10__t0000 \
  --mass-step-mev 1 --execute
```

After the anchor ledger is complete and reviewed, targeted injection closure
uses 60, 120, and 220 MeV; factor-15-anchored absolute amplitudes labeled
0, 1, 3, and 5 anchor sigma; Poisson signal injection; the fixed paired
background histogram; and a fully optimized 12-restart GP refit. Omitting
`--execute` remains a no-fit dry run:

```bash
python3 study_results/v4p1_2021_ls_exposure_ensembles_20260804/run_ensemble.py \
  run-task injection__f15__gengamma__2021_1pct_x10__t0000

python3 study_results/v4p1_2021_ls_exposure_ensembles_20260804/run_ensemble.py \
  run-task injection__f15__gengamma__2021_1pct_x10__t0000 --execute
```

Collection fails closed until every selected task is complete. For example,
collect the 300-task primary-truth lane independently:

```bash
python3 study_results/v4p1_2021_ls_exposure_ensembles_20260804/run_ensemble.py \
  collect --kind scan --truth gengamma
```

Use `--allow-partial` only for explicitly labeled interim diagnostics. It
writes filenames containing `partial`.

Run lightweight tests:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s study_results/v4p1_2021_ls_exposure_ensembles_20260804/tests -v
```

## Caveats and decision boundary

- The primary functional-form route preserves native binning and applies the
  production rebin-5 scan path. The older GP-toy helper instead scans an
  already-rebinned toy with forced rebin 1 and is not used here.
- Sequential mass fits and one-thread numerical libraries are execution
  controls for reproducibility; the scientific card remains the reviewed
  2021 10% k15 observed-only card except for the requested 2021 upper factor.
- Common optimizer seeds improve paired comparison but do not guarantee
  bitwise equality across different BLAS, sklearn, or operating-system
  versions. Commit, module hashes, config hashes, and per-row seeds are
  retained.
- The pinned immutable checkout's `io.py` computes density from the rebinned
  fit histogram with whole-bin inclusion. It does not contain the later v4
  uncropped native-histogram fractional-overlap conversion. The driver marks
  every resulting epsilon-squared row non-promotable; a final v4 epsilon
  comparison requires a separately pinned clean checkout containing that
  implementation.
- A larger allowed parameter space cannot have a worse true maximum LML.
  Negative adjacent-factor Delta-LML beyond numerical tolerance indicates an
  optimizer-branch failure and requires an unchanged-card rerun, never
  interpolation.
- Bound selection must use predeclared boundary occupancy, LML stability,
  unchanged-card optimizer stability, and injection closure. Do not choose a
  bound because it gives a favorable observed limit or p-value.
- The fixed count-amplitude protocol enables direct extraction-bias and
  sensitivity comparisons among length-scale candidates. It is not a fixed
  physical epsilon-squared injection: a physical sensitivity claim still
  requires a separate scenario-specific signal-yield ledger and the
  promotable uncropped density conversion.
- This package deliberately makes no expected limit bands and no direct CLs
  coverage claim.
