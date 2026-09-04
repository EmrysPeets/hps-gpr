# v4.9.7 combined observed limit and 100-toy band scaffold

This directory contains the fail-closed continuation of the v4.2 combined
workflow. It computes the shared nonnegative epsilon-squared observed upper
limit for **2015 full + 2016 full + 2021 10%**, plus 100 conditional
background-only expected-limit toys per mass.

Production is intentionally blocked until a 2016 support edge has passed the
frozen protocol, the decision file has status support_edge_frozen and
observed_scan_authorized true, and the 142-row 2016 observed-state review is
complete. No script infers the selected edge from filenames, scan rankings, or
the data.

## Frozen numerical lineage

The following files started from the origin/main v4.2 campaign-local
implementation:

| v4.2 source | frozen source SHA-256 | v4.9.7 treatment |
|---|---|---|
| cached_profile_solver.py | fb9c02e2a3a8fc6240d15357945e2a5c73c5859492f23ef17012f98c022e316e | copied byte-for-byte |
| run_combined_bands_cached_fixed_reviewed.py | 513b3440d343a9b963761fe6ac69057975cd09a4ca299ae24a469d5d6da22688 | adapted to 100 toys and the v4.9.7 gates |
| benchmark_cached_profile_closure.py | 164aff565d0b1de212dd191fd69207eca4da401ae468953b671c0c774de0ab85 | adapted to the same gates |

The cached solver is only a memoized form of the same profiled Gaussian
likelihood and asymptotic CLs bisection. A fresh direct-versus-cached bitwise
closure report covering one-, two-, and three-dataset active sets is mandatory
for production.

The workflow does not rely on whichever hps_gpr package happens to appear
first on the caller's Python path. runtime_combined/hps_gpr is a complete
20-module package copied from clean origin/main commit
e2c930f3f879742b2846e3fca1ee1b7e8d99ecc6. runtime_guard.py verifies the
SHA-pinned manifest and every module before import, requires the package
__init__.py, and then checks the actual file origin of every numerically
relevant imported module. Both the closure and production provenance record
the manifest and resolved module origins.

The archived reviewed-state sources are SHA-pinned:

- 2015 rows are filtered from the exact v4.1 415-row ledger
  (a962c01a...9870aea9), giving 72 states at 19--90 MeV.
- 2021 rows are the exact v4.9.5 repaired observed ledger
  (e1b568a8...109de447), giving 201 states at 50--250 MeV and using
  data_range_2021 [0.036, 0.300].
- Only the new 142-row 2016 reviewed ledger at 39--180 MeV is supplied by this
  campaign. Its SHA-256 is recorded rather than predeclared.

The assembled state ledger must contain exactly 415 noninterpolated rows. Every
row retains the source path, source SHA-256, source-row index, source role, and
the selected 2016 support coordinates.

Byte-identical copies of both archived state sources live at
inputs/archived_2015_source_ledger.csv and
inputs/archived_2021_source_ledger.csv; these are the assembler defaults.

The bundled observed ROOT inputs are also SHA-pinned: source_2015_full.root,
source_2016_full.root, and source_2021_10pct.root. The card builder uses these
campaign-local copies by default, so production does not depend on mutable
external paths.

## Exact production contract

- Union mass grid: 19--250 MeV inclusive in 1 MeV steps (232 masses).
- Active-set partition: 20 rows of 2015; 11 of 2015+2016; 41 of all three;
  90 of 2016+2021; and 70 of 2021.
- Outer ensemble: exactly 100 finite mass-local limits at every mass, hence
  23,200 finite toy limits.
- All 23,200 epsilon-squared limits are retained in ordered per-mass JSON
  arrays with byte-level SHA-256 values; validation recomputes every band
  quantile, mean, and observed-rank tail count from those stored limits.
- Master seed: 24680. Child index is mass_MeV minus 19.
- Within each mass, active datasets retain the order 2015, 2016, 2021.
- Toy construction: draw the reviewed fixed-GP latent rate from its posterior
  covariance with the frozen nonnegative policy, then draw Poisson counts.
- Inner upper limit: 90% asymptotic tilde-q-mu CLs.
- The GP is not refit in any band toy.
- combined_mode count_scale is an exact numerical reparameterization of one
  shared nonnegative epsilon-squared coordinate. It is not independent
  signal-count parameters by run.
- Cross-run background covariance is block diagonal; this workflow does not
  add or profile correlated systematic nuisance parameters.

A 100-toy run with seed 24680 is a new deterministic ensemble. It must not be
described as the first 100 toys of the earlier 300-toy result: changing the
array draw length changes RNG advancement before later active datasets are
drawn. Toys are paired by index only within a mass and are not coherent scans
across mass.

## Required future 2016 ledger

assemble_reviewed_state_ledger.py requires these fields on every 2016 row:

    dataset,mass_GeV,const_opt,ls_opt,lml,interpolated,branch_multiplicity,
    selected_source,row_source,review_status,selected_support_low_MeV,
    support_high_MeV

There must be exactly one finite row at every integer mass from 39 through
180 MeV, all interpolated values must be false, and the support columns must
equal the freeze decision and CLI. This preserves the unchanged-card repeat
and branch-selection evidence rather than silently filling a missing state.

## Remaining invocation

Run from the repository root after replacing only the three angle-bracket
values with the completed 2016 freeze and review products:

    CAMPAIGN=study_results/v4p9p7_2016_support_combined_100toy_20260902
    FREEZE=<support_freeze_decision.json>
    REVIEWED_2016=<2016_results_single_repaired.csv>
    LOW_MEV=<selected_support_low_MeV>
    HIGH_MEV=210

Materialize the derivative card from the hash-checked frozen card and bundled
observed ROOT files:

    python3 "$CAMPAIGN/make_combined_card.py" \
      --support-freeze-json "$FREEZE" \
      --support-2016-low-mev "$LOW_MEV" \
      --support-2016-high-mev "$HIGH_MEV" \
      --analysis-output-dir "$CAMPAIGN/combined" \
      --config-out "$CAMPAIGN/combined/config_combined_100toy.yaml" \
      --provenance-out "$CAMPAIGN/combined/config_combined_100toy_provenance.json"

Assemble only the new 2016 rows with the exact archived 2015 and 2021 rows:

    python3 "$CAMPAIGN/assemble_reviewed_state_ledger.py" \
      --reviewed-2016-csv "$REVIEWED_2016" \
      --support-freeze-json "$FREEZE" \
      --support-2016-low-mev "$LOW_MEV" \
      --support-2016-high-mev "$HIGH_MEV" \
      --output-csv "$CAMPAIGN/combined/reviewed_gp_states_v4p9p7.csv" \
      --provenance-out "$CAMPAIGN/combined/reviewed_gp_states_v4p9p7_provenance.json"

Establish direct-versus-cache bitwise closure. The default masses 20, 40, 60,
100, and 200 MeV exercise both one-dataset regions, both two-dataset regions,
and the three-dataset region:

    python3 "$CAMPAIGN/benchmark_cached_profile_closure.py" \
      --config "$CAMPAIGN/combined/config_combined_100toy.yaml" \
      --config-provenance-json "$CAMPAIGN/combined/config_combined_100toy_provenance.json" \
      --reviewed-state-csv "$CAMPAIGN/combined/reviewed_gp_states_v4p9p7.csv" \
      --reviewed-state-provenance-json "$CAMPAIGN/combined/reviewed_gp_states_v4p9p7_provenance.json" \
      --support-freeze-json "$FREEZE" \
      --support-2016-low-mev "$LOW_MEV" \
      --support-2016-high-mev "$HIGH_MEV" \
      --toys-per-mass 20 \
      --json-out "$CAMPAIGN/qa/cached_profile_closure.json"

Run the full grid. Subsets and shards are deliberately rejected by the
production gate so a file cannot masquerade as the requested 232-mass result:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    python3 "$CAMPAIGN/run_combined_bands_cached_fixed_reviewed.py" \
      --config "$CAMPAIGN/combined/config_combined_100toy.yaml" \
      --config-provenance-json "$CAMPAIGN/combined/config_combined_100toy_provenance.json" \
      --reviewed-state-csv "$CAMPAIGN/combined/reviewed_gp_states_v4p9p7.csv" \
      --reviewed-state-provenance-json "$CAMPAIGN/combined/reviewed_gp_states_v4p9p7_provenance.json" \
      --closure-report "$CAMPAIGN/qa/cached_profile_closure.json" \
      --support-freeze-json "$FREEZE" \
      --support-2016-low-mev "$LOW_MEV" \
      --support-2016-high-mev "$HIGH_MEV" \
      --output-dir "$CAMPAIGN/combined/bands_100toy_cached" \
      --workers 8 \
      --confirm-production

Validate all hashes, aliases, quantiles, active sets, tail-count identities, and
the exact 232 x 100 counts:

    python3 "$CAMPAIGN/validate_combined_release.py" \
      --config "$CAMPAIGN/combined/config_combined_100toy.yaml" \
      --config-provenance-json "$CAMPAIGN/combined/config_combined_100toy_provenance.json" \
      --reviewed-state-csv "$CAMPAIGN/combined/reviewed_gp_states_v4p9p7.csv" \
      --reviewed-state-provenance-json "$CAMPAIGN/combined/reviewed_gp_states_v4p9p7_provenance.json" \
      --support-freeze-json "$FREEZE" \
      --support-2016-low-mev "$LOW_MEV" \
      --support-2016-high-mev "$HIGH_MEV" \
      --closure-report "$CAMPAIGN/qa/cached_profile_closure.json" \
      --bands-csv "$CAMPAIGN/combined/bands_100toy_cached/ul_bands_combined_all.csv" \
      --bands-provenance-json "$CAMPAIGN/combined/bands_100toy_cached/ul_bands_combined_all_provenance.json" \
      --report-out "$CAMPAIGN/qa/combined_release_validation.json"

The earlier v4.2 232 x 300 run took 339.253 seconds with eight loky workers and
one BLAS thread per worker. That is a scale reference, not a promised runtime
for this derivative.

## Claim and language boundary

Use “observed 90% CL asymptotic CLs upper limit on epsilon-squared” for the
observed curve and “100-toy conditional fixed-GP expected-limit quantiles” for
the band. Do not call this direct coverage, toy-calibrated CLs, a global
significance ensemble, or a calibrated sensitivity.

The output retains p0_analytic only as a mass-local diagnostic and labels its
scope explicitly. The empirical toy tail fractions rank the observed limit
inside the conditional expected-limit distribution; they are not discovery
p-values. With 100 toys their raw one-sided granularity is 0.01.

Any explanation of the earlier apparent excess changing after support
optimization must be phrased as a decomposition, not a disappearance claim:
compare the same observed histogram under the old and frozen supports; record
the changed training bins and fixed GP mean/covariance in the blinded window;
separate optimizer-branch changes from support changes; and report local
mass-by-mass diagnostics. Selection of the support edge must remain independent
of observed p0, fitted signal amplitude, and observed upper-limit strength.
