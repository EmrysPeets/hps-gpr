# v4.8 2021 functional-form qualification

## Outcome

No common analytic source generator qualified in the nominal qualification
branch.  A later, explicitly user-directed conditional branch nevertheless
freezes a sparse threshold stress mean fitted to native 1% and transfers its
shape to native 10% by normalization only.  The authoritative v4p8p2 closure
and locked length scan analyze toys 0--19 only.  Indices 20--22 had already
been inspected in a superseded one-lane development run, so no unopened-reserve
or reserve-pooling claim is made; indices 23--24 were not consumed by the
authoritative products.

- `fSigPowExpQ` is rejected by source fit quality and bound contact.
- Literal five-parameter `fSigPow` and the archived simple-family pool are also
  rejected by the implementation audit.
- Positive log-Chebyshev degrees 18, 20, and 24 pass only a broad in-sample
  engineering screen.  No degree passes predictive qualification.
- Degree 18 has blocked-validation deviance/bin 18.107 and 24.045 for the 1%
  and native-10% sources, with maximum fake-gap model shifts 2.679 and 9.889
  diagonal-Poisson standard deviations.  Gates were 1.25 and 0.20.
- The reconnaissance fitter also lacks the required multistart/stationarity and
  bin-integration certificate.  `build_stress_toys.py` therefore refuses to
  write toys unless a separately reviewed override and repaired optimizer gate
  are present.
- Frozen v4.2 settings, 40--300 MeV support, factor-15 ceiling, and 90% CLs
  (`cls_alpha=0.10`) remain unchanged.

The conditional sparse branch uses a thresholded generalized-gamma core with
only `T2` and `T6` broad corrections.  Its native-bin Pearson/deviance ratios
over the primary 50--250 MeV search region are 1.088/1.088 for native 1% and
2.676/2.676 for the shape-frozen native-10% application.  The 40--50 and
250--300 MeV regions are positive GP-training shoulders, not primary fidelity
targets.  This is an engineering stress-generator criterion, not formal source
goodness-of-fit or physical truth.

The completed extraction screen has 1,599 accepted of 1,600 raw states.  The
single exclusion is native-10% x10, toy 12, 65 MeV, five-sigma injection, where
the top optimizer branch did not reproduce after five attempts.  Conditional
closure is not uniform: the background-only mean pulls at 65 MeV are about
-1.31 for both 1% x10 and native 10%, so this branch is not promoted beyond
its declared conditional-stress scope and no production-card or kernel-ceiling
change follows from it.

## Reproduce the fail-closed reconnaissance diagnostics

```bash
python3 fit_qualify.py run
python3 make_qualification_figures.py all
```

The deliberately blocked builder demonstrates the promotion gate:

```bash
python3 build_stress_toys.py build
```

It must terminate with a `BuildError` while
`optimizer_reproducibility_gate_passed=false` and no explicit conditional-stress
override is present.

## Reproduce the user-directed conditional branch

The cached ROOT product contains 25 backgrounds per source family, but the
frozen analysis partition is exactly indices 0--19:

```bash
python3 build_rigid_toys.py validate
python3 make_rigid_generator_figures.py all
python3 run_rigid_study.py preflight
python3 run_rigid_study.py run --toy-start 0 --toy-stop 20 --workers 2
python3 run_rigid_study.py collect
python3 run_rigid_study.py analytic-mean
python3 make_rigid_closure_figures.py
```

The separate background-only length diagnostic is externally locked and uses
the same 20 backgrounds.  It evaluates factors 15/20/25 at 50,70,...,250 MeV
without producing signal amplitudes, pulls, CLs, limits, or a factor choice:

```bash
python3 run_rigid_length_scan.py validate
python3 run_rigid_length_scan.py run --toy-start 0 --toy-stop 20 --workers 1
python3 run_rigid_length_scan.py collect
python3 make_rigid_length_figures.py
```

Run multiple length-scan processes only on disjoint `--scenario` selections.
The locked length runner refuses indices 20--24 and fails if its own output
directories for those indices are populated.  That is a scan-local scheduling
guard, not evidence that every earlier development workflow left those toys
uninspected.

The `heavy_scan_launched=false` field printed by the read-only `validate`
subcommand means that validation itself never launches fits.  The authoritative
completed-run status is `derived/rigid_length_scan/collection_summary.json`,
which records 80 current tasks and zero missing/stale tasks.

The completed optimizer-only collection selects 2,637 of 2,640 factor--mass
states.  Factor 15 is exact-bound in 599/878 selected states, factor 20 in
2/880, and factor 25 in 0/879.  The 20-to-25 comparison is a strong near-
plateau, with ten numerical-scale strict nested-LML reversals and no material
per-training-bin reversals.  This is a conditional optimizer diagnosis only:
the scan deliberately makes no production factor or card choice.

## Main artifacts

- `RELEASE_PROVENANCE_ERRATA.md`: post-run disclosure for the superseded
  inspection of toys 20--22, the stale build summary, pre-run QA snapshots,
  and the frozen absolute-path portability boundary.
- `GENERATOR_QUALIFICATION_PROTOCOL.md`: post-exploratory development protocol
  and one-factor study design; it is not a blind predeclaration.
- `ARCHIVED_SIMPLE_FAMILY_REJECTION.md`: exact 30/35/40/50-MeV ROOT-family
  implementation audit, including literal `fSigPow`.
- `derived/archived_root_family_edge_audit.csv`: machine-readable 64-row audit;
  the eight compact input metadata records are archived under `reference/`.
- `study_spec.json`: frozen source/card hashes, intended four-lane design, 90%
  CLs contract, and explicit not-run status for the nominal-qualification
  branch.
- `derived/generator_qualification.json`: machine-readable source-fit ledger.
- `derived/generator_qualification_summary.csv`: compact candidate/source table.
- `figures/v4p8_source_generator_qualification_failed.{png,pdf}`: visually
  reviewed failure diagnostic.
- `figures/v4p8_requested_toy_figures_blocked.{png,pdf}`: explicit record that
  Figure-46/48/136 production did not pass the nominal-qualification generator
  gate; it is not the status of the later conditional branch.
- `SECTION5_RECOMMENDED_CHANGES.md`: joint GPR/statistics/physics rewrite plan.
- `RIGID_CONDITIONAL_STUDY_RESULTS.md`: concise scientific disposition of the
  completed generator, closure, and optimizer-only length studies.
- `FINAL_ARTIFACT_SHA256.txt`: checksums for the main frozen inputs, collected
  summaries, reports, and visually reviewed PDFs.
- `rigid_generator_spec.json`: frozen sparse-generator formula, source-fit
  metrics, transfer rule, signal-absorption audit, and claim boundary.
- `rigid_study_spec.json` and `run_rigid_study.py`: hash-bound 20-toy closure
  contract and mature pull-blind optimizer-gated runner.
- `inputs/rigid_ggt26_scaled1pct_nested_toys_25.{root,manifest.json}`: cached
  nested product; only indices 0--19 are in this analysis partition.
- `derived/rigid_closure_v4p8p2_20toy_frozen/`: collected ledgers for 1,600 raw
  states, with 1,599 accepted and one explicit exclusion, plus 90% diagnostic
  intervals and deterministic analytic-mean closure.
- `figures/v4p8_rigid_source_fit_qualification.{png,pdf}` and
  `figures/v4p8_rigid_toy_generation_20.{png,pdf}`: source and Figure-46-style
  generator/sampling QA.
- `make_rigid_closure_figures.py`: Figure-48-style conditional spurious-signal
  diagnostic, mirrored under `output/pdf/v4p8_2021_rigid_threshold_truth_20260813/`.
- `run_rigid_length_scan.py`, `rigid_length_scan_core.py`, and
  `rigid_length_scan_lock.json`: externally locked, background-only
  factor-15/20/25 Figure-136 diagnostic; it cannot consume reserve toys or
  compute pulls, amplitudes, CLs, limits, or choose a card.
- `derived/rigid_length_scan/`: hash-validated 80-task optimizer collection
  with 2,637 selected states, three reproducibility exclusions, occupancy, and
  paired nested-LML ledgers.
- `output/pdf/v4p8_2021_rigid_threshold_truth_20260813/`: visually reviewed
  source, Figure-46-style sampling, Figure-48-style zero-signal closure,
  Figure-136-style factor trajectories, and pull-blind optimizer companion
  PDFs/PNGs.
- `quarantine/rejected_fsigpowexpq_prototype/`: nonconforming initial prototype,
  retained only for provenance and excluded from all scientific claims.
- `quarantine/superseded_rigid_closure_development_20260813/`: superseded
  one-lane closure attempts, including prior inspection of toys 20--22; excluded
  from the authoritative v4p8p2 result.
- `quarantine/stale_rigid_toy_build_summary_20260813/`: build summary tied to a
  superseded manifest serialization; the current ROOT and manifest hashes are
  frozen directly in `FINAL_ARTIFACT_SHA256.txt`.
- `qa/rigid_length_scan/`: frozen preflight and scheduling records.  Its task
  manifest is a pre-run snapshot; final task status lives in
  `derived/rigid_length_scan/task_product_audit.csv` and the collection summary.

## Claim boundary

The source-fit and extraction ledgers are reconstructed-spectrum conditional
diagnostics.  They are not a physical background measurement, unconditional
estimator bias, coverage, expected bands, a limit, or evidence for a
kernel-bound/card change.  The sparse ROOT toy product and extraction outputs
are accepted only as a versioned conditional stress screen; they do not rescue
the rejected nominal-generator branch or enter production inference.
