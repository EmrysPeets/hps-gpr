# Combined 74 MeV scalar-reference initialization recovery

This derivative handles the recorded failure in
`chunked_resource8_v1/attempt2_combined/all_2015_2016_2021/m074/FAILURE.txt`.
Its SHA256 is `1c74b4f4797cb6c21598c15709ff240fcc10106bac5f03191b83b66b4151a1a5`;
the numerical failure ledger is
`e17a1eb730c52a48961200fbf8e12039b92cb8b949bb25258eea9fb2d47239af`.
The old source, contracts, successful numerical audits, and failed outputs remain
unchanged. This is an optimizer initialization recovery, with no new truth,
likelihood, limit definition, sampling choice, or validation ensemble.

The failed scalar reference is the zero-nuisance, free-amplitude fixed-background
fit at stress-bank row 9728: proposal 9 (unshifted, 1.5 reference sigmas), toy 512
within its 1,024-toy draw. Chunk 9728:9856 has SHA256
`941b988f24d9d8553c47b039b6c2124ac1c5266062f3caef938f102efaefc397`.
Both profiled scalar checks and both production batch models had completed.
The original scalar solver exhausted 101 Newton iterations with score
`4.4949521033066375e-7`, above its unchanged `2e-7` threshold.

## Permitted recovery and failure handling

`scalar_reference_recovery.py` always calls the original `Profile.fit` first.
Only a linear, zero-nuisance, free-amplitude call with no supplied initial state,
raising the original finite `Unconverged fit, score=...` error above the score
threshold, may receive a different initializer. Every other failure is fatal.

The helper brackets the original scaled score over strictly positive Poisson
expectations and checks positive curvature at every evaluation. It uses Brent's
method with absolute tolerance `5e-15`, relative tolerance four machine epsilons,
and at most 128 iterations. The root itself must satisfy `abs(score)<2e-7`.
It is then passed as `initial` to the unchanged original `Profile.fit`.
The original returned score and positivity checks remain compulsory; neither
the objective nor its acceptance threshold is substituted. Bracketing or restart
failure halts the run and retains the original and recovery exceptions.

The Context is an ordinary subclass passed through a new core namespace to the
existing chunked runtime. A source AST comparison proves that its copied
`make_models` differs solely at the scalar-reference constructor. That comparison
never generates or executes rewritten code. `BatchProfile`, all actual fitted
statistics, GP prediction functions, chunk partitions, and q caches remain the
original implementations. No original module globals are modified. Every
reference still passes signed-root agreement `2e-5` and q agreement `1e-4`.

Each fallback records the full and window count hashes, background/template
hashes, original exception, coordinates, bracket evaluations, root, unchanged
fit return, and the corresponding signed-root/q agreement row. The known failure
must occur and be recovered exactly once. Failure to reproduce it is reported,
not silently accepted. Full spectrum arrays are regenerated rather than published.

## Replay before production

Run source/pure checks first:

```
python3 -B scalar_reference_recovery.py --pure-check
python3 -B diagnose_scalar_reference_recovery.py
```

After review and a free CPU slot, explicitly authorize the small replay with
`diagnose_scalar_reference_recovery.py --run`. It makes exactly the first ten
original stress-proposal RNG calls, each shaped `(1024,1626)`, verifies all ten
saved per-proposal hashes, and retains the known 128-row chunk. Its window hash
must match the failed run. It verifies the same proposal arrays and accepted GP
backend before reproducing the scalar failure and restart. JSON ledgers retain
all evidence; these are reused spectra, not additional validation observations.
An existing output directory, including a failure, requires a new output path.

## One-coordinate full replay

Only `all_2015_2016_2021`, 74 MeV, attempt 2 is permitted. The driver requires an
explicit completed diagnostic directory. It copies the failed attempt's numeric
plan; only execution/selection identity changes. Both science proposal arrays and
both complete bank hashes must equal the failed ledger before any bank fitting.
The accepted backend must stay `eigenfeature_rtol_1e-15`, with nuisance cutoff
`1e-5`; drift fails closed. The GP bank retains 33 proposals x 256 spectra and
the stress bank 102 proposals x 1,024: 112,896 total spectra. No seeds, proposal
nodes, strength grid, counts, fallback tolerances, or spectrum ordering change.

The original 18 numerical-audit draws retain their original seeds. Extended
numerical-audit draws use the fresh proposal-plan identity through the unchanged
runner, providing additional numerical verification. They are separate from the
unchanged science banks and 500-spectrum validation cells and are never pooled.
The original validation function and its `(validation, scope, mass, truth,
strength)` seed namespaces are retained. All original numerical, memory, MC,
bounded-statistic and 90% CLs gates still apply.

The inherited resource policy remains version 2: chunk size 128, one BLAS thread,
at most one 8 GiB worker plus at most one 4 GiB companion, aggregate at most
12 GiB. The coordinator must check fresh memory pressure before each launch.
The source-derived memory estimate and 512 MiB reserve remain unchanged.

## Supplemental audit interface

`run_scalar_reference_recovery.layout_marker(8)` preserves the inherited four-file
`source_hashes` map and adds `scalar_reference_recovery`, containing its own
type/version, four additional source hashes, failure identities, narrow scope,
unchanged gates, and backend policy. All new sources, pure QA, selected diagnostic
files, and original failed artifacts are frozen in `sampling_hashes`; original
47 inference hashes remain identical. `verify_recovery_layout(layout, data=None)`
returns consumed source/QA paths, and with result data additionally verifies the
fallback ledger and original bank/seed identities.

Final results add `scalar_reference_recovery` with the ledger path/SHA, explicit
diagnostic directory/SHA, original failed contract, whole-bank hashes and
validation seed namespaces. The separate `scalar_reference_recovery.json` contains
all fallback records. The supplemental auditor must strip the nested recovery
marker to validate the unchanged resource8 layout, then call the recovery helper
and retain all existing point, chunk, density and scalar gates. Publication must
include the returned dependencies and ledgers. A failed final audit retains an
`unverified_result.json` and FAILURE file; it does not leave a completed checkpoint.
