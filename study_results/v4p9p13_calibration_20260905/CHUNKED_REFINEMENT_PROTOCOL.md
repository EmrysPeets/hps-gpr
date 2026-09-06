# Chunked sampling execution, version 1

This is an execution derivative of `run_sampling_refinement.py`, governed by
the unchanged `REFINEMENT_PROTOCOL.md`. It changes allocation and batch layout,
not the statistical analysis, sampling policy or accepted numerical tolerances.
Its new driver is `run_chunked_refinement.py`. The six original inference files,
the original refinement driver/protocol and the original collector remain
unchanged. Source and pure tests precede any numerical execution and root review.

## Preserved analysis and selection

Keep the conditional 90% CLs target, exact bounded q, both methods, the two joint
generating scenarios, native spectra, full signal tails, shared epsilon-squared
normalization, fixed reviewed kernels, count-dependent alpha and per-spectrum GP
posterior retraining. This does not repeat hyperparameter optimization per toy.
It introduces no new truth, coverage claim, significance scaling or validation
selection. The combination still represents two joint scenarios only.

Import the original refinement eligibility, priority, Fisher mesh, proposal-law,
readiness, strict-candidate and extended-audit functions. Original grid, caps,
512/optional 1024 refined draws, 256 unrefined draws, seed namespaces and attempt
rules remain unchanged. The first pass must contain all 456 completed points.
`--scope` and `--masses` select a bounded subset of eligible coordinates.
`--skip-completed-input` accepts explicit sampling directories and skips every
completed coordinate of the same attempt regardless of its reported endpoint.
Read only source identity, completion/count metadata and MC endpoint eligibility;
do not open validation tables or select from their outcomes. Freeze the complete
eligible/selected/deferred/skipped ledger and prior contract/checkpoint hashes.
An input directory may be actively gaining other checkpoints: the frozen snapshot
contains precisely the checkpoints present at selection time.

Resume reads the saved selection directly. It verifies the normalized CLI
configuration, execution-source identities, and every frozen prior contract and
checkpoint hash, then opens only the saved checkpoint paths. Newly completed
coordinates in a prior input directory do not change that snapshot. Changing
scope, masses, input directories, batch settings, resource bounds or source files
requires a new output directory. `--plan-only` is an invocation mode and may be
removed to execute a saved plan with otherwise identical arguments.

The default scheduling slice remains 24 coordinates/60 minutes. Larger time
slices require `--batch-size 1`; batches defer work without ending the overall
task. This driver runs one worker with one BLAS thread. The external coordinator
controls the user's separately authorized maximum of two simultaneous workers.
This script never starts another worker or probes host memory. The final
`batch_summary.json` sets `invocation_finished=true` and identifies the scheduling
slice while retaining deferred, skipped and incomplete work. Intermediate
summaries use `invocation_finished=false`; finishing an invocation does not mean
that every required endpoint has been resolved.

## Allocation and model interface

Use production chunks of 128 rows. `ChunkedContext` calls the original
`Context.make_models` on each chunk. Each chunk retains its original
`BatchProfile` objects and fitted free/null state, so posterior retraining is
performed once per calibration spectrum. `AggregateModel` exposes `q(A)`, `r`,
`free['A']`, integer row access through `b[i]`/`L[i]`, dynamic maximum score and
summed scalar-fallback count. Row lookup never concatenates factors. The two
methods share background arrays within each chunk as before. Record chunk
boundaries, retained ranks/block layouts and original scalar-check indices.

`ChunkedBank` inherits unchanged moments, tails and local-p0 definitions.
Its initialization and weights evaluate the identical full-spectrum Poisson
log-density expressions in row blocks. All constituent bins and all proposals
remain in each density evaluation; no approximate density or weight clipping is
introduced. Original inversion and validation functions are imported unchanged.
Both truth banks remain available throughout inversion and validation.

Preallocate each whole count array as C-contiguous int64. For every proposal,
make exactly the original `rng.poisson(mean,size=(nper,len(mean)))` call, in
original order, and copy into its original row interval. Hash each returned block,
the concatenation of their bytes, and the completed whole array. The latter two
hashes must agree. The unrefined proposal and whole-array hashes must also match
the first-pass checkpoint. Use a contiguous memory view for whole-array SHA256;
never allocate a whole-array `.tobytes()` copy. Refined arrays have their own
original-recipe seed/shape/hash ledger, not a claimed first-pass identity.

## Numerical equivalence and failures

First reproduce the original backend, freeze all proposal arrays under it, and
run the unchanged stricter-candidate and extended-range audits. Retain every
rejected approximation and its eventual exact fallback. An exact reference
failure, actual scalar/batch mismatch or failure of the additional execution
checks is fatal; it never permits dropping a spectrum or declaring completion.

The execution audit replays all 18 original numerical spectra, generated from
the exact-backend proposals at strengths 0/2/5, and every extended candidate/
ceiling spectrum with all three proposal shifts for both truths. Extended
full-count hashes must match the preceding reference-audit ledger. These are
numerical audit spectra, not additional calibration or validation observations.

Under the final accepted backend, compare the entire audit set in one unsplit
batch with one-row chunks and scalar reference fits. The set must fit within 128
rows. One-row splitting exercises a different active-column and Newton-fallback
grouping from the unsplit reference. For every spectrum and method, compare signed
r and exact bounded q at its prescribed strengths: 2/5/12 and, for extended rows,
the candidate regions and ceiling. Require every pairwise r discrepancy at most
2e-5 and q discrepancy at most1e-4. Original per-chunk scalar checks and every-row
solver convergence/nesting checks also remain active during production.

Batch size can change which rows trigger scalar fallback. Chunk-local removal of
all-zero nuisance columns can also change matrix layouts. Consequently the claim
is an unchanged likelihood with checked numerical agreement, not bitwise identity
of fits, fitted parameters or fallback counts. The existing approximate-GP gates
of 0.001 remain unchanged and distinct from these tighter execution gates.

Compare blocked and original full-matrix Poisson log mixtures on the prescribed
audit spectra converted to the science count dtype. Require absolute log-density
error at most 1e-7. For weights at zero, every inversion scan node and every audit
strength, require finite values and
`max_scaled_error=max(abs(old-new)/(1e-12+2e-7*abs(old)))<=1`.
This is the numeric form of the original absolute-plus-relative tolerance.
Save that maximum with the absolute/relative errors and finite flag; the
supplemental gate must recompute acceptance from finite numeric errors rather
than trust the saved pass boolean. These bounds are far below the MC precision
target. Preserve raw discrepancies in the ledger.

Any failure requires a fresh output directory; retain `FAILURE.txt` and numerical
QA. A memory/geometry guard writes `DEFERRED.json`. Neither constitutes a result.
No failed or deferred coordinate is silently replaced by a smaller ensemble.

## Memory and reproducibility contracts

Default and maximum declared peak-array budget is 4 GiB per worker. Before any
calibration generation, choose the conservative estimate for the actually
accepted backend. It includes both banks' integer spectra, backgrounds/counts,
factor arrays, free/null state and q caches; chunk fit/density workspaces;
proposal arrays; two retained 500-row validation sets at the assignment boundary;
one original-shape RNG temporary; and a 512 MiB runtime reserve. This is a
source-derived estimate, not measured process RSS or a system memory guarantee.
Keep an explicit deferred record if the bound fails. Dense combined banks still
require too much retained factor memory at some coordinates; no disk backing or
other approximation is implemented here.

Each bank reserves `max(128,2*(len(scan_nodes)+18))` full q vectors. The second term
bounds both methods' positive scan nodes, up to 14 bisections, center/two slope
evaluations and validation strengths 2/5. Save capacity, keys, actual count and
retained bytes. An unexpected extra cache key beyond capacity fails; never evict
statistics or lower the sample count to make the estimate pass.

Preserve the original 47 `hashes` exactly and the declared sampling marker/type/
version. `sampling_hashes` additionally freezes this driver/protocol, the original
refinement sources, selection and companion runtime identities. Both contract
and result add an identical `execution_layout` object:

- `type='chunked_sampling_execution'`, `version=1`, `chunk_size=128`;
- `runtime_reserve_bytes=536870912`, `max_memory_gib`, `blas_threads=1`;
- repository-relative driver/protocol `source_hashes`;
- `sampling_policy_unchanged=true`, unchanged sampling/numerical policy names,
  and the explicit floating-point identity qualification.

The result adds `chunked_equivalence_checks`, `qcache_ledger` and the SHA256 of
`model_chunk_ledger.json`. The equivalence ledger contains exact spectrum
coordinates/hashes, all method-specific scalar/split/unsplit r/q comparisons,
truth-specific density/weight comparisons and whole-bank generation closure.
Its overall `passed` is true only after both bank closures and all numerical
checks pass. A supplemental execution gate must verify these fields and source
identities before the existing collector's selected results enter the note.
Do not modify or bypass the original collector's scientific and candidate gates.

Per-truth counts remain explicit; legacy heterogeneous counts remain null.
Validation rescores the same 500 original holdout spectra per cell. Those reused
spectra are not additional independent observations, and counts are never pooled
across execution layouts or sampling attempts.
