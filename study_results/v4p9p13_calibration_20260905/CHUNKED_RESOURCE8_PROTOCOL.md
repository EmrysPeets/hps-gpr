# Combined-only memory scheduling allowance, version 2

`run_chunked_refinement_8gib.py` is a resource wrapper around the frozen
`run_chunked_refinement.py` and `CHUNKED_REFINEMENT_PROTOCOL.md`. It permits a
source-derived per-worker peak-array budget of at most 8 GiB for the all-three
combination. The wrapper verifies those two inherited files against their frozen
SHA256 identities. It does not modify either file or any original inference,
sampling, collector or audit source. The numerical functions are imported
directly from the same frozen base; the 6 GiB wrapper is not a runtime dependency.

This allowance addresses prescribed second-attempt combined plans deferred by
the frozen 6 GiB guard. The version-1 6 GiB wrapper and protocol remain unchanged;
this version uses its own wrapper, protocol, pure QA and output tree. It is not a
new statistical study. The coordinator supplies the
memory-deferred combined masses after review; the wrapper also retains the
original MC eligibility and ordering rules. Individual-dataset scopes are
rejected. The default scope is the all-three combination and the default memory
budget is 8 GiB; a smaller positive budget is permitted.

## Operational authorization and launch condition

At most one worker using this allowance may run, alongside at most one ordinary
worker with a 4 GiB budget. Their aggregate declared array budgets must not exceed
12 GiB. Every worker retains one BLAS thread and chunks of 128 spectra. The
coordinator must perform a fresh memory-pressure check before every numerical
launch, including a resumed invocation, and keep the laptop responsive. It must
defer the launch if the current workload is unsuitable. The wrapper neither
starts companion workers nor probes host state, and its pure QA is not evidence
that this operational launch condition has been satisfied. The coordinator
records the actual check with the launch; no stale pressure percentage is
embedded in a reusable scientific contract.

## Unchanged inference and execution

The wrapper imports the original `run_point`, `runtime_types`, `memory_estimate`,
`array_sha`, `sampling_input`, `saved_selection` and `verify_inputs` functions as
the same Python function objects. Its plain main dispatch changes only allowed
scope, memory bound, output namespace and resource/source identity. There is no
AST rewrite, monkey-patching or replacement of imported module globals.

Proposal laws, truth models, signal normalization, frozen reviewed kernels,
per-spectrum posterior retraining, seed namespaces, draw order/shapes, retained
whole-array hashes, both truth banks, q-cache bounds, 90% CLs inversion and
validation are inherited unchanged. Numerical tolerances, strict-candidate
selection, exact fallback, scalar checks, importance-weight checks and all failure
handling remain unchanged. The same holdout spectra are rescored and never pooled
as new independent validation observations. This resource allowance does not
relax MC precision or provide another calibrated truth scenario.

The same memory estimate, including its 512 MiB runtime reserve, is compared with
the requested bound for the actually accepted backend. A bound above 8 GiB is
rejected before native reconstruction. An estimate exceeding the bound remains
explicitly deferred; there is no smaller ensemble, coarser mesh, disk backing or
automatic additional memory allowance. The bound is a source-derived array
estimate, not a measured process-RSS or system-memory guarantee.

Frozen selection and resume behavior are retained. A saved selection opens only
its frozen prior checkpoint snapshot after verifying CLI configuration and
source/input hashes, even if an input directory later gains other completed
points. Removing `--plan-only` permits execution with otherwise identical
arguments. Resource/source/configuration changes require a separate output tree.
The default output namespace is `chunked_resource8_v1`. Scheduling slices, deferred
work and final invocation flags retain their original meanings.

## Resource identity and QA

`layout_marker(max_memory_gib)` returns the inherited execution layout with the
same `type='chunked_sampling_execution'`, version 1, chunk size, thread count and
statistical/numerical policy fields. Its `source_hashes` contains exactly the
base chunked driver/protocol and this wrapper/protocol. It adds:

```
resource_policy:
  type: combined_memory_override
  version: 2
  scope_key: all_2015_2016_2021
  max_worker_memory_gib: 8.0
  max_companion_workers: 1
  max_companion_memory_gib: 4.0
  aggregate_memory_limit_gib: 12.0
  fresh_memory_pressure_check_required: true
  prelaunch_check_owner: coordinator
  statistical_policy_unchanged: true
```

The actual requested per-worker bound remains `execution_layout.max_memory_gib`.
Contract, selection, point plan and result retain the same layout object. All four
source identities occur in `sampling_hashes` as well, alongside the original
refinement dependencies and frozen selection. The original 47-entry inference
hash map is unchanged. Existing per-point equivalence and model-chunk ledgers
are produced by the inherited runner without schema changes.

Run `python3 -B run_chunked_refinement_8gib.py --pure-check` from this directory
to write `qa/chunked_resource8_contract_test.json`. Its 26 checks cover allowed
and rejected memory/scope values, the four source identities, original execution
settings, unchanged runtime function objects, inherited 24-case QA and original
47 dependencies, and absence of fitting-module imports. It performs no fits or
random draws. The resource QA and inherited pure-QA file are frozen in the
wrapper's sampling contract; numerical dispatch requires their current identity.

Supplemental auditing must explicitly recognize this resource marker, compare
against this wrapper's `layout_marker`, enforce combined scope and the 8 GiB
ceiling, and retain every original numerical/MC gate. The note and release tools
must be updated separately before results using this allowance can be published.
