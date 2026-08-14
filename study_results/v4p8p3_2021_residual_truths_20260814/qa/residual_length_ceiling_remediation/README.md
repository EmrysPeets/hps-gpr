# Ceiling-remediation QA snapshots

`task_manifest.csv` is the immutable pre-selection authorization snapshot.  It
correctly records confirmation tasks as unauthorized before the selection
gate was evaluated.  It must not be rewritten after the fact.

Current authorization and completion are instead established by the chained
artifacts:

1. `derived/residual_length_ceiling_remediation/selection/selection_disposition.json`
   selects factor 50 against the factor-75 sentinel and authorizes only the
   five predeclared confirmation toys;
2. `derived/residual_length_ceiling_remediation/confirmation/task_product_audit.csv`
   records those five completed products; and
3. `derived/residual_length_ceiling_remediation/confirmation/final_disposition.json`
   reports `qualified_targeted` while explicitly denying an all-lane
   qualification or closure rerun.

The pre-selection manifest and the later disposition are therefore different
time-indexed records, not contradictory current-state declarations.
