# Post-phase-1 steering amendment

This amendment records explicit user steering received after the original
phase-1 selection summary returned no candidate under the stricter uniform
`abs(mean pull) < 0.5` criterion. It does not alter, overwrite, or relabel the
hash-bound extraction protocol or its initial failed-gate result.

The user clarified that the pull need not be perfect and that the support is
acceptable if the positive or negative bias is generally below 0.75. For an
auditable decision, v4.9.5 implements "generally" as follows:

- at least 9 of the 12 mass/injection cell means have `abs(mean pull) < 0.75`;
- at least 3 of the 4 zero-signal cell means have `abs(mean pull) < 0.75`;
- no cell at the selected edge has `abs(mean pull) >= 1.25`;
- the original technical, covariance, optimizer-reproducibility, and
  no-kernel-bound requirements remain unchanged;
- among qualifying candidates, retain the predeclared minimax score and 0.10
  tie margin, choosing the smallest tied edge to keep the most GP support;
- the independent continuation must satisfy the same fractions and gross-bias
  guard both separately and in the combined full-100 summary.

Under the already completed phase-1 ledger, 36 and 38 MeV satisfy the amended
acceptability rule. The 38 MeV edge has the smallest worst-cell absolute mean
pull, while 36 MeV lies within the predeclared 0.10 tie margin and therefore is
the provisional edge. Phase 2 is authorized for 34, 36, and 38 MeV only.

This steering amendment is a post-phase-1 decision rule and must be identified
as such in the note. It is not a predeclared equivalence test, coverage
criterion, or permission to choose the strongest observed limit.
