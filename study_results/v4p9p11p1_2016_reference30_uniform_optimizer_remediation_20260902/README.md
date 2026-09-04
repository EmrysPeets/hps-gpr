# v4.9.11p1 terminal numerical-remediation result

This release is terminal and **does not authorize combination or inference**.

The prospectively frozen, uniform optimizer run completed exactly 2,556 paths
(18 at each of 142 masses) and its runner-level decision provisionally marked
142/142 states resolved.  The separately frozen independent validator then
failed 3 of 17 checks.  It reproduced path eligibility, the global fixed-LML
maximum, and the two-method-family clusters, but selected-state fixed-GP replay
closed within the frozen `1e-6` LML tolerance at only 87/142 masses.  Exact
prediction/covariance hashes also showed numerical replay drift.  The validator
failure overrides the provisional runner authorization.

No retry, tolerance change, state substitution, signal fit, p-value, or limit
was performed.  The stored 142-row state CSV is provisional evidence only and
must not be consumed by a combined-result driver.

Fit-only diagnostics from the provisional states show that the k12 upper
length bound was not contacted under the frozen `rtol=1e-3, atol=1e-12` rule:
the largest selected length was `0.4775337971` at 103 MeV versus an upper bound
of `0.4909913063` (ratio `0.9725911458`), and there were zero selected contacts
with either length or constant bound.  This does not repair the failed
independent replay and is not an inference result.

Canonical outcome and hashes are in `TERMINAL_RELEASE_STATUS.json` and
`TERMINAL_STATE_SHA256`.
