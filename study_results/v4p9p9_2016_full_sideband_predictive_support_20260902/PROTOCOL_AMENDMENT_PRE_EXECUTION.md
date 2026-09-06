# Pre-execution amendment: separated control regions

Frozen before any candidate-specific fit or predictive score was run on
2026-09-02.  This amendment is authoritative where it differs from
`STUDY_PROTOCOL.md`; all other provisions remain unchanged.

The statistics review identified that fitting one GP jointly to low and high
controls would bridge the unobserved 39--180 MeV search gap.  Although no
search bin would enter that fit, the resulting extrapolation should not be
called a production-like control validation.  The control regions are
therefore separated as follows.

## Low-control support selector

The primary and only support-dependent score uses the four predeclared low
blocks L1--L4.  For a held-out low block, training uses only:

- the candidate-specific interval `[edge, 35.25)` MeV; and
- the other three low-control blocks in `[35.25, 38.75)` MeV.

No high-control bin and no 39--180 MeV bin enters a low-control fit.  Candidate
comparison, paired standard errors, the `>1 SE` displacement rule, the positive
low-score direction, and leave-one-low-block-out stability are computed from
L1--L4 only.  The earlier equal low/high weighting formula is superseded.

With four low blocks, the paired standard error is

`SE_low = sd(Delta_L1..Delta_L4) / sqrt(4)`.

The Poisson-deviance robustness direction must remain nonnegative on the same
four paired low blocks.

## Separate high-control check

The four high blocks H1--H4 are evaluated once per dataset and kernel anchor,
independently of the lower-support candidate.  For a held-out high block,
training uses only the other three high blocks in `[181, 210)` MeV.  No low or
search-region bin enters.  This is a fixed-upper-edge/model technical check,
not a lower-edge ranking score.  Every high-control cell must satisfy the same
finite optimizer, LML reproduction, kernel-bound, and predictive-covariance
gates.  Failure of the high-control check stops the study without selecting a
support.

## Sequential fallback after separation

Phase 1 still evaluates supports 29--33 plus control 34 on the 2016 10%
development histogram.  A candidate may advance only with low-control NLPD
improvement greater than one paired low-block SE, nonnegative low-control
Poisson-deviance improvement, technical passage, and positive improvement
after deletion of any one low block.

Phase 2 confirms support 30, every Phase-1 qualifier, and their immediate
eligible neighbors using the same low-only procedure on full-2016 controls.
Only a Phase-1 qualifier may displace support 30.  If there is no common clear
winner, evidence is tied or ambiguous, either dataset's high-only check fails,
or support 30 fails a required technical gate, the original fail-closed/default
rules apply.

The resulting selector is deliberately conservative.  It assesses whether a
small change in the low support edge improves prediction of the adjacent
out-of-search threshold controls.  It does not claim to validate interpolation
through the search range or to calibrate later limits or p-values.
