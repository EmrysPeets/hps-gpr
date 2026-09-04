# Pre-execution amendment 2: absolute predictive-adequacy guard

Frozen before any candidate-specific fit or predictive score was run on
2026-09-02.  This amendment adds an absolute gross-misfit guard to the protocol
and first pre-execution amendment.  It does not alter the candidate grid,
training masks, primary NLPD comparison, or null fallback.

Relative improvement is insufficient if every support predicts the held-out
controls badly.  Therefore each support in each phase must also satisfy all of
the following on its low-control selected branches:

1. the mean joint Mahalanobis statistic per held-out bin, averaged first over
   the five kernel anchors and then over L1--L4, is strictly below 4;
2. no individual anchor/block joint Mahalanobis statistic per bin is 9 or
   larger; and
3. no held-out-bin marginal standardized residual
   `(y_i - mu_i) / sqrt(V_ii)` has absolute value 5 or larger.

The candidate-independent high-control check must satisfy the same three
requirements using H1--H4.  These are pragmatic gross-misfit guards, not
calibrated goodness-of-fit p-values.  Correlations among bins, blocks, and
kernel anchors prevent interpreting the thresholds as nominal chi-square tail
probabilities.

The 30--210 MeV fallback must pass the absolute guard and all technical gates
in both development and full-control phases.  Otherwise the study stops with
no selected support.  A nonreference candidate must pass the absolute guard in
addition to every relative displacement rule.

The implementation ledger must record exact training and held-out masks,
training-center minima/maxima and counts by region, and the number of training
or scored centers in 39--180 MeV.  That forbidden count must be exactly zero
for every fit.  Optimizer seeds are set through scikit-learn's deterministic
`random_state`, not merely recorded as labels.

Finally, a control-only support freeze is not a production-fit validation.
Before downstream observed limits are promoted, every production mass point
must separately pass no-bound, optimizer-repeat, finite covariance, and branch
reproduction checks under the frozen support.
