# Independent protocol review: v6.2 MC injection study

Review date: 23 September 2026. This review checks the proposed study against
the live v6.1 implementation. It does not attest to unrun v6.2 numerical results.

## Accepted design

- Use only supplied native masses 60, 80, ..., 260 MeV; exclude 40 MeV.
  There are eleven masses and four nonzero injection counts: 1000, 5000,
  10000, and 30000, each with twenty toys.
- Treat the stored selected-candidate histogram as the conditional empirical
  signal model. Retain its full shape, including broad components. Do not add
  smearing, translate it, truth-match it, or replace it with the exploratory
  analytic shared-shape approximation.
- Draw exactly N selected candidates per toy by multinomial sampling. The
  analysis-bin probabilities are the unconditioned selected-MC probabilities;
  the remaining probability is an explicit outside-analysis category. The
  outside category includes stored out-of-range selected candidates. This is
  equivalent to drawing from the piecewise-uniform empirical histogram and
  retaining candidates in the analysis support, with measured overflow
  probability carried separately. It is not a Poisson draw with mean N.
- Fit with the same unconditioned analysis-bin probabilities so that the
  unconstrained amplitude is in full-selected event units and its truth is N.
  Never normalize the template inside the fit window. Save actual in-support,
  in-window, and outside counts separately from their expectations.
- Supply the identical full toy spectrum to both extraction approaches.
  The pole-centered method uses m +/- 2.25 sigma_nominal(m). The shifted
  method uses c_MC(m) +/- 2.25 sigma_nominal(m), with the v6.1 MC-only core
  location. The signal shape is identical in the two fits. The fitted MC core
  width does not set the extraction window. No center is inferred from a toy.
- Recompute the GP prediction and covariance from each injected toy's exterior
  bins using its method's mask, retaining archived kernel coordinates. This is
  required to measure absorption of the broad signal tails by GP training.
- Share each Poisson background across injection strengths and methods;
  independent fixed-N signal draws at each strength are acceptable when
  recorded by deterministic seed. Reuse those backgrounds for zero-injection
  controls. Background-only-sideband and known-background controls distinguish
  signal absorption from background offset and likelihood response.

## Background and 260 MeV provenance

The proposed truth, v5.8.2 `inputs/null_2021.npz`, was verified independently:

- SHA-256: `306a915b9ba6230aafbe058c94438c6af11f85b60d68f5aae4ebbfec4d4a9424`.
- Its 422 bin edges and observed counts exactly match the v6.1 2021 inputs.
- Every mean is positive; minimum 4381.299311240344 events/bin and total
  141304390.30422956 events.
- The v5.8.2 preparation code constructs this all-data GP truth with a fixed
  76 MeV kernel anchor. That is the generator's provenance, distinct from the
  extraction kernels selected at each test mass. This remains a conditional
  nominal-null experiment, not detector/background validation.

Native 260 MeV MC exists, but v6.1 archived kernel states end at 250 MeV.
The extension must explicitly use the 250 MeV kernel anchor, while using the
260 MeV resolution and direct MC. It must not be described as an archived
260 MeV production analysis. The unchanged v6.1 core locator succeeds at 260:
c = 255.17306511579105 MeV, nominal sigma = 7.2959 MeV, fitted core sigma =
6.083724920739287 MeV; no mode-search edge or fit-bound hit was found.

## Required interpretation and numerical checks

Use signed, unconstrained fitted yields for bias and pulls. A clipped
nonnegative yield biases those summaries. The Hessian uncertainty must be
finite and positive, and fit convergence, expectation positivity, and
fixed-truth/free-fit likelihood nesting must be verified. The signed profile
root at N and profile containment at q(N) <= 1 or 3.841458820694124 are useful
alongside Hessian pulls. Any reported intervals must identify whether they
are Wald or nominal profile intervals.

For exact-N injections the signal-bin covariance is
N [diag(p) - p p^T], whereas the inherited likelihood uses Poisson counts.
Consequently a unit pull width is a diagnostic reference, not an identity,
even for a known-background control. State this conditioning explicitly.
Twenty-toy containment estimates require counts and binomial intervals;
they do not establish calibrated coverage. Pull-mean uncertainty and the
uncertainty of the width should remain visible. Do not pool different masses
or strengths into a single purported calibration statistic.

## Code inspected

The v6.1 `README.md`, `protocol.json`, `scripts/core_centering.py`,
`scripts/run_mc_study.py`, `scripts/analytic_shapes.py`, `scripts/common.py`,
`scripts/parent_core.py`, and `scripts/limit_solver.py` were read. The
v5.8.2 truth preparation and prior v5.1.1/v5.2.1 injection routines supplied
cross-checks for shared backgrounds, full-spectrum injection, and per-toy GP
refitting. Numerical v6.2 implementation review remains a separate step.
