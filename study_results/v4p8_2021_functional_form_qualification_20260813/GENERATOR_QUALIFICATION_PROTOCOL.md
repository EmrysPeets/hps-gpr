# v4.8 source-generator qualification protocol

Status: development protocol, formalized after an exploratory fit reconnaissance on
2026-08-13.  It is therefore not a blind model-selection procedure and cannot by
itself promote a production analysis card.

## Scope and vocabulary

The object being fitted is a reconstructed 2021 invariant-mass spectrum after
trigger, reconstruction, and selection.  A fitted curve is called a **declared
generating mean** or **analytic stress generator**, never the physical background
truth.  The native 1% and native 10% spectra are fitted independently.  A common
family means a common formula and fitting protocol, not common parameters or a
nested data sample.

The requested name `fSigPowExp` is not a literal repository tag.  The repository
distinguishes the five-parameter `fSigPow` from the seven-parameter
`fSigPowExpQ`.  Both interpretations must be named explicitly in every ledger.

## Frozen extraction contract

The primary extraction control remains the frozen v4.2/v4.5 contract:

- search interval 50--250 MeV;
- data/GPR support 40--300 MeV;
- `pre_log=true`, `alpha=1/y`, rebin factor 5;
- 2.25-sigma blind and training masks;
- 12 optimizer restarts;
- 2021 length-scale factors 1.1--15;
- matched-refit background-only signal reference;
- signed extraction amplitude; and
- asymptotic `tilde_q_mu` CLs with `alpha=0.10` when limits are actually computed.

The 30--300 MeV support is a separate one-factor geometry control.  It may use
the same 30--300 MeV fitted mean and the same realized counts at and above 40 MeV,
with only the additional 30--40 MeV bins exposed.  It must not be combined with a
length-scale-bound change in the same comparison.

## Candidate families

1. Literal five-parameter `fSigPow`:

   `A sigmoid((m-mt)/w) m^a exp(-m/theta)`.

2. Archived seven-parameter `fSigPowExpQ`, retained only to document rejection.
   Its `c1*m` term is algebraically confounded with `exp(-m/theta)` and both
   archived source fits place `c1` at its +50 bound.

3. Archived implementation/optimizer controls: `fShiftSigPow`,
   `fShiftSigPowTail`, `fGenGammaThresh`, `fGenGammaShift`, `fEndpoint`,
   `fLogPolyThresh`, and `fBern5`.  These test the existing ROOT machinery but
   do not count as qualified merely because an analytic array is finite.

4. Positive global log-Chebyshev candidates:

   `sigmoid((m-mt)/w) exp(sum_{k=0}^d c_k T_k(u(m)))`,

   with a fixed mapping `u: [30,300] MeV -> [-1,1]`, common degree `d`, and
   source-specific coefficients.  Candidate degrees are 8, 12, 16, 17, 18, 20,
   and 24.  Degree is selected only from source-fit diagnostics, before any GPR
   pull, recovery, CLs, p0, or limit result is inspected.

## Development qualification gates

For both source spectra and for both 30--300 and 40--300 MeV evaluation domains:

- all fit attempts are finite and the selected solution converges reproducibly;
- the generating mean is strictly positive in every in-support native bin;
- native-bin and factor-five-rebinned Pearson and Poisson-deviance ratios are
  each in `[0.75, 1.25]`;
- outer blocked validation uses the five single contiguous folds
  `[30,84)`, `[84,138)`, `[138,192)`, `[192,246)`, and `[246,300)` MeV;
  every fold is refitted with starts constructed only from its training bins,
  and every fold plus the pooled deviance per held-out bin is reported;
- no fitted turn-on parameter is within 2% of a declared bound;
- no fitted peak--trough pair in 50--250 MeV is separated by less than the local
  full 4.5-sigma training-mask width; and
- deleting a +/-2.25-sigma fake blind window at each declared mass is followed
  by a signal-template projection of the full-fit versus held-out prediction.

The target projection budget is `|Delta A_model|/sigma_A <= 0.2`.  A candidate
that passes ordinary goodness-of-fit but fails this gate may still be used as a
clearly labeled conditional stress generator; it is not a qualified nominal
generator and cannot select the kernel ceiling or production card.

The current `fit_qualify.py` is a deliberately fail-closed reconnaissance
implementation, not a conforming realization of every gate above.  It uses a
single fit start, cyclic blocked subsets, center-evaluated means, and full-fit
warm starts for held-out fits.  Its output therefore cannot set a generator
even if its numerical flags appeared favorable; in the actual run, the
predictive gates fail by large margins as well.

## Toy and optimizer design after qualification

The requested conditional screen uses 25 independent background clusters per
source family.  The 1% exposure chain is nested by independent Poisson increments
from 1x to 10x to 100x; the native-10% chain is nested from 1x to 10x.  The two
source families are unpaired.  Toy index is reused over masses and injected
strengths, so the independent unit is the background spectrum, not an extraction
row.

The four reported lanes are 1%x10, 1%x100, native 10%, and 10%x10 at 65, 90,
120, 180, and 210 MeV with injected strengths 0, 1, 3, and 5 reference sigmas.
Twenty-five backgrounds are a screening ensemble, not a coverage study.

A separate pull-blind optimizer scan reuses the same backgrounds at masses
50, 70, ..., 250 MeV and factors 15, 20, and 25.  A bound recommendation requires
boundary/near-bound occupancy, reproducible higher-likelihood branches, nested
likelihood non-regression, and a 20-to-25 plateau.  Pulls, recovery, CLs, p0, and
limits are forbidden selection inputs.
