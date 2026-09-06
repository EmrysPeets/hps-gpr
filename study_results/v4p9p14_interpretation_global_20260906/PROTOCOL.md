# v4.9.14 interpretation and 2015 global-significance pilot

Declared 6 September 2026 before generating this study's scan ensembles.

The parent v4.9.13 results remain frozen. The primary statistical review is
independent of this implementation. This derivative explains their physical
meaning and implements an Ananiev--Read (arXiv:2206.12328v3) covariance pilot.

## Scope and ensemble

Start with full 2015, all 72 existing hypotheses from 19 to 90 MeV in 1 MeV
steps. Use the archived 2015 smooth functional-form expected histogram already
named in the v4.9.13 input contract, rebinned identically over the full GP
support (14--135 MeV). It supplies one coherent background-only spectrum for
the entire scan. It is a conditional stress model, not certified background
truth. Each Poisson toy is scanned at every hypothesis with its identical
full-spectrum counts. Refits update log-count means and count-dependent errors;
reviewed mass-dependent kernel coordinates, masks and signal templates stay
fixed. Gaussian-profiled and fixed methods use paired spectra.

First generate ten complete scans (ten toys at every hypothesis). Keep them
separate as an implementation/timing pilot. If efficient, generate 1,000 new
independent full scans for validation. Run one numerical process with one BLAS
thread, in small batches; preserve all failures. Defer other datasets if a
faithful extension is not quick. Existing pointwise toy IDs must not be joined
across mass: they were generated independently or from different truths.

## Covariance and inference

Scan the unfluctuated mean B, then B+sqrt(B_i)e_i for every full-support bin i.
Let r be the signed square root of the profiled likelihood ratio. Store
mu(m)=r(B;m), D[i,m]=r(B+sqrt(B_i)e_i;m)-mu(m),
C=D^T D, s(m)=sqrt(C_mm), and K_mn=C_mn/(s_m s_n).
This extends the paper's mean-zero/unit-variance premise by recording the
nonzero interpolation offset and nonunit width explicitly. Never silently
force the raw likelihood-ratio statistic to have its asymptotic distribution.
The centered response construction reduces to the paper's Asimov outer-product
method in its regular unbiased regime. Independent Poisson scans test this
linear/Gaussian approximation, including correlations and maximum tails.

Sample 200,000 inexpensive Gaussian fields from K. Reconstruct r*=mu+s*z.
Retain the bounded discovery atom: p_local=1 for r<=0; otherwise
p_local=normal_survival((r-mu)/s). The principal global pilot statistic is the
minimum of these truth-specific local p-values over the declared 72 points.
Equivalently maximize z only where r>0. Report the probability of any point
being at least as extreme as each observed local p, with binomial MC intervals.
Also retain the maximum of the raw nonnegative signed root as a separate
ordering cross-check. These are separate scan statistics; neither is a global
calibration of the parent's two-truth p-value envelope.

Verify observed signed roots against the frozen parent. Compare the Asimov
Gaussian approximation with held-out direct Poisson scans without tuning the
covariance, mean or widths to obtain agreement. Report marginal means/widths,
normality diagnostics, covariance discrepancy, and scan-tail binomial
intervals. Test 2 MeV subgrids to expose discretization sensitivity; 1 MeV
results apply only to the finite grid. No continuous-mass, combined-dataset,
width-search or method-selection trials correction is claimed.

## Reproducibility and publication

Use deterministic seeds keyed by dataset and ensemble identity. Persist full
toy counts, signed-root scan vectors, Asimov response vectors, source hashes,
all numerical checks, summary CSV/JSON, figure data, LaTeX and final PDF. Resume
only when the saved source/count contract agrees. A changed contract goes to
a new directory. Report scientific qualifications independently of numerical
or rendered-page QA. Expected-limit bands and a new final analysis selection
are outside this derivative.
