# v4.9.15 global-significance extension: 2016 full and 2021 10%

Declared 6 September 2026 before generating extension ensembles. This extends
the frozen v4.9.14 2015 study to the existing individual-dataset scans: 142
masses from 39 to 180 MeV for full 2016, and 201 masses from 50 to 250 MeV
for 2021 10%, both in 1 MeV steps. These are separate searches; no joint
three-dataset or continuous-mass significance is claimed.

## Fixed statistical procedure

Use the v4.9.14 numerical runner without changing its likelihood, masks,
kernels, signal templates, batch size, tolerances, scalar cross-checks or
seed convention (the seed key remains v4p9p14-global, dataset, ensemble).
Generate ten coherent full-spectrum Poisson scans per dataset first. If they
pass and are efficient, generate 1,000 additional independent validation
scans and the full Asimov one-bin response ensemble per dataset. Run one
numerical process at a time, one BLAS thread, at low priority. Every toy is
scanned across all masses, preserving the cross-mass correlations. No joining
of independently generated mass-local toys is permitted.

Use the already contracted archived stress mean for the whole dataset:
2016 threshold-qualified full-count mean; 2021 native-10% fsig-anchor mean.
Require exact equality of the generating full spectrum across hypotheses.
These are conditional stress backgrounds. Retain the 2016 parent numerical
exception and its background-model qualifications, including the earlier
source-fit waiver, transition region and lack of independent truth certification.
Per-toy means and count-dependent errors are retrained; kernels stay fixed.

As in v4.9.14, compute mu=r(B), D_i=r(B+sqrt(B_i)e_i)-mu, C=D^T D,
s=sqrt(diag(C)) and K=C/(s s^T). Generate 200,000 Gaussian fields from K.
Retain the bounded atom: p_local=1 for raw r<=0, otherwise sf((r-mu)/s).
The principal scan statistic is the minimum of these common-truth local
p-values over the declared grid. Also show the separate maximum-raw-root
ordering, which preserves the original asymptotic p-value ordering. Neither
statistic calibrates the v4.9.13 two-truth envelope. The GP here models the
correlation of a likelihood-based scan, and does not replace the background
likelihood with a new global fit.

Validate the Asimov prediction against independent Poisson scans: marginal
mean, width and normality; correlation discrepancy; and maximum-tail
agreement with exact binomial uncertainty. Do not tune the model to these
validation scans. Report failures or unresolved tails without discarding
toys or choosing a more favorable ordering. Zero tails have binomial bounds.
Compare both offset 2 MeV subgrids only as a discretization diagnostic.
The two mass grids and dataset fractions are frozen before looking at results.

## Products and provenance

Keep all parent releases and the v4.9.14 study unchanged. Store full spectra,
per-mass signed roots and numerical audits, response covariance, field maxima,
plots and plot data, commands, source hashes, a standalone LaTeX extension
report, rendered-page review and a product manifest. The 2015 reference is
reused with explicit file hashes; it is not rerun or pooled with the new toys.
If a numerical failure requires a new implementation, preserve the failure
and create a separately contracted derivative; do not loosen gates silently.
