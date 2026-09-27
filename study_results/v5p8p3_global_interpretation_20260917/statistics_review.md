# Statistical interpretation audit of v5.8.2

Source paths below are relative to `study_results/v5p8p2_nominal_gp_significance_20260917/`. This is a read-only audit of the saved construction; no new fits or toys were performed.

## Ordering and conditioning

No consequential mathematical defect was found in the implemented local/global ordering. `scripts/analyze.py:16–27` builds the correlation matrix from the full count response, simulates correlated Gaussian fields, applies the raw-positive-fit gate, and compares the maximum with the observed standardized score. With fixed source B, a=r(B), s²=DᵀD on the diagonal, and W~N(0,R), the approximation is r*=a+sW. At a positive observed fit, zobs=(robs−a)/s>−a/s. Therefore W_m≥zobs automatically implies r*_m>0 at that mass: the gated local exceedance probability is exactly sf(zobs) within this Gaussian approximation. Its event is contained in the scan-maximum event, proving pglobal≥plocal apart from simulation fluctuations. For nonpositive observed fits, the inclusive excess-statistic tail is one. The gray conventional curve instead uses sf(max(0,r)), which is one half at its zero boundary; that plotting convention is explicitly distinguished in the report.

The background regression is redone as counts fluctuate. `scripts/engine.py:21–30,40–59` includes count dependence of both predictive mean and covariance in the differential response. The likelihood profiles its Gaussian background nuisance and the signal amplitude; dataset-bin directions concatenate independently in the combination. These operations preserve correlations between neighboring masses, between overlapping fit windows, and across changing combined composition. There is no multiplication by the number of plotted points and no assumption that neighboring masses are independent datasets.

The approximation nevertheless conditions on an observed-data-derived source B. It is a plug-in reference, not an independently measured background. Source construction itself is not repeated to recompute B, a and s in every replicate. Good conditional Poisson closure does not establish unconditional calibration of that whole procedure, rule out source-signal absorption, or quantify source-model uncertainty. Source estimation from data is not by itself proof of invalidity, but the stronger discovery claim needs validation that includes the full source-estimation pipeline and plausible alternative nulls. The 256 complete Poisson scans validate moderate tails, not a rare discovery tail. Separate scope-wise global p values do not account for choosing the most significant of the four reported scopes.

## Exact relation to the existing 90% CLs limits

Both procedures use the same underlying Poisson plus Gaussian-nuisance profile likelihood, but answer different questions. `scripts/limit_solver.py:222–278` tests a positive signal amplitude A. Its denominator is the unconstrained best fit if Ahat≥0 and the null fit otherwise; it sets the upper-limit statistic to zero when A≤max(0,Ahat). It profiles both the observed statistic q and a background-Asimov statistic qa. `scripts/limit_solver.py:22–44` then computes

    CLs(A) = CL_sb / CL_b = sf(z_sb) / sf(z_b),
    if q <= qa: z_sb=sqrt(q), z_b=sqrt(q)−sqrt(qa),
    otherwise: z_sb=(q+qa)/(2sqrt(qa)), z_b=(q−qa)/(2sqrt(qa)).

The routine solves CLs(A90)=0.1. These are asymptotic bounded-q tails at fixed mass. The v5.8.2 discovery mapping instead asks whether a background-only experiment produces an excess at one mass or anywhere in a declared search. It neither uses the CLs ratio nor rederives upper limits. Existing 90% CLs limits therefore remain unchanged; calibrating these new background tails does not automatically recalibrate signal-plus-background coverage.

Derived illustration: if Ahat~N(A,σ²), the same formula reduces to CLs(A)=sf((A−Ahat)/σ)/Φ(Ahat/σ) for A≥max(0,Ahat). At median background Ahat=0, CLs=0.1 gives A90=1.64485σ, while the conventional excess p0 is 0.5 (the exact inclusive zero-statistic tail is one). Thus a 90% CLs exclusion is neither a 90% probability that a signal is real nor a 1.28σ discovery threshold. The CLs denominator protects against exclusion where sensitivity is poor; it is not a look-elsewhere adjustment.

Authoritative sources: [Cowan et al., arXiv:1007.1727v3](https://arxiv.org/html/1007.1727v3), especially separate discovery and upper-limit statistics and Eqs. 16, 65–67; [Read, Presentation of search results: the CLs technique](https://cds.cern.ch/record/722145?ln=en), DOI 10.1088/0954-3899/28/10/313. The source-specific implementation statements and Gaussian illustration above come directly from the audited repository equations, not assumed CLs conventions.

## Why a resolution-element count need not match the tail penalty

The mass resolution creates correlations; it does not partition a moving scan into an exact integer number of independent tests. As a derived idealized illustration, a Gaussian signal template of standard deviation σ in white noise has response correlation R(Δ)=exp[−Δ²/(4σ²)]. A smooth interval of length L then has high-threshold crossing approximation

    P(max W >= u) ≈ sf(u) + L/(2π sqrt(2) σ) exp(−u²/2).

Consequently its tail multiplier is approximately 1+Lu/(2sqrt(π)σ) for large u. It grows with threshold and differs from L/FWHM even though the spectrum contains no new independent data when the scan step shrinks. This formula is explanatory only: the actual HPS response has moving masks, variable resolution, GP retraining, and dataset transitions, and its saved covariance should decide the correction. A Sidak-equivalent N_eff=log(1−pglobal)/log(1−plocal) is a descriptive, threshold-specific number, not a measured number of independent physical bins.

The saved half-MeV maximum is at least as large as its nested one-MeV maximum (`scripts/analyze.py:19–22,34–36`). Denser sampling can catch maxima missed by the coarse grid; it cannot reduce a fixed-threshold scan tail. Half-MeV continuum convergence remains unestablished. Conversely, a smaller predeclared domain legitimately has a smaller trials penalty. Selecting that domain after seeing the favorable peak would require accounting for the selection.

## Defensible consequence

This study is a substantial methodological cross-check: it separates reference bias, fluctuation width, and mass-search multiplicity, and propagates the actual fitted estimator through all three datasets and their common-coupling combination. It demonstrates how the significance of the same unchanged observed likelihood root depends on a coherent null reference. It does not increase observed event information, establish a new physical excess, or justify weakening the trials penalty to match a hand-counted resolution estimate. The value is a more explicit, testable calibration framework, with fixed-source and rare-tail limitations kept visible.
