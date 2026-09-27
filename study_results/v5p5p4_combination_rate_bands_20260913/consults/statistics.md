# Statistical consultation: comparing combination rules at the same mass

The fixed-mass alternatives answer different questions. A common coupling specifies relative expected campaign yields; unconstrained positive campaign amplitudes only require nonnegative signals at the same specified mass. Independent experiments are already combined by multiplying their likelihoods. Replacing that product by a combination of extracted signals cannot create additional information. It can, however, remove an unsuitable relative-rate restriction or choose a test that emphasizes concordant standardized excesses.

## Matched experiment and signed extraction

`scripts/extracted_combinations.py` uses the unchanged v5.5.3 fixed fitting windows, background state, MC92 normalization and fully scaled widths. The grid is 90–94 MeV in 0.25 MeV steps, exactly the archived score bank. The primary extracted estimate in campaign d is

    a_d = (S_d^T V_d^-1 r_d)/(S_d^T V_d^-1 S_d),
    sigma_d² = 1/(S_d^T V_d^-1 S_d), z_d=a_d/sigma_d,
    r_d=n_d-b_d, V_d=diag(lambda_null,d)+L_d L_d^T.

The amplitude unit is epsilon²/10^-8 only as an equivalent normalization; an independent campaign amplitude does not establish an actual common kinetic-mixing coupling. Negative estimates are retained. These three estimates and their variances preserve the entire amplitude-dependent Gaussian likelihood at fixed shape. Direct bin-level GLS and the compressed likelihood agree to numerical precision. A separate overlay compresses the exact signed Poisson MLEs with their fitted curvatures; this latter approximation is not assigned the exact Gaussian sampling claim.

The reference treats the fitted V as known and fluctuates count noise plus the GP auxiliary/predictive uncertainty. It is not direct Poisson calibration, a sideband-refit ensemble or a proof of coverage for the full analysis. Campaign covariance is block diagonal, as in the parent model; unmodeled shared systematics are absent. Correlation across masses is explicitly saved in `derived/extracted_combination_covariance.npz`.

## Local combination rules

| Rule | Statistic and null reference | Interpretation |
|---|---|---|
| Common coupling | Z=(sum a_d/sigma_d²)/sqrt(sum 1/sigma_d²), p=1-Phi(Z) | One specified relative-rate direction |
| Independent positive amplitudes | Q+=sum max(z_d,0)²; p=sum(k=1..3) C(3,k) chi²_k.sf(Q+)/8 | Tests the positive orthant; the null atom has weight 1/8 |
| Signed Stouffer | Z=sum z_d/sqrt(3), p=1-Phi(Z) | Equal predeclared weight for standardized campaign evidence |
| Signed Fisher | F=-2 sum log[1-Phi(z_d)], p=chi²_6.sf(F) | Another predeclared rule for independent one-sided probabilities |

The independent-amplitude inclusive tail is p=1 at Q+=0. Fisher uses probabilities derived from signed z, never clipped positive-only probabilities. Under the stated Gaussian reference, the signed individual probabilities are independent uniform variables at the null; this gives the displayed Fisher law directly. Stouffer corresponds to a common standardized shift; it need not be the most powerful test for unequal physical signals. Neither positive-amplitude LRT nor Fisher proves a signal in every campaign: a strong subset can drive rejection. Requiring a replicated signal in all three would be a different hypothesis test.

At 92 MeV the primary Gaussian-reference results are common coupling Z=2.44664 (p=.00720966), independent positive amplitudes Z=3.64560 (p=1.33383e-4), Stouffer Z=3.96240 (p=3.71004e-5), and Fisher Z=3.81124 (p=6.91359e-5). The independent raw root is sqrt(Q+)=4.195 rather than its equivalent Z=3.646. Choosing whichever displayed rule looks strongest requires an additional model-choice correction; none is supplied here. Every displayed mass is pointwise. The fixed-mass references cannot be applied directly after optimizing mass offsets, widths, energy-law parameters or a mass scan. An existing effective-trial Sidak curve can be shown as an explicit common reference transformation, not as calibration of these new statistics.

## Upper limits with a comparable estimand

Let g_d=sum S_d over the fixed fitted bins. The common-coupling total expected reconstructed signal is T_common=a*sum g_d. Without any relative-rate law, the signed estimator of the same total signal T=sum g_d a_d is T_hat=sum g_d a_hat_d, with sigma_T²=sum g_d² sigma_d². Its mean is T for any signal allocation. Thus the known-Gaussian CLs 90% bound

    U90 = x - s Phi^-1[0.1 Phi(x/s)]

can use x=T_hat,s=sigma_T without imposing a coupling law. It is positive and at least as large as x+1.28155s. Consequently it has at least 90% coverage for physical T>=0 in this specified Gaussian model. Use the same formula for the common-law estimator and then map it to T. At 92 the bounds are 30,943.54 and 36,522.90 reconstructed signal rows, respectively. These are fitted-window yields across the supplied samples, not cross sections or universal epsilon² limits. Limits after floating mass/width would require a new construction.

Validation includes all 17 exact common-coupling Poisson CLs fits, exact independent-campaign Q replay, Gaussian likelihood compression, source-score reconstruction, Gaussian CLs roots and analytic coverage checks. No random toys or new global-significance calibration were run.

## Primary references checked

- G. Cowan, K. Cranmer, E. Gross and O. Vitells, *Asymptotic formulae for likelihood-based tests of new physics*, EPJC71 (2011)1554, [arXiv:1007.1727](https://arxiv.org/abs/1007.1727). Likelihood profiling and asymptotic reference distinction.
- S. G. Self and K.-Y. Liang, *Asymptotic Properties of Maximum Likelihood Estimators and Likelihood Ratio Tests under Nonstandard Conditions*, JASA82 (1987)605–610, [primary paper](https://www.stat.cmu.edu/~brian/763-2015/week06/papers/self-liang-1987.pdf). Boundary/cone mixture reference; the orthogonal Gaussian mixture above also follows directly from independent signs.
- E. Gross and O. Vitells, *Trial factors for the look elsewhere effect in high energy physics*, EPJC70 (2010)525–530, [arXiv:1005.1891](https://arxiv.org/abs/1005.1891). Pointwise versus searched-mass probability.
- N. A. Heard and P. Rubin-Delanchy, *Choosing Between Methods of Combining p-values*, [arXiv:1707.06897](https://arxiv.org/abs/1707.06897). Combination rules encode different alternatives and power choices.
