# Reading the GP checks

## What is r_i on slide 13?

It is the **standardized residual in mass bin i**:

\[
r_i=\frac{n_i-\widehat b_i}{\sqrt{\widehat b_i+C_{\mathrm{GP},ii}}}.
\]

Here n_i is the observed count, b̂_i is the GP-predicted count, and C_GP,ii is the GP prediction variance in that bin. The denominator combines counting variance, approximated by b̂_i, and GP variance. Therefore:

- r_i = 0: the count equals the GP prediction.
- r_i = +2: the count is two estimated combined standard deviations above the prediction.
- r_i = −2: the count is two estimated combined standard deviations below it.

The green ±2 band is a visual reference. Each bin is withheld from its own local prediction, but overlapping fits correlate the residuals across bins. The band is not a calibrated simultaneous confidence band.

## What is D_side on slide 14?

D_side is **Poisson deviance summed over the sideband bins**. It measures how far their observed counts depart from the fitted GP means:

\[
D_{\mathrm{side}}=2\sum_{i\in\mathrm{side}}
\left[n_i\log\!\left(\frac{n_i}{\widehat b_i}\right)-n_i+\widehat b_i\right].
\]

“Side” means the scored bins in the 50–250 MeV search region **outside** the ±2.25σ window centered at the mass on the horizontal axis. GP training also uses the available support outside that window.

Each bracket compares the GP prediction with a perfect fit that assigns this bin exactly its observed count. Multiplication by two gives the usual likelihood-deviance convention. Every contribution is nonnegative and is zero when data equal prediction. For small discrepancies it is approximately (n_i−b̂_i)²/b̂_i: a squared count difference measured relative to Poisson variance. A prediction of 100 events and an observation of 110 contribute about **0.968** to D_side.

The plotted quantity is **D_side/N_side**, the average contribution per scored bin. N_side is a bin count, not an effective number of fit degrees of freedom. D_side is neither a signal significance nor a probability. Unlike r_i, the deviance formula has no extra GP-variance term; the conditional toy comparison repeats the fitted-mean procedure.

## What counts as a good value?

A value **typical of the refitted-toy reference at that same center** is reassuring for this particular sideband check. The conditional median is about **0.95** here, not exactly 1. Fitting uses the same sidebands that enter the score, so the expected residual size depends on how the fit adapts to those data.

The plot now covers 41 centers at 5 MeV spacing from 50 to 250 MeV, plus the previous 78 MeV anchor. At each of these 42 centers, 256 paired toy spectra supply a median and a pointwise central 90% band.

- Observed D_side/N_side: **0.8399–0.9588**.
- Conditional median: **0.9436–0.9572**.
- Every displayed value lies inside its own pointwise 90% band.

An unusually large value indicates larger sideband discrepancies than this reference usually produces. An unusually small value can motivate checks of fit flexibility or the generating source; it does not by itself prove overfitting. Smaller is not automatically better.

The centers are strongly correlated. Being inside all 42 pointwise bands is not a global goodness-of-fit probability or 42 independent successes. The source and archived kernels are fixed from observed-data-derived inputs, so source uncertainty and kernel selection are not propagated. These in-sample sideband checks also do not validate prediction in the excluded signal window. Exact results, toy-count intervals and reproduction details are in [science/findings.md](science/findings.md).
