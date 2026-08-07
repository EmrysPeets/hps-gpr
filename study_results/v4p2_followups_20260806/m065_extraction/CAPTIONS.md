# Figure captions

## `figure61_common_0p5MeV`

Observed extraction at 65 MeV after an exact, count-preserving common-bin refit.
The 2015, 2016, and 2021 histograms all use 0.5 MeV bins, obtained by integer
aggregation factors 10, 10, and 4 from their source histograms. The top panels
show observed data, the newly optimized sideband-trained GP mean and predictive
uncertainty, and display extensions of the standalone and simultaneous
shared-$\epsilon^2$ signals. The lower panels show data minus the pre-profile
GP mean with the original Figure 61 display uncertainty
$\sqrt{\mu_{\rm GP}+C_{ii}}$. This is a prior-predictive,
correlated-background diagnostic; the bars are not independent-bin post-fit
errors. This is a local binning-robustness result, not a replacement for the
accepted native v4.2 scan.

## `figure61_common_0p5MeV_profiled`

Count-space profiled extraction for the same exact 0.5 MeV refit, restricted to
the actual $\pm2.25\sigma_m$ likelihood window. Blue shows the background-only
profile, orange the standalone signal-plus-background profile, and red the
simultaneous shared-$\epsilon^2$ signal-plus-background profile. The lower
panels show count residuals relative to each profiled expectation. These are
correlated fit diagnostics, not standardized per-bin significances. The curves
are not extended into sidebands because the v4.2 nuisance likelihood profiles
the GP background only in the extraction window.

## `figure62_profiled_residuals_physical68`

Corrected native-v4.2 Figure 62 composite. The three dataset panels retain the
conditional Pearson residuals for the background-only, standalone
signal-plus-background, and simultaneous shared-$\epsilon^2$
signal-plus-background profiles in the exact $\pm2.25\sigma_m$ likelihood
windows. These correlated fit diagnostics are not independent local
significances. The lower-right panel replaces the signed symmetric-Wald
display with the physical $\epsilon^2\geq0$ profile-likelihood sets defined
by $\Delta(-2\ln L)=1$. The 68% interpretation is nominal and asymptotic,
not coverage calibrated; the native 2016 lower endpoint is zero.

## `figure62_coefficients_physical68`

Physical-domain 65 MeV signal-strength estimates for the authoritative native
v4.2 extraction and the exact common-0.5-MeV refit. Horizontal intervals are
the $\epsilon^2\geq0$ profile-likelihood sets defined by
$\Delta(-2\ln L)=1$; their 68% interpretation is nominal and asymptotic, not
coverage calibrated. The native 2016 lower endpoint is zero. In the original
Figure 62, the 2016 signed estimator was
$(1.51921\pm2.55749)
\times10^{-6}$; its symmetric Wald extension reached
$-1.03828\times10^{-6}$. That extension was an
unconstrained estimator uncertainty, not a negative physical coupling and not
a conversion error.

## Numerical scope

At 65 MeV the native combined reconstruction gives
$Z_{\rm local}=3.99321$ and
$\widehat{\epsilon^2}=6.42706\times10^{-6}$.
The exact common-0.5-MeV refit gives
$Z_{\rm local}=3.85429$ and
$\widehat{\epsilon^2}=6.20719\times10^{-6}$.
These are fixed-mass asymptotic profile-likelihood quantities. No scan-wide
minimum or global significance was recomputed in this local study.
