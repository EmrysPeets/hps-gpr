# v4.1 analysis-note update draft

Target worktree (do not apply blindly because it is already dirty):

`/Users/emryspeets/Desktop/gp_mods/hps-gpr-analysis-note-v4-20260803`

This draft treats v4.1 as a post-v4, observed/asymptotic candidate update. It
does not call the factor-12 setting predeclared, calibrated, or frozen for a
physics interpretation. It does not attach the v4 expected bands or
upper-limit-tail pseudoexperiment diagnostics to the new observed curve.

## Note-local figure paths proposed below

Copy only the final PDF/PNG pairs into:

`hps_gpr_analysis_note/final_limit_projection_figs/v4p1_20260804_2016_lsupper/`

Use these note-local names:

- `lml_and_length_scale_boundary_occupancy.pdf`
- `combined_observed_limit_k12_vs_v4_no_bands.pdf`
- `combined_asymptotic_p0_k12_vs_v4.pdf`

If the production plot names differ, change the three `\graphicorplaceholder`
paths below and the corresponding `FIGURE_MANIFEST.md` rows together.

## `main.tex`

Change both running headers from `v4` to `v4.1`.

Replace the title and date with:

```tex
\title{\vspace{-1.0em}HPS Gaussian-Process Resonance Search Analysis Note\\
\large Version 4.1: 2016 full-statistics length-scale range and
observed-only update}
...
\date{August 4, 2026}
```

Replace the final paragraph of the abstract, beginning `Version 4
separates...`, with:

```tex
Version 4 separates the mass interval searched for a signal from the wider spectrum
used to train each GP. Version 4.1 keeps those supports and combines the full 2015
and 2016 samples with the 2021 10\% sample, but revisits one numerical range after
the v4 observed scan showed that all 142 full-2016 fits saturated the
resolution-scaled length-scale ceiling. A controlled observed-only scan of
$k_{\max,2016}=8,10,12,15,$ and 20 identifies 12 as the first nonbinding value
followed by a stable plateau. The resulting shared-$\eps^2$ observed 90\% asymptotic
\CLs{} limit and fixed-mass asymptotic $p_0$ scan are reported as a post-v4 candidate
update. No new expected-limit bands, limit-tail ensemble diagnostics,
toy-calibrated global $p$-value, or direct coverage claim is made. Production-matched
hyperparameter closure and direct limit coverage remain required before this
candidate can replace a frozen analysis card.
```

## `sections/00_change_log.tex`

Append this row after v4:

```tex
\addlinespace
v4.1 & August 4, 2026 &
Documents the post-v4 response to the full-2016 upper-length-scale saturation.
Holding the data, supports, blind geometry, lower bounds, likelihood, and asymptotic
90\% \CLs{} construction fixed, an observed-only scan of
$k_{\max,2016}=8,10,12,15,$ and 20 selects 12 as the first nonbinding value followed
by a stable factor-15 and factor-20 plateau. The combined observed limit and local
asymptotic $p_0$ scan are recomputed from exact reviewed GP states. No new expected
bands or toy-calibrated discovery claim is included; the result remains a candidate
pending production-matched hyperparameter closure and direct coverage. \\
```

Change the caption tail to:

```tex
\caption{Analysis-note version history. The v4.1 row defines the scope of the
present document. It is explicitly a post-v4 observed/asymptotic candidate update,
not a pre-unblinding kernel-card selection; earlier versions are summarized only to
make changes in inputs, validation, and statistical interpretation traceable to
reviewers.}
```

## `sections/04_methodology.tex`

Replace the paragraph beginning `The July 2026 pre-unblinding configuration...`,
the length-scale table, and the two following prose paragraphs through `presented
in Section~\ref{sec:toys}.` with:

```tex
The v4.1 observed/asymptotic candidate uses the dataset-specific factors in
Table~\ref{tab:production-lengthscale-bounds}. The lower factors retain their earlier
full-refit closure provenance; changing the 2016 upper factor does not requalify those
closure studies at full statistics.

\begin{table}[H]
\centering
\begin{tabular}{lccc}
\toprule
Dataset & $k_{\min,d}$ & $k_{\max,d}$ & v4.1 status \\
\midrule
2015 full & 1.0 & 8 & unchanged from v4 \\
2016 full & 0.9 & 12 & observed-only candidate \\
2021 10\% & 1.1 & 15 & unchanged from v4 \\
\bottomrule
\end{tabular}
\caption{Resolution-scaled RBF length-scale factors in the v4.1
observed/asymptotic candidate. The factors multiply $\sigma_x(m)$ in
Eq.~\eqref{eq:dataset-lengthscale-bounds}. Only the 2016 upper factor differs from
v4. The factor-12 row is not yet a coverage-qualified production freeze.}
\label{tab:production-lengthscale-bounds}
\end{table}

The lower factors are based on the staged full-refit functional-form closure scans
described in Section~\ref{sec:toys}, with the observed-spectrum result kept out of
their original selection criteria. For 2015, $k_{\min}=1.0$ was retained as the
conservative baseline. The 2016 development scan supported $k_{\min}=0.9$, and the
2021 development scan retained $k_{\min}=1.1$ as a calibration--reach tradeoff.
Those studies used the data fractions stated in Section~\ref{sec:toys}; they should
not be read as a new full-statistics validation of the v4.1 upper range.

Version 4 used upper factors 8, 8, and 15 for 2015, 2016, and 2021. In its reviewed
observed scan, the 2016 optimum reached its upper bound at all 142 hypotheses. The
v4.1 range study therefore changes only $k_{\max,2016}$ and evaluates factors
8, 10, 12, 15, and 20 on the same 39--180~MeV mass grid. A fit is counted as
upper-bound occupied when $\ell_{\mathrm{opt}}/\ell_{\max}\geq0.999$. The respective
occupancies are 142, 56, 0, 0, and 0 of 142. Factor 12 is the first scanned setting
with no occupied hypotheses. It is also followed by a numerical plateau: from factor
12 to 15, the largest pointwise change in the 2016 observed yield upper limit is
0.083\%, the largest $|\Delta Z|$ is 0.00255, and the log-marginal-likelihood
differences lie between $-4.5\times10^{-6}$ and $8.0\times10^{-6}$. The factor-15 to
factor-20 comparison is similarly stable. The selection is based on removal of the
artificial boundary and the next-setting plateau, not on which observed limit is
tighter or which local $p_0$ is smaller.

Because this diagnostic was prompted by the v4 observed spectrum, factor 12 is
reported as an observed/asymptotic candidate rather than a new calibrated nominal.
Production-matched functional-form closure, optimizer-repeat stability, and direct
coverage remain separate requirements.

\subsection{Planned 2021 exposure-scaled hyperparameter closure}
\label{sec:planned-2021-exposure-ls-toys}

The corresponding 2021 study should vary exposure through the expected count
intensity, not by multiplying the observed bin contents or rescaling an existing
limit. Let $\lambda_i^{10\%}$ be a validated smooth expectation for the present 2021
10\% spectrum in bin $i$. For exposure factors $s\in\{1,2,5,10\}$, define
\[
\lambda_i(s)=s\,\lambda_i^{10\%},
\qquad
Y_i^{(t,s)}\sim\operatorname{Pois}\!\left[\lambda_i(s)\right],
\]
so that $s=10$ represents 100\%-equivalent statistics. Paired toy identifiers should
be used across exposures and candidate upper-bound settings. A nested implementation
may generate independent Poisson increments between successive exposure points and
accumulate them, giving the correct Poisson marginal at each exposure while reducing
the Monte Carlo noise in exposure-to-exposure comparisons.

Each pseudo-spectrum must retrain the GP with the production 40--300~MeV support,
50--250~MeV search interval, rebin-five histogram geometry, and
$2.25\,\sigma_m$ training exclusion. The study should include GP self-closure and
independent smooth functional-form truth families, use a deliberately loose
$k_{\max}$ grid, and repeat questionable optimizer branches without changing the
toy. Per mass and exposure it should retain
$\ell_{\mathrm{opt}}/\sigma_x$, $\ell_{\mathrm{opt}}/\ell_{\max}$, boundary
occupancy, log marginal likelihood, and repeat stability, together with background
bias, pull, and injected-signal recovery diagnostics. Medians and central 68\% and
95\% intervals of these hyperparameter quantities versus exposure can then determine
whether the preferred range grows between 10\% and 100\% statistics.

This is a hyperparameter-closure study. It does not produce expected limit bands and
does not by itself establish \CLs{} coverage or a scan-wise global discovery
$p$-value. Those require distinct, production-faithful ensembles after a candidate
card is frozen. The current convenience toy path must also be made production
matched---in particular, it must preserve rebin five and expose the luminosity scale
rather than silently using its present fixed geometry---before its output can support
that decision.
```

Also change the last sentence immediately before this replacement if it still says
`No production-matched ... result is used to select the v3 baseline.` to
`... used to qualify the v4.1 candidate.`

## `sections/06_results.tex`

Replace the file with the following. The old labels are retained as aliases because
`sections/05_toys_validation.tex` and
`sections/appendix_prior_validation_results.tex` currently reference them.

```tex
% These summaries are generated from the reviewed 232-row v4.1 candidate table.
\providecommand{\VFourPOneLocalPZeroMinimum}{\ensuremath{3.2592\times10^{-5}}}
\providecommand{\VFourPOneLocalPZeroMass}{\SI{65}{MeV}}
\providecommand{\VFourPOneLocalZ}{\ensuremath{3.993}}
\providecommand{\VFourPOneEffectiveTrials}{\ensuremath{35.381}}
\providecommand{\VFourPOneSidakValue}{\ensuremath{0.0011525}}
\providecommand{\VFourPOneSidakZ}{\ensuremath{3.048}}

\section{Results}
\label{sec:results}

Version 4.1 reports a controlled post-v4 observed/asymptotic candidate update. The
full 2015 and 2016 samples and the reviewed 2021 10\% sample are combined in one
shared-$\eps^2$ likelihood. Relative to v4, only the 2016
resolution-scaled length-scale upper factor changes, from 8 to 12; the output
directory changes to keep the products isolated. The data, mass grids, fit supports,
blind and training-exclusion geometry, lower factors, likelihood, and 90\%
asymptotic \CLs{} construction are unchanged.

The reported products are the observed 90\% \CLs{} upper limit and fixed-mass
asymptotic $p_0$ scan. No v4.1 expected-limit bands or upper-limit-tail ensemble
diagnostics have been generated. In particular, the conditional v4 bands are not
recentered on or otherwise assigned to the new observed curve.

\subsection{Declared v4.1 candidate state}
\label{sec:results-v4p1-analysis-state}

\begin{table}[H]
\centering
\small
{\setlength{\tabcolsep}{5pt}
\begin{tabular}{lccccc}
\toprule
Campaign & Sample & Search [MeV] & GP support [MeV] &
$k_{\min}$ & $k_{\max}$ \\
\midrule
2015 & full & 19--90  & 14--135 & 1.0 & 8 \\
2016 & full & 39--180 & 30--210 & 0.9 & 12 \\
2021 & 10\% & 50--250 & 40--300 & 1.1 & 15 \\
\bottomrule
\end{tabular}
}
\caption{Data samples, search intervals, GP fit supports, and
resolution-scaled length-scale factors for the v4.1 observed/asymptotic candidate.
Only $k_{\max,2016}$ differs from v4. Because that change follows the observed v4
boundary diagnostic, this table must not be described as a pre-unblinding freeze.}
\label{tab:results-v4p1-configuration}
\end{table}

At a mass covered by more than one campaign, the simultaneous likelihood profiles
one common $\eps^2$. Campaign-specific luminosity, radiative fraction, resolution,
efficiency, and branching inputs map that parameter to separate expected signal
yields. The curve is therefore a direct shared-coupling result rather than a
pointwise envelope or luminosity-scaled earlier limit. The minimal-visible branching
correction above the dimuon threshold is applied consistently to v4 and v4.1.

\subsection{Controlled 2016 upper-range study}
\label{sec:results-v4p1-lsupper-study}
\label{sec:results-v4-support-audit}

The study evaluates $k_{\max,2016}=8,10,12,15,$ and 20 on the same 142 hypotheses
from 39 to 180~MeV. A state is classified as upper-bound occupied when
$\ell_{\mathrm{opt}}/\ell_{\max}\geq0.999$. The occupancy falls from 142 of 142 at
factor 8 to 56 of 142 at factor 10 and zero at factors 12, 15, and 20.

\begin{figure}[H]
\centering
\graphicorplaceholder{0.92\linewidth}{final_limit_projection_figs/v4p1_20260804_2016_lsupper/lml_and_length_scale_boundary_occupancy.pdf}
\caption{Controlled full-2016 upper-length-scale study. The panels show the optimized
length scale relative to the local resolution and ceiling, together with the
log-marginal-likelihood response for factors 8, 10, 12, 15, and 20. Factor 12 is the
first scanned value with zero upper-bound occupancy and is followed by a stable
factor-15 and factor-20 plateau. The scan uses observed data, so it diagnoses a
numerical-range restriction; it is not a coverage or expected-sensitivity
optimization.}
\label{fig:results-v4p1-lsupper-study}
\end{figure}

Questionable branches were rerun with the data and card held fixed. Across the four
new factor scans, 14 selected repair rows use the largest log marginal likelihood
among actual repeated fits, and each selected row is independently reproduced. No
limit, significance, or GP coordinate is interpolated. From factor 12 to 15, the
largest pointwise change in the 2016 observed yield upper limit is 0.083\%, the
largest $|\Delta Z|$ is 0.00255, and the log-marginal-likelihood differences lie in
$[-4.5,8.0]\times10^{-6}$. The factor-15 to factor-20 comparison is similarly
stable. These results define factor 12 as the first nonbinding plateau setting.
Neither the direction of the observed-limit change nor the local $p_0$ values enter
that rule.

\subsection{Exact combined observed reconstruction}
\label{sec:results-v4p1-reconstruction}

The reviewed state ledger contains exactly 415 campaign--mass GP states: 72 for
2015, 142 for 2016, and 201 for 2021. The final runner reconstructs the GP mean and
covariance at those exact optimized coordinates and evaluates the shared
\texttt{count\_scale} likelihood on the union of 232 mass hypotheses. All 232
observed limits and all 232 local asymptotic $p_0$ values are finite. Cached and
reference implementations give bitwise-identical observed limits at 20, 40, and
60~MeV, which exercise one-, two-, and three-campaign likelihoods. The runner draws
zero toys and has no expected-band output path.

\subsection{Combined observed limit without expected bands}
\label{sec:results-v4p1-combined-limit}
\label{sec:results-v4-combined-limit}

\begin{figure}[H]
\centering
\graphicorplaceholder{0.92\linewidth}{final_limit_projection_figs/v4p1_20260804_2016_lsupper/combined_observed_limit_k12_vs_v4_no_bands.pdf}
\caption{Combined observed 90\% asymptotic \CLs{} upper limit on $\eps^2$ for the
v4.1 factor-12 candidate, compared at matched masses with the v4 factor-8 result.
The lower panel shows the v4.1-to-v4 ratio. Values below unity are tighter and
values above unity are weaker. Both curves use the minimal-visible interpretation;
no expected-limit band is shown or implied.}
\label{fig:results-v4p1-combined-limit}
\end{figure}

Only the 142 hypotheses at which 2016 is active change. Over 39--180~MeV, the
v4.1-to-v4 observed-limit ratio has a minimum of 0.6941 at 103~MeV, corresponding
to a 30.59\% tightening, and a maximum of 1.2789 at 90~MeV, corresponding to a
27.89\% weakening. Its median is 0.9933. The response is therefore neither a
uniform sensitivity gain nor a luminosity rescaling. It records how removing an
active numerical ceiling changes the observed background interpolation and
conditional uncertainty. The comparison was not used to select whichever curve was
tighter.

\subsection{Local asymptotic p-values}
\label{sec:results-v4p1-local-p0}
\label{sec:results-v4-local-p0}

\begin{figure}[H]
\centering
\graphicorplaceholder{0.90\linewidth}{final_limit_projection_figs/v4p1_20260804_2016_lsupper/combined_asymptotic_p0_k12_vs_v4.pdf}
\caption{Combined fixed-mass asymptotic $p_0$ values for the v4.1 factor-12 candidate
and the v4 factor-8 result. The associated analytic \v{S}id\'ak reference, if shown,
uses the same resolution-spacing effective-trials estimate as v4. It corrects only
the mass scan for a fixed card; it does not account for the post-v4 upper-bound
study and is not a toy-calibrated global $p$-value.}
\label{fig:results-v4p1-local-p0}
\end{figure}

The smallest v4.1 fixed-mass asymptotic value is
\VFourPOneLocalPZeroMinimum{} at \VFourPOneLocalPZeroMass, corresponding to
$Z_{\mathrm{local}}=\VFourPOneLocalZ$. For comparison, the v4 minimum was
$1.7636\times10^{-4}$ at 66~MeV ($Z_{\mathrm{local}}=3.573$). Applying the unchanged
resolution-spacing estimate $N_{\mathrm{eff}}=\VFourPOneEffectiveTrials$ to the new
minimum gives the conditional analytic reference
$p_{\mathrm{Sidak}}=\VFourPOneSidakValue$
($Z_{\mathrm{Sidak}}=\VFourPOneSidakZ$). This number is not a global discovery
significance: it neither includes the observed-data upper-bound scan nor replaces a
scan-wise maximum-$q_0$ ensemble.

\subsection{Relation to v4 and validation boundary}
\label{sec:results-v4p1-v4-context}

Version 4 remains the frozen reference for this diagnostic. Its factor-8 saturation,
observed limit, local $p_0$ scan, and conditional 300-toy limit ensemble motivated or
contextualize the present study, but those conditional bands and their strong-, weak-,
and bounded two-sided tail fractions are not v4.1 outputs. They are not mixed with
the factor-12 curve. The factor-12 result is an observed/asymptotic candidate until
production-matched hyperparameter closure and direct coverage are complete.
```

## `sections/07_conclusions.tex`

Replace the file with:

```tex
\section{Conclusions and Outlook}
\label{sec:conclusions}

Version 4.1 documents a controlled response to a numerical-range diagnostic in the
v4 observed scan. The search intervals and wider GP supports remain 19--90 and
14--135~MeV for 2015, 39--180 and 30--210~MeV for 2016, and 50--250 and
40--300~MeV for 2021. The full 2015 and 2016 samples and the 2021 10\% sample are
combined in a shared-$\eps^2$ likelihood. Only the 2016 resolution-scaled
length-scale upper factor changes, from 8 to 12.

That change follows a controlled factor-8, 10, 12, 15, and 20 observed-only scan.
The v4 ceiling is occupied at all 142 full-2016 hypotheses; factor 10 remains occupied
at 56, while factors 12, 15, and 20 have no occupied hypotheses. Factor 12 is selected
as the first nonbinding value and is followed by a stable plateau in log marginal
likelihood, observed yield limits, and local $Z$. Repeated actual fits resolve the
reviewed branches, and no GP state, limit, or $p$-value is interpolated. The rule
does not select whichever observed limit is tighter or whichever local $p_0$ is
smaller.

The exact 415-state reconstruction gives finite observed 90\% asymptotic \CLs{}
limits and fixed-mass asymptotic $p_0$ values on all 232 combined hypotheses. Relative
to v4, the observed-limit ratio over the 2016-active interval ranges from 0.6941 at
103~MeV to 1.2789 at 90~MeV. The smallest local asymptotic value is
$3.2592\times10^{-5}$ at 65~MeV ($Z=3.993$). The corresponding fixed-card analytic
\v{S}id\'ak reference is not a calibrated global $p$-value because it does not
account for the post-v4 upper-bound scan.

No v4.1 expected-limit bands or upper-limit-tail pseudoexperiment diagnostics have
been made. The v4 conditional bands are not transferred to the new observed curve.
Likewise, no scan-wise toy-calibrated discovery probability or direct \CLs{} coverage
claim is inferred. Version 4.1 is therefore an observed/asymptotic candidate update,
not yet a replacement for a coverage-qualified frozen card.

The next hyperparameter study should use 2021 pseudo-spectra whose expected
intensities are scaled from the validated 10\% expectation through 20\%, 50\%, and
100\%-equivalent statistics, with paired toy identifiers across exposures and
upper-bound settings. Each pseudo-spectrum should retrain the GP with production
support, rebinning, and blind geometry. The principal outputs are the distributions
of $\ell_{\mathrm{opt}}/\sigma_x$, $\ell_{\mathrm{opt}}/\ell_{\max}$, boundary
occupancy, log marginal likelihood, repeat stability, and closure diagnostics versus
exposure. Those toys determine whether the admissible length-scale range should grow
with statistics; they are not expected-limit-band toys.

After a candidate upper range is selected without using observed limit direction or
local significance, production-faithful functional-form closure and direct limit
coverage should be rerun under that frozen card. A separate scan-wise
background-only maximum-$q_0$ ensemble would be required for a toy-calibrated global
discovery statement. The eventual full-2021 observed spectrum remains a new analysis
input, not a deterministic scale factor applied to the present 10\% observation.
```

## `FIGURE_MANIFEST.md`

Add this status definition:

```md
- `observed-only candidate`: exact observed/asymptotic product from the v4.1 card;
  it has no associated expected-limit band and is not yet coverage-qualified.
```

Append this section near the active Results figures:

```md
## v4.1 2016 length-scale upper-range and observed-only figures

| Bundle path | Original source | Status | Notes |
| --- | --- | --- | --- |
| `final_limit_projection_figs/v4p1_20260804_2016_lsupper/lml_and_length_scale_boundary_occupancy.pdf` | `study_results/v4p1_2016_ls_upper_optimization_20260804/plots/lml_and_length_scale_boundary_occupancy.pdf` | diagnostic | Controlled 2016 factor-8, 10, 12, 15, and 20 boundary-occupancy and log-marginal-likelihood scan. Factor 12 is the first nonbinding setting followed by the factor-15 and factor-20 plateau. |
| `final_limit_projection_figs/v4p1_20260804_2016_lsupper/combined_observed_limit_k12_vs_v4_no_bands.pdf` | v4.1 final observed comparison export from the exact 232-row factor-12 reconstruction and the reviewed v4 factor-8 table | observed-only candidate | Matched-mass shared-`\eps^2` observed 90% asymptotic `\CLs` comparison and ratio. No expected bands are shown or implied. |
| `final_limit_projection_figs/v4p1_20260804_2016_lsupper/combined_asymptotic_p0_k12_vs_v4.pdf` | v4.1 final p-value comparison export from the exact 232-row factor-12 reconstruction and the reviewed v4 factor-8 table | observed-only candidate | Fixed-mass asymptotic `p_0` comparison. Any Sidak curve is an analytic fixed-card reference, not a toy-calibrated global probability and not corrected for the upper-bound scan. |
```

Add a manifest note:

```md
- No v4.1 expected-limit-band or upper-limit-tail-ensemble asset exists. The v4
  300-toy conditional products remain historical reference products and must not be
  paired with the v4.1 factor-12 observed curve.
```

## Two optional caller cleanups outside the requested file list

The Results replacement retains old label aliases, so these are not required for a
successful build, but they avoid stale prose:

- In `sections/05_toys_validation.tex`, change `The current physics-facing
  simultaneous limit` to `The current observed-only simultaneous candidate limit`,
  and change `rather than a v4 result` to `rather than a v4.1 result`.
- In `sections/appendix_prior_validation_results.tex`, change `the v4 discovery
  diagnostic is shown separately` to `the v4.1 observed/asymptotic candidate
  diagnostic is shown separately`.

## LaTeX and review risks

1. `sections/05_toys_validation.tex` references
   `sec:results-v4-combined-limit`, and
   `sections/appendix_prior_validation_results.tex` references
   `sec:results-v4-local-p0`. The replacement Results text intentionally retains
   those two labels as aliases. Removing the aliases requires updating both callers.
2. Adding the v4.1 row may make the `[H]` `tabularx` change-log table too tall for
   one page. If rendering shows an overfull page, first shorten the v4.1 row or use
   `\footnotesize`; if it still does not fit, convert the version table to
   `longtable` rather than shrinking it to unreadable text.
3. The methodology subsection currently describes v3-era sample fractions and
   upper factors as if they were current. The replacement must land as one block;
   changing only the table leaves contradictory prose (`upper factors remain ... 9
   for 2021`) immediately below it.
4. Do not reuse `fig:results-v4-combined-band` or
   `fig:results-v4-limit-tail` for v4.1. Their names and captions assert bands/toy
   tails that were not rerun.
5. Escape `%` as `\%`, use `\texttt{count\_scale}`, and retain the existing
   `\CLs`, `\eps`, `siunitx`, and `graphicorplaceholder` conventions.
6. The p-value comparison is materially affected by the post-v4 range scan. The
   analytic Sidak number corrects only the mass scan conditional on a fixed card.
   It must not appear under a legend or caption labeled `global p-value`.
7. The note-local figures should be copied before building. The placeholder macro
   otherwise allows a seemingly successful PDF build with missing result graphics;
   audit the build log and visually inspect the affected pages.
8. After building, search the extracted PDF text for stale headline claims such as
   `Version 4 reports`, `principal v4 result`, `expected median`, `0 of 300`,
   `upper factors remain 8`, and `2021 1%` in the current-card table. Historical
   occurrences in the change log or appendices are acceptable only when explicitly
   labeled historical.
