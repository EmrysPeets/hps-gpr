# v4.1 analysis-note integration map and patch draft

This is a read-only integration draft for the sibling note tree
`/Users/emryspeets/Desktop/gp_mods/hps-gpr-analysis-note-v4-20260803`.
No file in that sibling worktree was edited.

The anchors below were inspected on August 4, 2026, on branch
`codex/analysis-note-v4-20260803` at `c3f30707981a`, with the existing dirty
v4.1 edits present. Line numbers are therefore snapshot anchors; when applying
the patch, match the quoted neighboring text as well.

## Source-tree map

- Main analysis-note driver:
  `hps_gpr_analysis_note/main.tex`.
- The separate `publication_folder/pub_main.tex` is a shorter REVTeX
  publication draft. It does not contain the v4.1 material and is not the
  target of this patch.
- `main.tex:126-139` includes the analysis-note sections in this order:
  change log, introduction, datasets, event selection, methodology, validation,
  results, conclusions, and appendices.
- `main.tex:142-143` uses classic BibTeX, not `biblatex`:

  ```tex
  \bibliographystyle{unsrt}
  \bibliography{hps_gpr_analysis_note}
  ```

  Citation keys therefore live in
  `hps_gpr_analysis_note/hps_gpr_analysis_note.bib`.
- `main.tex:61-75` defines `\graphicorplaceholder{width}{relative/path}`.
  Figure paths are relative to the `hps_gpr_analysis_note/` root; there is no
  `\graphicspath` in the analysis-note driver.
- Current v4.1 result figures use `[H]`, `\centering`,
  `\graphicorplaceholder`, a semantic caption, and a `fig:` label after the
  caption. See `sections/06_results.tex:66-77`, `106-115`, and `130-139`.
- `FIGURE_MANIFEST.md:3-21` defines asset-status language, while
  `FIGURE_MANIFEST.md:107-117` records the active v4.1 assets and explicitly
  states that no v4.1 expected-band asset exists.

## Recommended insertion points

1. Keep the observed 2016 factor scan in
   `sections/06_results.tex:57-88`. It is an observed numerical-range
   diagnostic and already has the correct physics-result caveat.
2. Put the ten-toy 2021 five-scenario source-and-exposure screen in
   `sections/05_toys_validation.tex` after line 503, between
   “Fixed-total GP-propagated toy-scan study” and
   “Asimov calibration and background-only pseudoexperiments.” This makes it
   validation material rather than a physics-result panel.
3. Cross-reference the pilot from the end of the design subsection in
   `sections/04_methodology.tex:314-320`.
4. Add one short non-promotion paragraph after
   `sections/06_results.tex:86-88`.
5. Replace the future-only outlook at
   `sections/07_conclusions.tex:34-42` with a pilot-plus-production outlook.
6. Update the v4.1 change-log row at
   `sections/00_change_log.tex:56-62`, the abstract at
   `main.tex:117-120`, and the figure manifest.

## Proposed note-local figure locations

Keep the pilot assets out of `final_limit_projection_figs/` so their directory
does not imply a physics limit or expected sensitivity:

```text
toy_generation_figs/v4p1_ls_exposure_pilot/
  fig_v4p1_ls_observed_dataset_comparison.pdf
  fig_v4p1_ls_toy_ensemble_curves.pdf
  fig_v4p1_ls_bound_sensitivity.pdf
```

PDF is the established v4.1 result convention. PNG companions may be retained
for review, but the LaTeX should cite the PDFs.

## Patch 1: abstract scope

In `hps_gpr_analysis_note/main.tex`, replace the paragraph tail at lines
117-120:

```tex
No new expected-limit bands, limit-tail ensemble diagnostics,
toy-calibrated global $p$-value, or direct coverage claim is made.
Production-matched hyperparameter closure and direct limit coverage remain required
before this candidate can replace a frozen analysis card.
```

with:

```tex
No new expected-limit bands, limit-tail ensemble diagnostics,
toy-calibrated global $p$-value, or direct coverage claim is made. A ten-toy
2021 five-scenario source-and-exposure screen is included only as a pilot
diagnostic of the planned hyperparameter-closure machinery; it is not used to
select a physics card or to qualify the observed result. Production-matched
hyperparameter closure and direct limit coverage remain required before this
candidate can replace a frozen analysis card.
```

## Patch 2: methodology cross-reference and interpretation

In `sections/04_methodology.tex`, after line 320 and before the paragraph
beginning “Figure~\ref{fig:gpr-window-construction}”, insert:

```tex
A ten-toy implementation screen of this construction is reported in
Section~\ref{sec:v4p1-ls-exposure-pilot}. Its purpose is to verify the
scaled-intensity generator, per-toy GP refit, and diagnostic export before a
production ensemble is launched. With only ten pseudoexperiments per scenario
and upper-bound setting, that screen cannot determine stable tail probabilities,
central intervals, or coverage and is not used to freeze $k_{\max}$.
```

Optionally strengthen the opening of the same subsection, after line 296, with
the primary-source wording:

```tex
This construction scales one of two declared expected Poisson intensities, not
either realized observed histogram
\cite{Kingman1993PoissonProcesses,HistFactory}. The RBF length scale is the
input-space covariance scale of the background prior, while its configured
bounds define the optimizer domain
\cite{RasmussenWilliams2006,ScikitLearnGPRDocs16}. More data can sharpen the
log-marginal-likelihood dependence on $\ell$, but it does not imply a general
monotonic law in which the fitted $\ell$ must increase with luminosity.
```

## Patch 3: ten-toy pilot subsection

In `sections/05_toys_validation.tex`, insert after line 503 and before the
current `\subsection{Asimov calibration and background-only
pseudoexperiments}`:

```tex
\subsection{Ten-toy 2021 source-and-exposure length-scale pilot}
\label{sec:v4p1-ls-exposure-pilot}

Before launching the production hyperparameter-closure ensemble, a deliberately
small screen was used to exercise the source-selection, exposure-scaling, and
refit path. Let $\lambda_i^{(1)}$ and $\lambda_i^{(10)}$ denote the independently
constructed smooth expected count intensities derived from the native 2021 1\%
and 10\% sources, respectively, on the common production support. The five
declared pilot scenarios are
\begin{align}
  \Lambda_i^{\mathrm{native}\,1\%} &= \lambda_i^{(1)}, &
  \Lambda_i^{1\%\times10} &= 10\lambda_i^{(1)}, &
  \Lambda_i^{1\%\times100} &= 100\lambda_i^{(1)}, \nonumber\\
  \Lambda_i^{\mathrm{native}\,10\%} &= \lambda_i^{(10)}, &
  \Lambda_i^{10\%\times10} &= 10\lambda_i^{(10)}, &&
\label{eq:v4p1-ls-pilot-scenarios}
\end{align}
with pseudoexperiments
\begin{equation}
  Y_i^{(t,r)}\sim\operatorname{Pois}\!\left[\Lambda_i^{(r)}\right]
  \qquad
  r\in\{
  \mathrm{native}\,1\%,\,1\%\!\times10,\,1\%\!\times100,\,
  \mathrm{native}\,10\%,\,10\%\!\times10\}.
\label{eq:v4p1-ls-pilot-poisson}
\end{equation}
Neither realized observed histogram is multiplied by a scale factor.

The native 10\% source contains 11.296 times the native 1\% support total.
It therefore contains 1.1296 times the strict $1\%\times10$ total. The
$1\%\times10$ versus native-10\% comparison is consequently a
source/selection-shape diagnostic, including the different native
normalization, rather than a pure tenfold-exposure closure test. The
$1\%\times100$ and $10\%\times10$ scenarios provide the corresponding two
100\%-scale projections from different declared truth sources.

Each pseudo-spectrum is passed through the full GP sideband refit using the
40--300~MeV support, 50--250~MeV search interval, production rebinning, and
$2.25\,\sigma_m$ training exclusion. The output retained for this pilot is the
optimized-kernel state---including $\ell_{\mathrm{opt}}/\sigma_x$,
$\ell_{\mathrm{opt}}/\ell_{\max}$, upper-bound occupancy, log marginal
likelihood, and fit status---rather than an upper-limit ensemble.

\paragraph{Pilot scope.}
There are ten pseudoexperiments per scenario and candidate upper-bound setting.
The resulting source-and-exposure curves are therefore \emph{pilot diagnostics}.
All ten toy curves within each scenario are shown directly; no central expected
band, one- or two-standard-deviation envelope, limit-tail fraction, coverage
statement, or scan-wise discovery probability is constructed from them. In
particular, an occupancy count in this screen has a granularity of one toy, or
10\%, and cannot by itself establish that a bound is nonbinding in production.

\begin{figure}[H]
\centering
\graphicorplaceholder{0.92\linewidth}{toy_generation_figs/v4p1_ls_exposure_pilot/fig_v4p1_ls_observed_dataset_comparison.pdf}
\caption{Observed optimized-length-scale diagnostics used to motivate the
source-and-exposure pilot. The curves compare the refitted length scale in units of
the local resolution and, where shown, in units of the configured upper bound.
They are data diagnostics rather than ensemble expectations. The full-2016
factor-8 saturation prompted the controlled range study in
Section~\ref{sec:results-v4p1-lsupper-study}; it does not establish a universal
luminosity dependence of the GP length scale.}
\label{fig:v4p1-ls-observed-dataset-comparison}
\end{figure}

\begin{figure}[H]
\centering
\graphicorplaceholder{0.94\linewidth}{toy_generation_figs/v4p1_ls_exposure_pilot/fig_v4p1_ls_toy_ensemble_curves.pdf}
\caption{Ten-toy 2021 source-and-exposure pilot. Thin curves show the
individual refitted hyperparameter trajectories for the native 1\%,
$1\%\times10$, $1\%\times100$, native 10\%, and $10\%\times10$
scenarios. They are projections of the
\emph{hyperparameter diagnostic} under the smooth pilot truth, not projected
limits, expected-limit bands, or a prediction of the eventual 100\% observed
spectrum. Because the native 10\% support total is 11.296 times the native 1\%
total, the $1\%\times10$ versus native-10\% comparison also probes the
source/selection shape and native normalization. The small ensemble is intended
to expose generator, refit, and optimizer-path failures before the production
closure campaign.}
\label{fig:v4p1-ls-toy-ensemble-curves}
\end{figure}

\begin{figure}[H]
\centering
\graphicorplaceholder{0.92\linewidth}{toy_generation_figs/v4p1_ls_exposure_pilot/fig_v4p1_ls_bound_sensitivity.pdf}
\caption{Upper-bound sensitivity in the same ten-toy pilot. Each candidate
$k_{\max}$ is evaluated for the same five declared scenarios and production
geometry. The panels diagnose boundary occupancy, source dependence, and
optimizer response; they
do not select the setting that gives the strongest limit or smallest local
$p_0$. A production choice requires a larger predeclared ensemble, independent
smooth truth families, repeat-fit stability, signal-injection closure, and
direct coverage after the card is frozen.}
\label{fig:v4p1-ls-bound-sensitivity}
\end{figure}

The pilot can establish that the intended exposure-scaling code path runs and
can reveal gross saturation or optimizer branching. It cannot show that
increased luminosity \emph{requires} a longer correlation scale, because the
direction of the refitted $\ell$ response depends on the truth spectrum, noise
realization, support, and competing kernel hyperparameters. Nor does the pilot
retroactively validate the 2016 factor-12 candidate, choose a 2021 upper factor,
or alter the combined observed limit in Section~\ref{sec:results}. Those are
separate physics-result and calibration decisions.

\FloatBarrier
```

If the reviewed pilot ledger omits any of the five scenarios, change
Eqs.~\eqref{eq:v4p1-ls-pilot-scenarios}--%
\eqref{eq:v4p1-ls-pilot-poisson} and the two captions. Do not silently retain
scenario labels that were not generated.

## Patch 4: keep the pilot out of the observed-result selection

In `sections/06_results.tex`, after line 88 and before
`\subsection{Exact combined observed reconstruction}`, insert:

```tex
The ten-toy five-scenario source-and-exposure screen in
Section~\ref{sec:v4p1-ls-exposure-pilot} is not part of this selection rule.
It is a later pilot of the planned 2021 hyperparameter-closure machinery and
does not supply an expected band, a coverage qualification, or a trials
correction for the observed factor scan. Consequently, none of the observed
limits or asymptotic $p_0$ values below is conditioned on a conclusion drawn
from those ten toys.
```

This is the key physics-result separation: the factor-12 observed/asymptotic
candidate remains exactly the result already documented in Section 6, while
the pilot belongs to validation planning.

## Patch 5: conclusions and outlook

In `sections/07_conclusions.tex`, replace the current future-only
hyperparameter-study paragraph at lines 34-42 with:

```tex
A ten-toy 2021 source-and-exposure pilot now exercises the expected-intensity
scaling, production-geometry GP refit, and hyperparameter export for five
declared scenarios: native 1\%, $1\%\times10$, $1\%\times100$, native 10\%,
and $10\%\times10$. Its individual curves are reported as pilot diagnostics
only. Ten toys are too few to establish stable quantiles, tail probabilities,
or coverage, and no expected-limit band is constructed from this screen. The
native 10\% support total is 11.296 times the native 1\% total, so the
$1\%\times10$ versus native-10\% comparison probes the source/selection shape
and native normalization rather than a pure exposure scaling. The pilot
therefore tests implementation and exposes gross source, boundary, or optimizer
behavior; it does not establish that luminosity must drive the preferred length
scale upward.

The production follow-up should use a predeclared larger ensemble, paired toy
identifiers within each truth-source scaling family and across upper-bound
settings, GP self-closure and independent smooth truth families, and the
production support, rebinning, and blind geometry. Its principal outputs should remain
$\ell_{\mathrm{opt}}/\sigma_x$, $\ell_{\mathrm{opt}}/\ell_{\max}$, boundary
occupancy, log marginal likelihood, repeat stability, background bias, pull,
and injected-signal recovery versus exposure. Only that larger closure study
can support a frozen numerical range; direct limit coverage remains a separate
post-freeze requirement.
```

Leave lines 44-49 in place. They already state the correct coverage and
full-2021 interpretation gates.

## Patch 6: change log

In `sections/00_change_log.tex`, replace the final two sentences of the v4.1 row
at lines 60-62:

```tex
No new
expected bands or toy-calibrated discovery claim is included; the result remains a
candidate pending production-matched closure and direct coverage. \\
```

with:

```tex
A ten-toy 2021 five-scenario source-and-exposure screen is added only as a
pilot hyperparameter diagnostic. The scenarios are native 1\%,
$1\%\times10$, $1\%\times100$, native 10\%, and $10\%\times10$. No new
expected band, limit-tail ensemble result, toy-calibrated discovery claim, or
coverage claim is included; the observed result remains a candidate pending
production-matched closure and direct coverage. \\
```

## Patch 7: figure manifest

In `FIGURE_MANIFEST.md`, add this status definition after line 21:

```markdown
- `pilot diagnostic`: small implementation-screen ensemble shown only to expose
  generator, refit, or optimizer behavior; it is not an expected band, a
  coverage result, or a physics-card freeze.
```

Then insert the following rows after the existing v4.1 rows at lines 111-113:

```markdown
| `toy_generation_figs/v4p1_ls_exposure_pilot/fig_v4p1_ls_observed_dataset_comparison.pdf` | `study_results/v4p1_2021_ls_exposure_ensembles_20260804/plots/fig_v4p1_ls_observed_dataset_comparison.pdf` | diagnostic | Observed optimized-length-scale comparison used to motivate the exposure pilot; not an ensemble expectation. |
| `toy_generation_figs/v4p1_ls_exposure_pilot/fig_v4p1_ls_toy_ensemble_curves.pdf` | `study_results/v4p1_2021_ls_exposure_ensembles_20260804/plots/fig_v4p1_ls_toy_ensemble_curves.pdf` | pilot diagnostic | All ten refitted hyperparameter curves for each of the five declared native/scaled scenarios. The native-10% versus `1%-truth x10` comparison is a source/selection-shape diagnostic because their support totals differ. No expected-limit band or coverage interval is represented. |
| `toy_generation_figs/v4p1_ls_exposure_pilot/fig_v4p1_ls_bound_sensitivity.pdf` | `study_results/v4p1_2021_ls_exposure_ensembles_20260804/plots/fig_v4p1_ls_bound_sensitivity.pdf` | pilot diagnostic | Ten-toy upper-bound and boundary-occupancy screen across the five declared scenarios, used to plan the production closure ensemble rather than freeze a physics card. |
```

Replace the sentence at lines 115-117 with:

```markdown
No v4.1 expected-limit-band or upper-limit-tail-ensemble asset exists. The
ten-toy source-and-exposure assets above are hyperparameter pilot diagnostics and
must not be presented as bands. The v4 300-toy conditional products remain
historical reference products and must not be paired with the v4.1 factor-12
observed curve.
```

## Patch 8: optional primary bibliography additions

The prose above can compile with the existing `RasmussenWilliams2006`,
`ScikitLearn`, and `HistFactory` keys if the two optional citations are omitted.
For the sharper primary-source wording, add these entries to
`hps_gpr_analysis_note/hps_gpr_analysis_note.bib`. No change to
`\bibliography{hps_gpr_analysis_note}` is needed.

```bibtex
@book{Kingman1993PoissonProcesses,
  author = {Kingman, J. F. C.},
  title = {Poisson Processes},
  publisher = {Oxford University Press},
  address = {Oxford},
  series = {Oxford Studies in Probability},
  volume = {3},
  year = {1993},
  isbn = {978-0-19-853693-2},
  doi = {10.1093/oso/9780198536932.001.0001}
}

@misc{ScikitLearnGPRDocs16,
  author = {{scikit-learn developers}},
  title = {Gaussian Processes: scikit-learn 1.6.1 Documentation},
  year = {2025},
  note = {Version 1.6.1; accessed August 4, 2026},
  url = {https://scikit-learn.org/1.6/modules/gaussian_process.html}
}
```

## Final application checks

Before applying this patch to the sibling note:

1. Confirm that the reviewed pilot generated all five declared scenarios:
   native 1%, 1%-truth x10, 1%-truth x100, native 10%, and 10%-truth x10.
2. Confirm from the pilot ledger whether toy identifiers are genuinely paired
   within each truth-source scaling family and across bound settings. Do not
   say “paired” based only on equal integer labels, and do not imply that the
   independently derived native-1% and native-10% sources are paired draws.
3. Confirm both truth sources, the 11.296 native-support-total ratio, support,
   rebinning, exclusion geometry, and candidate `k_{\max}` grid against the
   final provenance file.
4. Copy the exact finalized PDFs into the note-local paths before compilation.
5. Compile from `hps_gpr_analysis_note/main.tex`, run BibTeX if the optional
   entries are used, and visually inspect every new page.
6. Verify that the rendered pilot captions contain “ten-toy,” “pilot
   diagnostic,” and “no expected-limit band.”
7. Search the final source for any sentence implying that the pilot validates
   factor 12, predicts the full-2021 observed spectrum, or proves a monotonic
   luminosity dependence. Such language should not remain.
