"""Readable v6.3.6 narrative built from frozen v6.3.5 tables; no fitting."""
from pathlib import Path

def build_document(B,pic,table,sub,f,NAME,h,c,l,o,op,j,je,w,wm,wa,summary):
 def page(title,body):return '\\clearpage\\section*{'+title+'}\n'+body
 def caption(text):return r'\textbf{How to read the figure.} '+text
 err=r'''\textbf{Uncertainty.} Means have sample-standard-error bars: the sample standard deviation divided by $\sqrt{100}$. Bootstrap calculations resample whole toy experiments 2,000 times, keeping masses, injected strengths and analysis choices together; the two background sources use separate streams. Widths have bootstrap uncertainty. Fractions retain all 100 trials and use exact two-sided 95\% Clopper--Pearson intervals. The caption specifies which of these bars appears here.'''
 mean=r'''Each point averages 100 independent evaluation experiments. Bars show sample standard errors, conditional on the frozen calibration table; uncertainty from estimating that table is assessed separately.'''
 doc=r'''\documentclass[11pt]{article}
\usepackage[margin=.73in]{geometry}
\usepackage{lmodern,amsmath,amssymb,booktabs,graphicx,microtype,fancyhdr,xurl,hyperref}
\hypersetup{colorlinks=true,urlcolor=blue,linkcolor=black}
\pagestyle{fancy}\fancyhf{}\lhead{HPS GPR: understanding the 2021 signal extraction}\rhead{v6.3.6}\cfoot{\thepage}
\setlength{\headheight}{14pt}\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}\setlength{\emergencystretch}{2em}
\newcommand{\Ah}{\widehat A}\newcommand{\sh}{\widehat\sigma}
\begin{document}
{\LARGE\bfseries Understanding signal extraction\par from the 2021 10\% sample}\par
{\large HPS Gaussian-process background analysis}\par
24 September 2026\hfill Version 6.3.6

\textbf{The result.} Moving the Gaussian signal template toward the reconstructed signal-MC core improves the fitted response. The Gaussian still represents only part of the asymmetric signal distribution. To make statements about the \emph{full selected signal yield}, the study therefore calibrates the fit statistic using simulated experiments with a known injected signal yield.

This note explains that procedure and the evidence supporting it. It is a presentation revision of v6.3.5: the main-body numerical ensembles, fitted results and observed scans are unchanged. Appendices A--C add v16 TC/UC core-shift studies; D--H add the v6.3.7 signal-template and window studies; I isolates signal leakage and its effect on the fit. The signal-MC catalogue and core-shift comparison from v6.1 are included so that the choice of Gaussian center can be understood here without consulting another note.

\textbf{Why the distinction matters.} There are three separate questions. Where is the reconstructed signal core? How much does the fitted Gaussian yield change when signal is added? Which injected yields are compatible with the observed fit result? A shifted Gaussian addresses the first question, an injection study measures the second, and toy-calibrated tests answer the third. Moving the Gaussian alone does not solve all three.

For signal-MC injections on the GP mean background source at $A=3s_0$ (where $s_0$ is the yield-error scale from background-only pilot fits), the added-signal response ranges from 0.208--0.422 for the central mass Gaussian and 0.265--0.833 for the shifted Gaussian across 60--240 MeV. A response of 0.5 means that adding 1,000 selected signal events increases the fitted Gaussian yield by about 500 events. It is not a detector efficiency. When the injected signal is itself a matched Gaussian, the shifted fit returns 0.963--0.980 of the added yield.

\textbf{What is usable now.} The finite-grid toy calibration provides a defined, testable procedure under the specified background sources and signal-MC shape. At $A=5s_0$, independent evaluation experiments retain the true injected yield in 81--96\% of cases for the GP mean source and 84--93\% for the functional form source. Each fraction uses 100 experiments. These are conditional validation results with finite statistical precision, rather than a guarantee of exactly 90\% coverage at every mass or for an unknown background shape.

\textbf{Reading guide.} Sections 1--2 explain the signal shape and fit. Sections 3--7 distinguish background-only bias, signal response and yield calibration. Section 8 explains the upper-limit construction. Sections 9--11 show the observed 2021 and combined results. Section 12 checks the cost of changing the fit window. Section 13 records the scope and reproducible sources.

The main search and calibration domain is 60--240 MeV. The 260 MeV signal-MC sample is shown only as a shape comparison. The report does not establish a global discovery probability or a detector-qualified physical exclusion.
'''
 doc+=page('1. Why move the Gaussian center?',r'''
A narrow generated resonance need not reconstruct as a symmetric Gaussian centered at its generated mass. Detector response and selection change the reconstructed distribution. In the supplied target-constrained (TC) signal Monte Carlo, the fitted core is displaced below the generated signal mass and a substantial tail extends toward higher reconstructed masses.

We compare two Gaussian extraction templates. The \textbf{central mass Gaussian} is centered at the tested signal mass $m$. The \textbf{shifted Gaussian} is centered at the empirical signal-core position $c(m)$:
\[
 c(m)=m-3.2243308693-2.2139928114\ln(m/150),\qquad 60\le m\le240\ \mathrm{MeV}.
\]
Here $m$ and $c(m)$ are in MeV; $m/150$ means $m/(150\,\mathrm{MeV})$. This is a logarithmic relation for the \emph{core displacement}, not a logarithmic probability model for the entire signal distribution.
'''+pic('intro_core_shift_comparison',caption(r'''Left: the v6.1 TC signal-MC core displacement and the logarithmic relation used in this study. Left error bars are the standard deviation of the fitted center over 32 signal-MC bin-resampling replicas, often smaller than the markers; they are not the 100-toy mean errors used later. Right: the HPS Internal image supplied with this revision, reproduced unchanged. Its vertical coordinate is fitted Gaussian mean minus generated mass; negative values indicate a lower reconstructed center. The blue and red curves are the two sample categories named in that image. Its error definition and reconstruction/selection equivalence to the present input were not supplied, so the right panel provides qualitative context, not a combined fit. The logarithmic curve on the left is the one used here.'''),height='3.3in')+r'''
Only the Gaussian center changes between the main fit choices. Both retain the same \textbf{reference analysis resolution},
\[
 \sigma_{\rm ref}=0.00184825-0.001375m+0.085875m^2,
 \quad m\ \hbox{and}\ \sigma_{\rm ref}\ \hbox{in GeV}.
\]
The signal-fit window and the region excluded from background training are both centered on that Gaussian and extend to $\pm2.25\sigma_{\rm ref}$. The fitted signal-MC core width used to compare shapes on the next pages is a different quantity.
''')
 doc+=page('1. Signal-MC catalogue: examine each sample first',pic('intro_signal_mc_catalogue',caption(r'''Each panel shows one generated signal-MC sample with the central mass Gaussian and the shifted Gaussian used for extraction. Both Gaussians have the reference analysis width, not the fitted signal-MC core width. The logarithmic vertical scale exposes tails over several orders of magnitude. These are signal-MC probability densities, not background toys; their full normalization is retained when only a local mass range is shown. The 260 MeV sample is a shape control outside the 60--240 MeV calibration domain and has no logarithmically shifted curve. No new fit was performed for this revision.'''),height='6.65in')+r'''
The catalogue makes two features visible together: the central peaks move relative to their generated masses, and the selected signal is not fully Gaussian. An extraction template can locate the core well while still returning less than the full selected signal yield.
''')
 doc+=page('1. Can one shape describe the signal cores?',pic('intro_common_core_shape',caption(r'''The signal-MC distributions are aligned by their individual fitted core centers and widths. The horizontal coordinate is $u=(m_{\rm rec}-c_{\rm core})/\sigma_{\rm core}$, so a unit of $u$ is one fitted core width. The comparison tests whether the central shapes look similar after alignment; it does not imply that their full tails or acceptances are identical. Left: each curve is normalized within $|u|<2$ to compare core shapes. Right: the original core and tail probabilities are retained. The dashed black common shape is the equal-mass average of the ten aligned empirical distributions, not an analytic Gaussian fit. The gray dotted reference is a standard Gaussian normalized within $|u|<2$. These curves reuse the saved v6.1 histograms and core fits; no toy error bars are shown.'''),height='4.6in')+r'''
\textbf{Answer.} A common empirical shape describes the cores reasonably well from about 80 MeV upward. Its peak is approximately Gaussian, with heavier shoulders than a standard Gaussian. The tails remain visibly different, and the low-mass sample needs separate treatment. Thus a common core description is a useful guide to the Gaussian center, but is insufficient to calibrate the total signal yield.

\textbf{The low-mass qualification.} The supplied independent signal-MC samples are generated at 60, 80, 100, \ldots, 260 MeV. There is no independently simulated 65 MeV sample in this catalogue. The fitted core of the 60 MeV sample is about 58.8 MeV. The acceptance-loss concern near 65 MeV is therefore illustrated here by the actual 60 MeV input, not by relabeling a sample. Its distorted distribution is consistent with the low-mass geometric acceptance turn-on discussed in the signal study; this plot does not isolate the cause. These selected distributions alone do not measure a separate acceptance value at 65 MeV.

The saved v6.1 diagnostic puts 32.39\% of its stored-range probability within two fitted core widths, compared with 60.73--71.47\% for 80--240 MeV. The figure also retains the tiny overflow probability (below 0.016\% in these samples). This helps explain why the 60 MeV response is particularly small in the injection study: fitting its central region with a Gaussian cannot account for the large fraction of selected signal outside that region. Subsequent sections measure that effect directly rather than assuming it away.
''')
 doc+=page('2. What the Gaussian fit measures',pic('shapes',caption(r'''Black: the selected signal-MC distribution used to inject events. Red: the central mass Gaussian. Blue: the shifted Gaussian, also called the \emph{log core Gaussian} because its center follows the logarithmic law in Section 1. Both Gaussians use the same reference analysis resolution. All curves retain their full probability normalization; the displayed mass range omits some tails without renormalizing what remains. These are the same extraction prescriptions as in the catalogue; the linear vertical scale emphasizes the core mismatch.'''),height='4.35in')+r'''
Let $A$ be the expected number of \textbf{full selected signal events}. This counts the whole selected signal distribution, including events outside the analysis support and fit window. Let $\Ah$ be the signed Gaussian yield returned by the fit, and $\sh$ its reported standard error. Negative fitted yields are allowed so that background fluctuations are treated symmetrically; expected bin counts must remain positive.

The background is estimated with Gaussian-process regression (GPR) from the sidebands, meaning the bins outside the excluded signal region. Each toy or observed spectrum supplies its own sideband counts. The GP kernel settings remain fixed, but its predicted background mean and covariance are recomputed for that spectrum. The likelihood uses Poisson bin counts and correlated background nuisance parameters; the error $\sh$ comes from the curvature of the fitted profile likelihood.

Two effects reduce $\Ah$ relative to the full signal yield. First, the Gaussian fits only a projection of an asymmetric signal distribution. Second, signal in the training sidebands can be absorbed into the estimated background. With the background known exactly, the separate window study still finds response 0.378--0.857. The corresponding GP response divided by that control is 0.703--0.974. Moving the center helps, but neither effect is removed automatically.
''')
 doc+=page('2. How the simulated experiments are organized',r'''
A \textbf{toy experiment} is a simulated binned count spectrum. Background counts are drawn independently in each bin from Poisson distributions with fixed expected counts. We use two background sources:

\textbf{GP mean background source.} The expected bin counts are the pinned full-support GP arithmetic mean. We fluctuate counts around that fixed model; we do not draw a new GP background function for each toy.

\textbf{Functional form background source.} The expected bin counts come from the archived anchored \texttt{fSigPowExpQ} function. This alternative tests sensitivity to one specified change in background shape. Its earlier local assessment does not establish that it describes every part of the data globally.

Signal events are added according to the fixed selected signal-MC probabilities, including training sidebands and outside-support categories. At fixed expected signal yield $A$, their total fluctuates according to a Poisson distribution. These are \emph{not} injections with an exactly fixed event total. Neither the signal MC nor the Gaussian is renormalized inside the fit window.

\textbf{Three separate sets of experiments.} First, 100 GP-source background-only \emph{pilot} experiments define $s_0(m)$, the mean reported yield error from the shifted Gaussian. This fixes the signal-strength unit $A=z s_0$. The label $z$ therefore specifies injected yield in pilot-error units; it is not a discovery significance.

Next, each source has 100 new \emph{calibration} backgrounds. These determine the background-only bias, the response, and the reference distributions used by the yield tests. Those choices are frozen. Finally, each source has another 100 \emph{independent evaluation} backgrounds to test the frozen calibration. These evaluation experiments were called ``heldout'' in earlier versions: they were simply not used to determine the calibration.

Calibration uses $z=0,1,2,3,4,5,6,8,10,12,16,20,24$; evaluation uses $z=0,1,3,5$. Gaussian signal controls use $z=1,3,5$ on the GP mean source. The same expected yield is used for both Gaussian centers and both background sources.

Within a source and cohort, the same background experiment is reused across masses, strengths and fit choices. The two Gaussian fits also receive identical signal counts. Such comparisons are paired: their differences remove common fluctuations. Different sources and cohorts have independent streams. There are 500 main background experiments, with many fits to each, not 76,000 independent experiments.
'''+table(['Main calculation','Attempted','Valid'],[['Pilot fits','2,000','2,000'],['Calibration fits','52,000','52,000'],['Evaluation fits / truth profiles','22,000 / 22,000','22,000 / 22,000'],['Gaussian asymptotic limit endpoints','16,000','16,000']])+err)
 doc+=page('3. Measure background-only bias before adding signal',pic('null',caption(r'''Both columns are background only ($A=0$). Left: calibration mean pull $\mu_0=\overline{\Ah_0/\sh_0}$. Right: mean of $\Ah_0/\sh_0-\mu_0$ in a separate evaluation cohort. Top: GP mean background source; bottom: functional form background source. Bars are sample standard errors of means from 100 experiments. The right-hand bars condition on the frozen left-hand calibration estimate. The uncertainty conventions are also printed on the figure.'''),height='4.6in')+r'''
A background-only fit can return a nonzero average signal yield even though no signal was injected. We call this the \textbf{background-only fitted-yield bias}, $\delta=\overline{\Ah_0}$, measured in events. The \textbf{background-only mean pull}, $\mu_0=\overline{\Ah_0/\sh_0}$, instead measures displacement in units of the fit's own error. They are different averages and are not interchangeable.

Subtracting the frozen $\mu_0$ centers a pull diagnostic. Subtracting $\delta$ corrects an additive yield bias. Neither operation restores a missing fraction of injected signal. Nor does subtracting a mean change the width of a distribution.

For the shifted Gaussian, independent background-only mean pulls after centering range from $-0.174$ to $0.227$ for the GP mean source and $-0.187$ to $0.160$ for the functional form source. The corresponding uncentered pull widths are 0.832--1.131 and 0.893--1.121. Finite calibration and evaluation samples leave visible fluctuations; zero mean and unit width are targets to check, not assumptions to impose.
''')
 doc+=page('4. Signal response before and after calibration',pic('recovery_nominal',caption(r'''Signal-MC injections on the GP mean background source. Columns use $A=z s_0$ with $z=1,3,5$. Top: fitted-to-injected yield ratio $\overline{\Ah/A}$. Middle: change caused by adding signal, $\overline{(\Ah_{s+b}-\Ah_b)/A}$, with the same background experiment in both fits. Bottom: calibrated response $\overline{A_c/A}$, where $A_c=(\Ah-\delta)/R$. The independent calibration cohort supplies $\delta$ and the response $R$ measured at $z=3$. A value of one means that the estimator returns the injected full selected yield on average. '''+mean),height='5.85in')+r'''
The middle row isolates response to the added signal; the top row also contains background-only yield bias. The bottom row answers the requested calibration question directly. It applies the same frozen correction at all three strengths and tests it on new experiments. Its bars describe the evaluation mean for that frozen correction; they do not claim exact response under a different background source or signal shape.

A fixed subtraction of $\delta$ cancels in the middle-row difference, so it cannot repair a response below one. Dividing by $R$ addresses that scale difference. Section 7 explains the uncertainty of the resulting yield estimator; Section 8 uses full toy distributions for inference instead of relying on this simple correction.
''')
 doc+=page('5. Repeat the test with a functional form background',pic('recovery_functional',caption(r'''Signal-MC injections on the functional form background source. The rows and strengths have the same definitions as in Section 4: fitted-to-injected yield, signal-induced change divided by injected yield, and calibrated response $(\Ah-\delta)/(RA)$. Here the correction is derived from an independent calibration cohort of the same functional form source. Expected signal yields match the GP-source study, while source ensembles are independent. '''+mean),height='5.85in')+r'''
At $z=3$, the shifted-Gaussian signal-induced response is 0.265--0.834 for this source, close to the GP-source range. The fitted-to-injected ratio can differ more because the background-only biases differ. Agreement of the incremental response under two chosen sources is useful evidence, but does not test every possible background shape.

As a separate check, injected Gaussians centered on the logarithmic law give shifted-fit response 0.963--0.980 on the GP mean source. This is much closer to one than the signal-MC response. Signal-shape mismatch therefore matters even after the center is corrected. Gaussian controls have independent signal draws and receive no signal-MC response correction.
''')
 doc+=page('6. What a pull says about bias and uncertainty',pic('pulls',caption(r'''Independent signal-MC evaluation experiments at $A=5s_0$. The pull is $(\Ah-A)/\sh$: fitted yield minus the full selected injected yield, divided by the reported error. Top: its mean after subtracting the background-only mean pull $\mu_0$; bars are sample standard errors. Bottom: its sample standard deviation; bars are bootstrap standard errors from 2,000 whole-experiment resamples. Each point uses 100 experiments. The ideal reference lines are zero for the mean and one for the width.'''),height='4.1in')+r'''
A negative mean pull says that the fitted Gaussian yield is below the injected full selected yield. A pull width near one only says that the fluctuation scale is roughly consistent with the reported error. It does not remove or excuse a displaced mean. This is why background-only centering can look satisfactory while the positive-signal test still shows a large bias.

The likelihood also provides a direct check. At the injected yield $A$, refit the background nuisance parameters and compute
\[
 q_A=2\{\ell(\Ah,\widehat\theta)-\ell(A,\widehat\theta_A)\},
\]
where $\ell$ is the log likelihood. The usual one-parameter Gaussian references accept $q_A\le1$ and $q_A\le3.841459$ for 68\% and 95\% likelihood sets. Here those sets are produced by a Gaussian-template model and checked against the full signal-MC yield. They need not contain that yield at the stated rates when the template response is biased.

The complete counts, always out of 100, and exact 95\% binomial intervals remain in \path{results/heldout_summary.csv}. That inherited filename denotes independent evaluation. The next two sections distinguish an approximate corrected-yield diagnostic from the toy-calibrated confidence construction.
''')
 doc+=page('7. A calibrated yield and its error',r'''
The corrected point estimate subtracts the background-only yield bias and divides by the measured response:
\[
 A_c=\frac{\Ah-\delta}{R},\qquad
 R=\overline{\frac{\Ah_{s+b}-\Ah_b}{3s_0}}\ \ \text{in the calibration cohort}.
\]
For example, if the fitted yield is 600 events, the calibrated background-only bias is 100 events and $R=0.5$, the corrected estimate is 1,000 selected events. Dividing by a small response amplifies the uncertainty as well as the yield.

Let $k_0={\rm SD}\{(\Ah_0-\delta)/\sh_0\}$ measure the background-only fluctuation scale. An approximate propagated variance is
\[
 \sigma_c^2\simeq\frac{k_0^2\sh^2+{\rm Var}(\delta)+A_c^2{\rm Var}(R)
             +2A_c\,{\rm Cov}(\delta,R)}{R^2}.
\]
The bootstrap estimates $\delta$ and $R$ together, retaining their covariance. The cross term has the displayed $+2A_c\,{\rm Cov}(\delta,R)$ form; its numerical contribution can have either sign.
'''+pic('affine',caption(r'''Pull after yield calibration, $(A_c-A)/\sigma_c$, for independent signal-MC experiments at $A=5s_0$. A mean near zero indicates agreement between calibrated and injected yields for this source and strength. Bars are sample standard errors across 100 evaluation experiments, conditional on the frozen calibration table. Each toy's $\sigma_c$ includes the calibration-parameter terms in the equation above. The figure's bootstrap and sample-size definitions apply here.'''),height='2.9in')+r'''
This first-order uncertainty estimate treats the response as locally constant and uses the background-only width scale. It omits uncertainty in the pilot scale, the chosen background source and the finite signal-MC template. It is a diagnostic of a corrected estimator, not a new likelihood or an exact confidence interval. The primary limits below therefore calibrate the fitted statistic at every tested injected yield; they are not obtained by simply dividing an existing limit by $R$.
''')
 doc+=page('8. Turn the toy calibration into an upper-limit test',r'''
The key question is: \emph{would an injected yield $A$ commonly give a fit result this small?} For each preselected yield on the calibration grid, compare the observed (or evaluation) fitted yield with 100 calibration fitted yields generated at that same true $A$:
\[
 p_A=\frac{1+\#\{\Ah_A^{\rm cal}\le\Ah^{\rm eval}\}}{101}.
\]
The numerator counts calibration results no larger than the result being tested, then adds one. Reject that yield when $p_A\le0.10$. A sufficiently small fitted yield is evidence against a large injected signal. This is a lower-side probability used for exclusion; Section 9 uses the opposite direction to ask about an excess above background.

Repeat this test at $z=0,1,2,3,4,5,6,8,10,12,16,20,24$, with $A=z s_0$. Retain the \emph{entire set} of accepted nodes. Its largest node is the displayed grid endpoint. We do not interpolate between simulated strengths. A missing interior node is recorded as a hole; acceptance of the largest node is recorded as right censoring, meaning that the grid does not reach the endpoint.

\textbf{An empty set is unresolved, not a zero physical limit.} If every node is rejected, the software stores endpoint zero together with an empty-set flag. If only zero is accepted, the grid resolves no behavior between zero and its first positive node. Neither case excludes all arbitrarily small positive signals. The accepted set and its flags must accompany the endpoint.

\textbf{Why this is called calibration.} The test uses distributions from signal-MC injections even though the fitted template is Gaussian. It therefore measures the distribution of the actual fitted statistic under the intended signal yield instead of assuming unit Gaussian-template response.

With continuous, exchangeable calibration and evaluation results, the add-one rank test rejects the true node in at most $10/101$ of experiments, averaged over both calibration and evaluation cohorts. A single frozen table can have a different acceptance probability. The order-statistic reference gives a central 95\% range of 83.60--95.10\% for that probability. This calibration-table variation is separate from the binomial error of checking it with 100 evaluation experiments; ties make the stated rank rule conservative.

For comparison, the study also records a toy-rank $CL_s$ diagnostic, $\min(1,p_A/p_{0,\mathrm{lower}})$, and the inherited Gaussian-template asymptotic $CL_s$ limit. The shared $1/101$ rank floor can limit the toy-rank diagnostic's sensitivity. The Gaussian asymptotic reference uses profile-likelihood approximations and does not automatically describe the full signal-MC yield.
''')
 doc+=page('8. Check the procedure on independent experiments',pic('limits',caption(r'''Shifted-Gaussian results, with calibration and evaluation using the same specified source. Upper panels: fraction of 100 independent experiments for which the injected $A=5s_0$ is accepted (or is below the continuous Gaussian asymptotic endpoint). Bars are exact two-sided 95\% Clopper--Pearson intervals. The dashed line marks 90\%. Lower panels: median upper endpoint in background-only experiments, divided by the pilot yield-error scale $s_0$; no uncertainty bars are shown on these medians. Toy-based endpoints are grid maxima, while Gaussian asymptotic endpoints are continuous.'''),height='4.45in')+r'''
The toy-calibrated yield test retains the injected $A=5s_0$ in 81--96\% of GP-source evaluations and 84--93\% of functional-source evaluations. The corresponding Gaussian asymptotic limits contain the full signal-MC yield in only 4--72\% and 2--87\%. This failure is why improving the core center alone is insufficient for full-yield inference.

Every planned evaluation ID remains in the tables; no fits failed in the main run. The denominator is always 100. The variation among masses includes finite evaluation statistics and fluctuations of the frozen calibration tables, so the plot should not be summarized as exact 90\% coverage everywhere.

At zero injected yield, the stored numerical endpoint of an empty set also equals zero. Counting that endpoint as covering zero would hide the rejection of every tested node. The tables therefore retain both endpoint coverage and the separate, primary \emph{truth-acceptance} field. Use the latter to assess the grid test.
''')
 doc+=page('9. What the observed 2021 spectrum shows',pic('observed_2021',caption(r'''2021 10\% observed data. Lines are Gaussian-template asymptotic results on a 1 MeV mass grid. Open circles are independent GP-source toy-calibrated results only at the simulated masses, 60--240 MeV in 20 MeV steps. Upper panel: yield limits converted to the inherited coupling display defined on the next page. Nonpositive grid endpoints are omitted from the logarithmic axis and retained with their flags in the tables. Lower panel: local excess probability. Lines use the profile-likelihood asymptotic approximation; open circles rank the observed fitted yield against background-only fitted yields. The line is not an interpolation of the points.'''),height='4.9in')+r'''
For the toy-based local excess probability, count how many of the 100 background-only calibration experiments return a fitted yield at least as large as the observed one:
\[
 p_0=\frac{1+k}{101},\qquad k=\#\{\Ah_0^{\rm cal}\ge\Ah^{\rm obs}\}.
\]
This is the meaning of the former phrase ``upper null tail.'' The smallest resolvable rank is $1/101$, even when there are no exceedances. A binomial interval for $k/100$ estimates the underlying background-only tail probability; it is distinct from the add-one rank.

The smallest local asymptotic $p_0$ moves from 0.00249 at 78 MeV for the central mass Gaussian to 0.00346 at 80 MeV for the shifted Gaussian. These masses were selected by scanning the data. Their local probabilities do not include the look-elsewhere effect and are not global discovery probabilities.
''')
 doc+=page('9. How fitted event yield is converted for display',r'''
The fit fundamentally measures an event-yield parameter. The displayed coupling variable uses the inherited conversion
\[
 A=\epsilon^2 C_{2021}(m),\qquad
 C_{2021}(m)=\frac{3\pi m f_{\rm rad}}{2\alpha}\,\rho(m).
\]
Here $\alpha=1/137$, $f_{\rm rad}$ is the archived effective radiative fraction, and $\rho(m)$ is the saved observed event density averaged around the \textbf{tested signal mass} $m$. The density average uses $m\pm1.64\sigma_{\rm ref}$, including fractional overlaps with histogram bins. Mass and resolution are in GeV, and the density is events per GeV.

\textbf{The mass at which this conversion is evaluated does not move when the Gaussian center moves.} The central and shifted Gaussian fits at a given tested mass use the same $C_{2021}(m)$. This specifies precisely how the event density enters the conversion. It separates a change in extraction shape from a change in the yield-to-coupling normalization.

The implementation fits $\psi=\epsilon^2/10^{-8}$, so the fitted event yield is $\Ah=\widehat\psi\,C_{2021}(m)10^{-8}$. The raw coupling columns are electron-channel proxies. For consistency with earlier visible-channel plots, the displayed values above the muon-pair threshold are multiplied once by
\[
 1+\sqrt{1-4r}\,(1+2r),\qquad
 r=(105.6583745\,\mathrm{MeV}/m)^2,
 \quad m>211.316749\,\mathrm{MeV}.
\]
Below threshold the factor is one. This display convention does not independently validate branching fractions, efficiency or selection equivalence. It is therefore not a detector-qualified physical exclusion.

\textbf{How to interpret the two kinds of curve.} The continuous asymptotic line answers a question within the fitted Gaussian likelihood approximation. The open points use signal-MC toy distributions and the finite accepted-yield grid. Their agreement or disagreement is informative, but neither a sparse set of calibrated points nor a dense asymptotic scan establishes a globally calibrated discovery probability.
'''+table(['Gaussian choice','Dense minimum mass [MeV]','Local asymptotic $p_0$'],[[NAME[p],str(int(q.mass_MeV)),f(q.p0_asymptotic,6)] for p in ['pole','logshift'] for _,q in sub(o,scope='2021',policy=p).nsmallest(1,'p0_asymptotic').iterrows()]))
 doc+=page('10. Combine campaigns while changing only 2021',pic('observed_combined',caption(r'''Combined observed result with a common coupling parameter. The 2015 and 2016 models are unchanged; only the 2021 Gaussian center and its matched fit/training exclusion change. Lines are dense asymptotic references; open points are direct joint toy calibrations at simulated masses. All three campaigns contribute through 100 MeV, 2016+2021 through 180 MeV, and only 2021 above 180 MeV; dotted lines mark those boundaries. Empty and zero-only grid sets are omitted from the logarithmic endpoint panel. The coupling display is defined in Section 9.'''),height='4.7in')+r'''
The joint calibration simulates the full common-coupling experiment, rather than combining separately corrected amplitudes. It uses 100 pilot experiments to set the coupling scale, 100 independent GP-source calibration experiments at the thirteen strengths, and another 100 independent evaluation experiments at $z=0,1,3,5$. The 2021 signal is drawn from signal MC; older campaigns retain their inherited signal models.

For the combined \emph{excess} probability, the ranked statistic is the signed profile-likelihood root. It is the square root of twice the best-fit log-likelihood improvement over background only, with the sign of the fitted coupling. Thus the background-only exceedance count uses likelihood evidence for an excess, not the standalone 2021 fitted yield. Exclusion still orders the fitted yield parameter.

The smallest combined local asymptotic $p_0$ changes from 0.00289 at 66 MeV to 0.000933 at 67 MeV. These mass-selected minima remain descriptive local references. Changing only the 2021 model does not independently recalibrate the older campaigns.
''')
 doc+=page('11. Check the combined toy calibration',pic('joint_validation',caption(r'''Fraction of independent joint evaluation experiments retaining the injected coupling at its grid node. Left: background only; right: signal plus background at $z=5$ in the joint pilot-defined coupling unit. Each point uses 100 experiments and exact two-sided 95\% Clopper--Pearson intervals. The two Gaussian choices share each whole experiment, preserving their correlation. Calibration and evaluation use the GP mean background source; campaign support follows Section 10.'''),height='3.05in')+r'''
Across all joint evaluation cells, acceptance ranges from 78\% to 100\%; 260 of 8,000 accepted-node sets are empty. This reflects a finite, frozen calibration with discrete strengths. It does not establish exactly 90\% coverage for each mass and strength.

The observed joint inversions have five empty sets and five additional zero-only sets among twenty mass/fit-choice pairs. None accepts the largest grid node or has an internal hole. Their zero endpoints are unresolved continuous limits, not exclusions of arbitrarily small signals.
'''+table(['Mass [MeV]','Central $p_0$','Shifted $p_0$','Central grid result','Shifted grid result'],[[str(m)]+[f(sub(j,policy=p,mass_MeV=m).p0_rank.iloc[0],4) for p in ['pole','logshift']]+[('empty' if bool(q['empty']) else ('zero only' if q.upper_psi==0 else f'{q.epsilon2_90_grid_visible_legacy:.2e}')) for p in ['pole','logshift'] for _,q in sub(j,policy=p,mass_MeV=m).iterrows()] for m in range(60,241,20)])+r'''
Positive grid entries are endpoints in the coupling display of Section 9. The smallest joint empirical rank is $4/101=0.03960$ at 80 MeV for the shifted Gaussian, corresponding to three background-only exceedances. This finite sample does not empirically resolve the much smaller dense asymptotic minimum.
''')
 doc+=page('12. Would a different fit or training window help?',r'''
A wider excluded training region removes more signal contamination from the background estimate, but also removes background information. The relevant test is the uncertainty on the full signal yield after accounting for response, not signal containment alone.

The baseline fit and excluded training region both extend $2.25\sigma_{\rm ref}$ on each side of the shifted center. Three alternatives change both together to left/right half-widths $(2,2)$, $(2.5,2)$ and $(2,2.5)$ in units of $\sigma_{\rm ref}$. A fourth keeps the baseline fit window and widens only the excluded training region to contain the central equal-tail 95\% of selected signal MC.
'''+pic('windows',caption(r'''Separate GP-source side study with 100 pilot and 100 independent evaluation backgrounds. Signal-MC injections use a common $A=3s_0$ and the same experiments across window choices. The plotted ratio compares $\mathrm{SD}(\Ah_0)/R$ for each choice with the baseline: a lower value means less full-yield noise. Bars are 95\% percentile intervals from 2,000 whole-toy bootstrap resamples, preserving mass and window-choice correlations; the pilot scale is fixed. These are sensitivity diagnostics, not calibrated upper limits.'''),height='3.5in')+r'''
The three matched-window alternatives give ratios 0.983--1.014, mostly with intervals spanning one. The wider-left 100 MeV case is 0.983 [0.972, 0.997], one exploratory point among thirty comparisons. This does not establish a reliable improvement. The 95\% training exclusion worsens the noise by factors 1.15--3.54, despite reducing training leakage to about 5\%.

At 60 MeV that equal-tail exclusion spans roughly 51.0--212.25 MeV, leaving a large gap for background prediction. Keeping three external bins on either side checks geometry, not the reliability of that extrapolation. The baseline $\pm2.25\sigma_{\rm ref}$ prescription is retained.

The shape coordinate $u=(m_{\rm rec}-c)/\sigma_{\rm core}$ from Section 1 must not be confused with these widths in $\sigma_{\rm ref}$. Translating a boundary between the two requires the ratio $\sigma_{\rm core}/\sigma_{\rm ref}$.
''')
 doc+=page('13. What is established, and how to reproduce it',r'''
\textbf{Established within this study.} A shifted Gaussian improves extraction of the supplied signal MC. Background-only bias and incomplete signal response are different effects and can be measured separately. A frozen toy-calibrated yield test gives a defined conditional inference procedure that can be checked on independent experiments. The chosen baseline window remains appropriate among the tested alternatives.

\textbf{Scope of that conclusion.} The background sources, empirical signal-MC histogram and archived GP kernel settings are fixed. The selected TC signal-MC metadata identifies v13 inputs; it does not establish full v13/v16 selection equivalence or daughter association. Uncertainty in the finite signal-MC template, other possible background shapes, model selection and a global look-elsewhere correction are outside this ensemble. Earlier 2016 support/optimizer qualifications remain unresolved by these 2021 toys. A residual in one observed spectrum is not itself an ensemble bias estimate.

\textbf{Numerical evidence.} All main planned fits, truth profiles and Gaussian asymptotic endpoints passed the archived finite/positive, score and Hessian checks. The original validation includes seed replay, probability normalization, paired draws, frozen-calibration hashes and representative direct calculations. The main-body revision reruns no inference fits. The appendices add new v16 signal-MC core fits. Hash comparisons verify that the numerical inputs, cohorts, calibration and result tables are unchanged from the supplied v6.3.5 archive.

\textbf{Reproduction.} The package contains \texttt{source/report.tex}, the plot and narrative builders in \texttt{scripts/}, figure source data, and the original numerical checkpoints. The README gives the report-only rebuild commands. \texttt{protocol.json} deliberately retains the v6.3.5 scientific protocol and seed so that editorial versioning cannot be mistaken for a new experiment. \texttt{revision.json} describes v6.3.6. The source image is preserved unchanged with a SHA-256 digest. New and inherited QA records are identified separately.

The source keys \texttt{nominal}, \texttt{functional}, \texttt{pole} and \texttt{logshift} are preserved in machine-readable files for compatibility. In this note they mean GP mean background source, functional form background source, central mass Gaussian and shifted Gaussian, respectively. Inherited ``heldout'' filenames contain independent evaluation results. No terminology change alters the estimator or data.

\textbf{Related studies.} The v6.1 signal-MC study supplies the catalogue, core fits and logarithmic center law. The v6.2 and v6.3.1 matched-template studies generated and fitted signal MC; their closure does not by itself validate a Gaussian fit to signal-MC injections. The v6.3.2 study concerned controlled 2016 exposure changes. The main-body inference results are the frozen v6.3.5 results. The appendices provide the new v16 TC and UC signal-shape studies.

\textbf{Statistical references retained from v6.3.5.} The PDG statistics review discusses bias, confidence constructions and $CL_s$: \url{https://pdg.lbl.gov/2026/reviews/rpp2026-rev-statistics.pdf}. The profile-likelihood asymptotic reference is Cowan et al., \url{https://arxiv.org/abs/1007.1727}. The finite-rank rule actually used here is given explicitly in Section 8.
''')
 return doc
