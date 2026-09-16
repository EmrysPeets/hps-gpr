from pathlib import Path
import re,shutil,json
B=Path('study_results/v5p0p5_analysis_note_20260916');S=B/'source'
p=S/'sections/v5_global_results.tex';t=p.read_text();a=t.index('These individual fits do not establish stability');z=t.index('\\paragraph{What the stress response means.}',a)
t=t[:a]+r'''A bounded joint check in v5.0.5 repeats the signed shared-coupling fit at
72--80 MeV using the same spectra, detector widths, native-density conversion
and frozen kernel coordinates. With the bin origins unchanged, doubling
and quadrupling the histogram widths changes the combined root at 76 MeV
from $+0.166$ to $-0.216$ and $-0.137$. Shifting each coarse origin by half
its new bin width gives $+0.530$ and $-0.695$, respectively. The first
positive tested point is consequently 76 or 77 MeV in these alternatives.
The zero crossing is sensitive to binning at the one-step level.

Figure~\ref{fig:v505-joint-binning} and
Appendix~\ref{app:v505-joint-binning} give the full comparison, including
the small accompanying changes in sampled support endpoints. These checks
recompute the observed pointwise fits; they do not recompute the stress
field or its global tail. The spike's position and height therefore cannot
be interpreted as a detector-resolved resonance feature. Binning should be
chosen from resolution and closure controls, before inspecting which choice
makes the observed probability more striking.

'''+t[z:];p.write_text(t)
p=S/'sections/v504_global_appendix.tex';t=p.read_text()+r'''
\subsection{Joint binning check around the 76 MeV gate crossing}
\label{app:v505-joint-binning}
The archived v5.1.0 scans above describe the individual campaigns. For this
revision, a short joint scan tests the effect on the combined signed fit
itself. Five binning choices are fixed before the refits: the nominal widths,
twice and four times those widths, and one additional origin for each coarse
choice shifted by half its bin width. The tested masses are the integers
from 72 through 80 MeV. The choice of this region follows the question about
the displayed spike; no new discovery test is assigned to the comparison.

The original analysis widths are 0.25 MeV for 2015 and 2016 and 0.625 MeV
for 2021. Counts are summed in adjacent original bins, with exact conservation
inside each included group. An incomplete group at the far upper training
edge is omitted, as in the original rebin operation; shifting the origin
also removes leading support bins. The effective endpoints are saved for
every campaign and choice. Thus this check includes the small change in
sideband sampling that accompanies the selected bin boundaries; it does not
isolate bin origin from support-edge effects.

Each tested mass retains its archived kernel amplitude and length scale.
The log-count GP is conditioned again on the rebinned sidebands, excluding
bin centers within $\pm2.25\sigma_m$. The count covariance and
bin-integrated Gaussian template are reconstructed, and the joint Poisson
likelihood is fitted with a common signed coupling and independent GP
background constraints. The native-bin prompt density, detector resolution
and radiative conversion are held fixed. The nominal 76 MeV fit reproduces
the reported $r=0.166$ to the quoted precision, and its individual roots
agree with the archived values. Every fitted expectation is positive and
all 180 fitted mass/scope/binning coordinates converge.

\begin{table}[htbp]\centering\small
\begin{tabular}{lrrr}\toprule
Widths relative to nominal & Origin shift & $r_{\rm comb}(76)$ & First $r>0$ [MeV]\\\midrule
$1\times$ & 0 & $+0.166$ & 76\\
$2\times$ & 0 & $-0.216$ & 77\\
$2\times$ & half a coarse bin & $+0.530$ & 76\\
$4\times$ & 0 & $-0.137$ & 77\\
$4\times$ & half a coarse bin & $-0.695$ & 77\\
\bottomrule\end{tabular}
\caption{Signed joint-profile response in the bounded v5.0.5 diagnostic.
``First'' refers only to the tested 72--80 MeV grid. The shifted origins
are applied separately using each campaign's bin width.}
\label{tab:v505-joint-binning}\end{table}

\fig{1.0}{../figures/v505_76_joint_binning.pdf}{Observed signed fit scans
under the five declared bin choices. The horizontal zero line is the
positive-fit gate boundary. At 76 MeV, the combined fit can lie on either
side of it while the 2016 component remains negative. Curves connect tested
integer masses; no sub-MeV zero-crossing precision is implied.}
{fig:v505-joint-binning}
The saved CSV and protocol record the bin widths, origin shifts, effective
supports and convergence diagnostics. This small comparison draws no toys,
reoptimizes no kernels and computes no global probabilities. A full binning
choice for unblinding still requires the corresponding closure and
calibration checks.
''';p.write_text(t)
p=S/'sections/v5_history.tex';t=p.read_text().replace('records the saved\nbinning evidence near 76 MeV','adds a bounded joint binning check near 76 MeV using the saved spectra');p.write_text(t)
p=S/'main.tex';t=p.read_text().replace('background-calibration comparisons, mass-resolution variations and candidate\nextractions are presented with their respective assumptions.','background-calibration comparisons, mass-resolution variations and candidate\nextractions are presented with their respective assumptions. Version 5.0.5\nadds conditional full-2021 projections, campaign geometry and magnetic\ntransport, a joint binning check near 76 MeV, and a summary of the\nbeam-energy and rate studies.');p.write_text(t)
# Keep the scripts that made this derivative as an editorial audit trail.
for src,name in [('/tmp/revise_hps_note_20260916.py','editorial_revision.py'),('/tmp/build_hps_revision_figures.py','prepare_figure_sources.py'),('/tmp/finish_v505_text.py','finish_editorial_revision.py')]:shutil.copy2(src,B/'provenance'/name)
