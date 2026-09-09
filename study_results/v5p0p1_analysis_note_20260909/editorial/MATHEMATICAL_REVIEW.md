# Mathematical source review for v5.0.1

Scope: read-only audit of published v5 notation against the frozen code used by the released fits and global calculations. No fits, toys, or likelihood results were regenerated. The root editor implements prose changes. The reference paths, line ranges and SHA-256 identities are in `math_sources.json`; these identify the reviewed old source and implementation, not an attestation of future edited v5.0.1 files.

## 1. Correlated background profile

At a fixed mass, use a signal coordinate \(\psi\), equal to total signal yield for an individual fit or the signed auxiliary extension of \(\epsilon^2\) for a combination. For the combination define \(v_d=K_d^{\rm eff}w_d\). The full signal template is integrated over histogram bins and then restricted to the fit window without renormalization.

\[
\lambda_d=b_d+L_d\theta_d+\psi v_d,\qquad
\ell_d(\psi,\theta_d)=\sum_i[\lambda_{di}-n_{di}+n_{di}\log(n_{di}/\lambda_{di})]+\tfrac12\theta_d^T\theta_d.
\]

Use the zero-count convention \(0\log(0/\lambda)=0\) and require total \(\lambda_{di}>0\). The Gaussian nuisance coordinates are otherwise unbounded. The additive model does not separately constrain \(b_d+L_d\theta_d\) positive. The covariance is the GP uncertainty on its inferred mean; Poisson variation of observed counts is represented separately. The Gaussian penalty is the interpolation constraint, not a new independent auxiliary measurement.

Define \(\hat\theta_d(\psi)=\arg\min_\theta\ell_d\) and \(\ell_{p,d}(\psi)=\ell_d(\psi,\hat\theta_d(\psi))\). The stationarity and curvature equations are
\[
L_d^T(1-n_d/\lambda_d)+\theta_d=0,\qquad
H_{\theta\theta}=L_d^T\operatorname{diag}(n_d/\lambda_d^2)L_d+I.
\]
The observed GP mean and covariance stay fixed while profiling the local likelihood. In the later bootstrap/global study each whole spectrum first retrains the sideband means, count-dependent errors and posterior using frozen kernel coordinates; it then performs that spectrum's local profile.

Source: dense reference `background_profile_comparison_20260905/run_comparison.py:35-74,76-153`; batched implementation `v4p9p13_calibration_20260905/batch_profile.py:5-42,64-98`; coherent retraining `calibration_core.py:40-85,122-145`.

Numerical qualification: write \(C_d^{\rm eff}=L_dL_d^T\) for the conditioned covariance. Production applies a declared diagonal load and further core Cholesky jitter; raw GP covariance is not literally the factor product. The attested older solver uses a rate floor and quadratic penalty near zero, whereas the dense later reference rejects nonpositive rates. The formal positive-rate objective describes accepted interior solutions; do not assert that all backends use an identical hard-constraint optimizer. See production `run_final_combinations.py:219-264` and attested `statistics.py:121-145,326-346`.

## 2. Shared-coupling likelihood: one important equation correction

With independent nuisance blocks,
\[
\ell_{p,\rm comb}(\psi)=\sum_{d\in\mathcal D(m)}\ell_{p,d}(\psi),\qquad
\hat\psi_{\rm unc}=\arg\min_{\psi\in\mathbb R}\ell_{p,\rm comb}(\psi).
\]
The profiled NLLs add. Independently normalized per-dataset profile-likelihood ratios generally do not: published v5 `04_methodology.tex:982-984` must not be read as adding ratios whose denominators are separately fitted dataset signal maxima. The correct combined likelihood ratio uses the same common best-fit signal in every denominator:
\[
q_{\rm comb}(\psi)=2\sum_d[\ell_{p,d}(\psi)-\ell_{p,d}(\hat\psi_{\rm comb})].
\]
Use the physical common denominator and one-sided clipping for upper limits. Actual code combines count vectors, block-diagonal covariance and one common signal vector before optimization; it does not add individual significances or limits. Source: `run_final_combinations.py:754-790`, `calibration_core.py:49-54`.

At a common local reference, independent channels add information, \(I_{\rm comb}=\sum_d I_d\), so the local quadratic approximation gives \(\sigma_{\rm comb}\simeq(\sum_d I_d)^{-1/2}\). This can illustrate the reason a combination helps; it does not promise an everywhere tighter observed limit or establish calibrated sensitivity.

## 3. Wald approximation and physical boundary

Wald means a local quadratic profile and an approximately normal repeated estimator:
\[
\ell_p(\psi)-\ell_p(\hat\psi_{\rm unc})\simeq(\psi-\hat\psi_{\rm unc})^2/(2\sigma_\psi^2),\qquad
\sigma_\psi^2=[H_{\psi\psi}-H_{\psi\theta}H_{\theta\theta}^{-1}H_{\theta\psi}]^{-1}.
\]
It is an approximation to the repeated-experiment probability law, not a replacement of the actual profile fit by least squares or a coverage guarantee. Let \(\hat\psi_+=\max(0,\hat\psi_{\rm unc})\). Then
\[
\widetilde q_\psi=\begin{cases}0,&\hat\psi_{\rm unc}>\psi,\\2[\ell_p(\psi)-\ell_p(\hat\psi_+)],&\hat\psi_{\rm unc}\le\psi.\end{cases}
\]
For negative \(\hat\psi_{\rm unc}\), the quadratic approximation gives \(\widetilde q_\psi=(\psi^2-2\psi\hat\psi_{\rm unc})/\sigma_\psi^2\). This differs from the interior squared displacement because the denominator is the physical null.

For positive Asimov reference \(q_A\), the implemented tail coordinates are
\[
(z_{s+b},z_b)=\begin{cases}(\sqrt q,\sqrt{q_A}-\sqrt q),&q\le q_A,\\((q+q_A)/(2\sqrt{q_A}),(q_A-q)/(2\sqrt{q_A})),&q>q_A.\end{cases}
\]
\(CL_{s+b}=\bar\Phi(z_{s+b}), CL_b=\Phi(z_b), CL_s=CL_{s+b}/CL_b\). The code computes exact observed and Asimov profiles before applying this asymptotic map and evaluates the ratio in log probability space. Both branches are asymptotic; the second is the boundary-modified branch. See release-local `runtime/bounded_tildeq_cls.py:50-106` and attested `statistics.py:638-706`.

The signed auxiliary root is \(r=\operatorname{sgn}(\hat\psi_{\rm unc})\sqrt{2[\ell_p(0)-\ell_p(\hat\psi_{\rm unc})]}\). Discovery uses the nonnegative alternative and the conventional asymptotic display \(Z=\max(0,r)\), \(p_0=\bar\Phi(Z)\).

## 4. Significance-field response and scan ordering

Use different symbols for the background covariance and the covariance of the significance field:
\[
a_m=r(B;m),\quad D_{im}=r(B+\sqrt{B_i}e_i;m)-a_m\simeq\sqrt{B_i}\,\partial r_m/\partial n_i,
\]
\[
\Gamma=D^TD,\quad s_m=\sqrt{\Gamma_{mm}},\quad R_{mn}=\Gamma_{mn}/(s_ms_n),\quad z^*\sim N(0,R),\quad r^*=a+s\odot z^*.
\]
There is no division by the number of response rows. They are unit-noise response directions, not empirical field samples. The finite positive one-bin perturbations approximate derivatives; they are not analytic infinitesimal derivatives. The Asimov offset is a deterministic reference, not necessarily the exact mean of the nonlinear Poisson estimator. The width is a linearized response width. The field correlation is constructed from responses, not fitted to validation outcomes or assumed to be an RBF.

For observed or simulated roots, set \(t_m=(r_m-a_m)/s_m\) when \(r_m>0\), otherwise \(t_m=-\infty\). Then \(T=\max_{m\in\mathcal M}t_m\), and \(p_{\rm global}(t)=P(T^*\ge t)\). A displayed global curve at mass \(m\) uses threshold \(t_m^{\rm obs}\) but always the same complete-grid maximum distribution. The local probability is \(\bar\Phi(t_m)\) for positive raw root and one otherwise. Retain the separate raw ordering \(U=\max_m\max(0,r_m)\); it is another test.

For the union, the shared 1626-bin basis contains 484, 720 and 422 bins from 2015, 2016 and 2021. The baseline plus perturbations requires 1627 deterministic full analyses over the 232-point grid. Inactive-dataset response rows are zero. Independent year streams pair into 1000 joint validation experiments; each complete spectrum is preserved across masses. Independent segment maxima cannot be spliced. See `v4p9p16_combined_global_20260906/PROTOCOL.md:19-62`, `analyze_combined.py:40-76,92-111`.

Computational diagram: coherent mean and one-bin perturbations -> expensive analysis response -> offset and covariance -> many cheap correlated root vectors -> full-grid maxima; a separate direct-Poisson validation branch checks the approximation. This illustrates computational reuse without inventing a timing gain or replacing physical background qualification.

For zero exceedances in N independent simulated fields, the one-sided 95% binomial upper bound is \(1-0.05^{1/N}\): 1.4978549e-5 for 200000 GP draws and 0.0029912495 for 1000 direct scans. The central two-sided 95% upper endpoint instead uses 0.025. These intervals concern sampling uncertainty under the respective simulation models. A zero count is neither measured zero probability nor a validated particle significance. See analyzer lines 16-22.

## 5. Yield-conversion dimensional correction

Published v5 `04_methodology.tex:559-562` has a differential signal density on the left, while its right side contains \(m\,dN_{\gamma^*}/dm\), which has units of events. The left side should be the total narrow-resonance yield \(A_d=N_{A'}\), not \(dN_{A'}/dm\):
\[
A_d=\frac{3\pi m\epsilon_{\rm phys}^2}{2\alpha N_{\rm eff}^{\rm BR}}\left.\frac{dN_{\gamma^*}}{dm}\right|_m,
\qquad N_{\rm eff}^{\rm BR}=1/\mathcal B_{ee}.
\]
Equivalently the stored unit-electron-branching coordinate is \(\epsilon_{ee}^2=\epsilon_{\rm phys}^2\mathcal B_{ee}\). The implementation uses \(A_d=\epsilon_{ee}^2 3\pi m f_{\rm rad}^{\rm eff}\rho/(2\alpha)\), with \(\rho\) in counts per GeV and mass in GeV. Thus the coupling is dimensionless and A is an event yield. The effective width ratio above the muon threshold is mass dependent, not an integer channel count.

This is confirmed by frozen `runtime_combined/hps_gpr/conversion.py:14-45,49-81`. Primary physics cross-check: [HPS 2016 paper, Section III.4, equations 5-7](https://arxiv.org/html/2212.10629v3#S3.SS4) explicitly converts the integrated signal-event limit. Its preceding Eq. 4 itself prints the analogous differential-left-side inconsistency; copy the operative event-yield relation, not that notation.

## Remaining limits of this audit

No new probability calibration, validation of physical generating backgrounds, numerical rerun or rendered-page review was performed here. In particular, additional Gaussian-field draws reduce sampling error of that approximation and cannot repair unqualified backgrounds or validate unresolved rare tails. The independent 2016 state exception remains a separate matter.
