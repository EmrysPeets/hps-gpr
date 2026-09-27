# HPS-GPR slide science source map

Verified against current checkout 2026-09-22. No fits rerun; no releases altered.
All paths below are repository relative. Only use content authorized by bracket instructions.

## Preferred current sources

| Purpose | Source | Figures / key content |
|---|---|---|
| Analysis model | `study_results/v5p0p5_analysis_note_20260916/source/sections/04_methodology.tex` | log-count GP, RBF kernel, blind-window covariance, likelihood, significance, CLs, shared coupling |
| Why profile background | `study_results/v5p0p5_analysis_note_20260916/source/sections/v5_method_updates.tex` | `figures/v5_profile_background_comparison.png`, `v5_profile_injected_bias.png` in same release; fixed means omit uncertainty |
| Latest unshifted significance | `study_results/v5p8p5p3_raw_significance_20260921/source/raw_significance.tex` | `figures/raw_local_{2015,2016,2021,combined}.png`; overview is tall and unsuitable for small slide placement |
| Raw global significance | same v5.8.5.3 release | `figures/raw_global_overview.png`; fixed-source conditional raw maximum, separate from local asymptotic display |
| Calibration mechanism | `study_results/v5p8p5p3_raw_significance_20260921/figures/response_diagnostics_{2015,2016,2021,combined}.png` | Panels A/B raw; C/D standardize toys; dense four-panel figure, consider using only the instructed panel |
| Statistical fairness | `study_results/v5p8p5_consolidated_fairness_20260921/source/statistics.tex` | Two GPs have different jobs; raw ordering versus source-standardized ordering; same-field upcrossing and correlation comparisons |
| Broader tails | `study_results/v5p9_tail_structure_20260921/source/report.tex` and README | `figures/shapes_note.png`, `primary_{2015,2016,2021}.png`, `structure_response.png`, `leakage.png`; guard is separate changed-fit diagnostic |

## Slide-ready model equations

Use only the equation(s) needed by a slide, with a one-line symbol key. Render as vector math when supported; otherwise high-resolution transparent equation images.

- Background kernel: $k(x,x')=C\exp[-(x-x')^2/(2\ell^2)]$, with $x=\log m$, $C$ covariance amplitude, $\ell$ smoothness scale. Bin noise $\alpha_i\simeq1/y_i$ is distinct from $C,\ell$.
- Correlated background profile: $\boldsymbol\lambda=\mathbf b+L\boldsymbol\theta+A\mathbf w$, $LL^T=C_{\rm GP}$, $\boldsymbol\theta\sim N(0,I)$. Here $b$ is predicted counts, $L\theta$ is the correlated background adjustment, $A$ total signal yield, and $w$ is the bin-integrated signal template (not renormalized inside the fit window).
- Likelihood: $\mathcal L(A,\theta)=\prod_i\mathrm{Pois}(n_i\mid\lambda_i)\exp(-\theta^T\theta/2)$. Profile nuisance coordinates at each tested $A$; GP kernel is fixed during this local optimization.
- Raw local excess: $Z_{\rm local}=\max(r_{\rm obs},0)$, $p_{\rm local}=1-\Phi(Z_{\rm local})$. This is the conventional asymptotic mapping; no reference subtraction, scale correction, or trials correction.
- Global raw scan: $T=\max_{m\in\mathcal M}\max[r(m),0]$, $p_{\rm global}=P_B(T^*\ge T_{\rm obs})$. Null source, domain and scan procedure must be specified.
- Distinct diagnostic: $z_{\rm ref}=(r-a)/s$. This changes local ranking and is not itself the look-elsewhere correction.
- Fixed-mass exclusion: $CL_s(A)=CL_{s+b}(A)/CL_b(A)$; $CL_s(A_{90})=0.10$. Discovery trials penalties do not directly alter pointwise limits.

## Current raw local peaks (v5.8.5.3, 0.5 MeV grid)

| Data | Range MeV | Peak MeV | Local Z | Local p |
|---|---:|---:|---:|---:|
| 2015 full | 19–100 | 51 | 3.1392 | 8.469e-4 |
| 2016 full | 39–180 | 90.5 | 3.4525 | 2.777e-4 |
| 2021 native 10% | 50–250 | 78 | 2.8086 | 2.488e-3 |
| One shared coupling | 19–250 | 66 | 2.7602 | 2.889e-3 |

The combined union uses available datasets at each mass; all three contribute only over 50–100 MeV. This is not Fisher, Stouffer or independent-amplitude combination.

## Tail-study statements

Nominal Gaussian core is held fixed inside ±2.25 sigma before full-support normalization; tail scale increases 10/20/30%. Primary median limit changes are approximately 0.168/0.345/0.529%. Primary strongest regions remain 51, 90, 78 MeV (1 MeV grid). Compare each alternate shape to its freshly recomputed same-window Gaussian. ±4.5 sigma guard changes the fit and training mask and must not be labeled a pure tail-only effect. No combined, global or new coverage calibration is supplied by v5.9. Legacy numerical stopping differences from published v5.0.5 are separate from tail changes.

## Claim boundaries to preserve

- Production blind half-width stays ±2.25 sigma; other widths are diagnostics.
- Raw local p-values are asymptotic. Global v5.8.5.3 probabilities are conditional on frozen observed-data-derived GP sources, not unconditional discovery claims.
- Centering/scaling was used in earlier reference-local plots for all datasets and shared coupling; current raw plots explicitly remove it.
- A zero count among 256 toys bounds a probability; it is not zero probability.
- Independent nonnegative amplitudes at common mass use $q_R=\sum_d\max(r_d,0)^2$; $\sqrt{q_R}$ is not a Z.
- No slide without bracket instructions should be changed; concerns belong in separate feedback.

## Memory navigation provenance

Memory was used only to locate and prioritize sources; numerical statements above were checked in present source files. Relevant navigation entries: MEMORY.md lines 23–31, 119–146; rollout IDs 01a0c616-92ff-7ab3-bd03-63cb71cca8bd and 01a0c536-fcf0-7972-8748-b2520ff51385.
