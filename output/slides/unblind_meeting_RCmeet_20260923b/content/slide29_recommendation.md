# Slide 29: one yield-to-coupling equation

Use the familiar **boxed conversion on the left, limit plot on the right** from the user's previous presentation. Preserve the current plot object `g409df70e3d8_1_115`. Remove the separate yield, inverse-yield and branching equations from the previous revision; explain the conversion with one equation and a compact symbol key.

## Verified primary source

“BEST” is **J. D. Bjorken, R. Essig, P. Schuster and N. Toro**, *New Fixed-Target Experiments to Search for Dark Gauge Forces*, **Phys. Rev. D 80, 075018 (2009)**, [arXiv:0906.0580](https://arxiv.org/abs/0906.0580), [DOI](https://doi.org/10.1103/PhysRevD.80.075018).

The relevant relation is **Eq. (19), printed page 5** of the [paper PDF](https://arxiv.org/pdf/0906.0580#page=5):

\[
\frac{d\sigma(X\to A'Y\to\ell^+\ell^-Y)}{d\sigma(X\to\gamma^*Y\to\ell^+\ell^-Y)}
=\left(\frac{3\pi\epsilon^2}{2N_f\alpha}\right)\left(\frac{m_{A'}}{\delta m}\right).
\]

The denominator is the **radiative** trident contribution. The paper integrates over a mass interval of full width δm and assumes Γ ≪ δm ≪ m. Its N_f is the available-species approximation, neglecting phase-space corrections. The screenshot `BEST-page5.png` was visually checked against the PDF.

## Recommended slide copy

Title: **Signal yield and coupling limits**

Left kicker: **From the yield limit to ε²**

One boxed equation:

\[
\boxed{\epsilon^2_{\rm up}
=\frac{2\alpha N_f N_{\rm sig}^{\rm up}}
{3\pi m_{A'} f_{\rm rad}\left(dN_{\rm bkg}/dm\right)}}
\]

Native definitions, 15–16 pt:

- N_sig^up: 90% signal-yield limit
- f_rad: radiative fraction of the background
- dN_bkg/dm: local background density
- N_f = 1 / BR(A′ → e⁺e⁻)

Footer: **BEST, Phys. Rev. D 80, 075018 (2009), Eq. (19), rearranged.** Link the citation to the paper. One short plot caption is sufficient: **Observed 90% limits; visible branching correction included.**

The density form matches the prior user slide directly and avoids introducing N_rad or δm in the visible symbol key. Keep N_f on the slide because the current plot extends above the dimuon threshold. The equation should occupy about 220 × 55 pt within the existing left column. Keep the plot's current aspect ratio and preferably its current frame, approximately x = 273, y = 107, w = 437, h = 240 pt. A thin burgundy box recalls the prior slide; an arrow is optional because it is not needed to understand the new two-column composition. Avoid additional equation panels.

## Mapping to the present analysis — speaker notes

The displayed yield formula is a **rearrangement and notation translation of Eq. (19)**, not the paper's equation copied verbatim. Identify N_sig^up with the current full-template yield limit A90. For the selected prompt histogram,

\[
N_{\rm rad}=f_{\rm rad}\left(\frac{dN_{\rm bkg}}{dm}\right)\delta m.
\]

Substitution cancels δm and gives the current analysis conversion. No extra window fraction is needed: A90 is the total fitted template yield, not a count limited to the extraction window. If an in-window signal yield were used instead, its fraction would need to be treated consistently; that is not the choice here. The current density convention uses a full width 3.28σ (±1.64σ); the training exclusion/extraction uses ±2.25σ. Do not replace δm with a half-width, one σ, or 4.5σ merely because the extraction window is wider.

For the displayed **physical minimal-visible coupling**, use the phase-space-aware electron-channel factor

\[
N_f\longmapsto\Gamma_{\rm tot}/\Gamma_{ee}=1/\mathcal B(A'\to e^+e^-).
\]

This is 1 below the dimuon threshold and approximately 1.725 at 250 MeV in the note's minimal model. It is the precise version of the paper's species-count approximation for this channel. The existing overlay **already includes this factor**; do not transform its curves again. α is the fine-structure constant, distinct from the GP observation-noise α. No new K_d or ρ notation is necessary on this slide.

## Previous user slide inspected

[4th_year_presentation_final, slide 13](https://docs.google.com/presentation/d/1zIMHUyeQhD-DipRmfFCufFdLxBs1-Zf5XVYKYY8Kf6U/edit#slide=id.g272063acd90_1_986), native page ID `g272063acd90_1_986`:

- Burgundy EB Garamond headings on the SLAC theme.
- Yield-limit explanation in the left column; boxed ε² conversion below it.
- Published exclusion plot in the right column; burgundy arrow from equation to plot.
- Its displayed conversion uses N_sig^up, m_A′, f_rad and dN_bkg/dm. This directly supports reusing the user's yield notation instead of the new K_d notation.

The equation image is `g272063acd90_1_1004`. Its historical expression omits N_f because the shown mass interval is below the dimuon threshold. Match the layout and notation, not the old 95% CL threshold or historical exclusion wording.

Files saved here: `raw-reference-4thyear.json`, `reference-design.json`, `reference-design.md`, `reference-slide13.png`, `BEST_0906.0580.pdf`, `BEST-page5.png`. The full-reference export helper failed to materialize the PDF; the native slide-thumbnail fallback supplied the inspected 1600 × 900 reference image. No native Slides writes were performed.

Local normalization references: `study_results/v5p0p5_analysis_note_20260916/source/sections/04_methodology.tex`, equations `eps2-conversion`, `Kd-conversion`, `dimuon-eps2-map`. The preserved overlay's generator is `study_results/v5p0p4_analysis_note_20260911/scripts/make_v504_figures.py`, lines 29–32 and 67–68; it multiplies the electron-channel limit by the branching factor before plotting. The identical v5.0.4/v5.0.5 overlay SHA was verified in the previous revision.
