# Harvard writing-sample language review catalogue

## Scope and editorial rule

This catalogue records a reader-facing revision of Sections 2--6 of the HPS
Gaussian-process writing sample. The accepted analysis-note release and the separate
selected-results release remain unchanged. The revision changes exposition, ordering,
and plot labels; it does not change numerical results, statistical definitions, or
claim boundaries.

The review combined four viewpoints:

1. a cold read with no project history;
2. a Harvard fellowship reader's flow audit;
3. a physicist-voice edit of the equations and surrounding prose;
4. a final continuity read across section boundaries.

## Highest-priority findings and disposition

| Priority | Finding | Required treatment | Status in this derivative |
|---|---|---|---|
| P0 | “Table 17” is unintelligible without the internal note history and was embedded in three plots. | Replace it everywhere, including raster/vector figure text, with a description of the physical test. | Complete. Figures now say “baseline smooth-threshold model,” “threshold-refined 65 MeV model,” and “extended-support 65 MeV model.” |
| P0 | Release numbers made the narrative read like a changelog. | Use scientific names in reader-facing prose and captions; retain internal identifiers only in filenames, source labels, and provenance records. | Complete for rendered text. |
| P0 | The method opened with configuration history instead of the inference problem. | Begin with the need to interpolate the smooth trident continuum beneath a blinded narrow-resonance window. | Complete. |
| P0 | The four-row version-lineage table required the reader to reconstruct internal development history. | Replace it with the two configurations that matter scientifically. | Complete: “three-campaign combination” and “2021-only analysis.” |
| P0 | The validation section ended with support-selection chronology after it had already reached a scientific conclusion. | Put the support study before the synthesis and end with what the pseudoexperiments do and do not establish. | Complete. |
| P0 | The excerpt lacked a self-contained statement of the physics question. | Define the dark-photon peak search, smooth trident background, GP interpolation, and validation question on the title page. | Complete. |
| P0 | The separately produced selected-results section repeated the same internal-version language. | Integrate its frozen results and figures, but rewrite the exposition using the same two scientific configuration names. | Complete. |
| P1 | Five pages of loose-preselection plots interrupted the transition from selection to resolution. | Omit those diagnostic plots from the fellowship excerpt while preserving them in the complete note. | Complete through the writing-sample conditional. |
| P1 | “Lane,” “cell,” “contract,” “card,” “production,” and “frozen” made statistical ideas sound like software or project management. | Prefer “ensemble,” “mass point,” “procedure,” “configuration,” “analysis,” and “fixed before examining the data.” | Complete in the reader-facing validation narrative. |
| P1 | The exposure shorthand was not defined. | Explain that the two scaled ensembles multiply expected counts from the native 1% or native 10% source, and that the two source samples are not interchangeable. | Complete. |
| P1 | Long captions carried provenance arguments and repeated the body. | Limit captions to what is plotted, interval definitions, marker meanings, and the one qualification needed to read the figure. | Complete for the principal validation and selected-results figures. |
| P1 | Statistical caveats were repeated until the prose sounded defensive. | State each boundary where it first matters and collect the non-claims once in the closing synthesis. | Complete, while retaining the distinctions between conditional closure, coverage, local significance, and global significance. |
| P1 | Section changes were abrupt. | Add bridges from campaigns to a common observable, from detector inputs to the GP, and from the likelihood to pseudoexperiment validation. | Complete. |
| P2 | The event-selection table remains dense for a general reader. | In a shorter application variant, retain only the physically consequential differences and move cut provenance to a note or appendix. | Catalogued; the present derivative keeps the complete selection so no requirement is silently dropped. |
| P2 | Separate detector-input source figures still make Section 3 longer than necessary. | If a strict page limit is imposed, retain the three-panel resolution comparison and one radiative-fraction summary, moving source screenshots to supplementary material. | Catalogued, not applied because no page limit was specified. |
| P2 | A few internal identifiers remain in LaTeX labels and asset filenames. | Keep them only as non-rendered traceability handles; do not expose them to the reader. | Complete for rendered text. |

## Reader-facing vocabulary

The writing sample uses the following names consistently:

| Internal identifier | Reader-facing name |
|---|---|
| v4.2 | three-campaign combined analysis |
| v4.5 | matched-refit validation procedure |
| v4.9 | paired lower-support-edge study |
| v4.9.1 | four-exposure 2021 validation study |
| v4.9.5 | 2021-only analysis with 36--300 MeV GP support |
| Table-17 scheme | threshold-refined 65 MeV model |
| fSig-anchored / 30--300 substitution | extended-support 65 MeV model |
| historical functional-form truth | baseline smooth-threshold model |
| pilot | development subset |
| reserve | independent continuation |

Internal identifiers may remain in source labels and filenames so that the derivative
can be traced back to the accepted analyses. They should not appear in the rendered
writing sample.

## Scientific statements that must remain intact

- The generating functions are conditional, source-fitted stress models, not unique
  physical descriptions of the trident continuum.
- The pseudoexperiments test extraction closure for declared backgrounds; they do not
  establish confidence-limit coverage, observed-data bias, or scan-global
  significance calibration.
- The post-result pull-mean tolerances are descriptive practical criteria, not
  predeclared equivalence tests.
- The 2021-only 36--300 MeV support must not be silently substituted into the
  three-campaign combinations, which use 40--300 MeV for the 2021 contribution.
- The 2016 full-statistics support optimization did not meet its selection criterion;
  it does not define a new observed result.
- The 3.1--3.4% signal attenuation, imperfect pull-width calibration, resolved small
  offsets, 55 MeV under-recovery, and high-mass-tail influence remain visible.
- Fixed-mass asymptotic p-values are local diagnostics until a scan-wide
  background-only calibration is performed.
- The 2021 1% points at 50--52 MeV remain support-edge diagnostics and are not promoted
  to candidate evidence.

## Recommended reading sequence

1. Physics primer and authorship statement.
2. Campaigns and their common invariant-mass observable.
3. Event selection, mass resolution, and radiative normalization.
4. Blinded GP interpolation, signal model, likelihood, and shared-coupling combination.
5. Pseudoexperiment design, generating spectra, and exposure ensembles.
6. Spurious-signal and injection-recovery results.
7. 2021 support selection and its 55 MeV limitation.
8. Integrated validation conclusion.
9. Selected observed limits and local fixed-mass diagnostics.

## Final flow check

The revised argument now moves from the experimental object to the statistical method,
then tests that method before showing observed results. Section endings pose the next
physical question rather than announcing workflow status. The last paragraph returns
to the central scientific boundary: combining campaigns adds information, while a
localized excess still requires scan-wide calibration.

