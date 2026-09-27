# Review of coherent GP and regional background references

17 September 2026. This review concerns the new generating backgrounds; the main fixed-GP inference policy is retained.

## Recommendation

A coherent smoothed GP mean and a separately fitted rising/falling construction are useful next references for the 2016 study. The archived hybrid stress function already has source-fit qualifications, so its large fitted offset need not be treated as an unavoidable property of every credible 2016 continuum. Define each alternative completely, freeze it before the new fit-response comparisons, and report the alternatives together. This tests whether the previous pathology was specific to the generating shape.

The decision criterion should include the source description, interpolation across held-out regions, numerical stability, and recovery of fixed injected signals. A small background-only signed root is desirable but insufficient to qualify a new physical null. Keep the old reference as a traceable stress comparison rather than silently replacing it in the released significance.

## What makes the reference coherent

For each named construction, specify one finite, positive vector of expected bin counts over the complete retained support,

\[
B=(B_1,\ldots,B_n),\qquad N_i\sim\operatorname{Poisson}(B_i).
\]

The same complete realization of `N` must be used at every mass in a scan. The fitted background can change with its moving signal mask; the generating `B` cannot change with the tested signal mass. This separates a coherent reference from the earlier mass-specific sideband GP controls, whose individual truth functions depend on the selected anchor.

A precomputed regional blend, or a single precomputed stitched function, is mathematically coherent once frozen. Its seam remains a modeling question. Require the same count units, fixed overlap and join rule, finite positive bin expectations, and adequate smoothness at the join. Record which bins train each component and whether their overlap reuses observations. Inspect the join and its neighboring fit-response roots explicitly; a smooth visible curve can still contain signal-shaped curvature.

## Relation to the significance-GP paper

[Ananiev and Read, arXiv:2206.12328](https://arxiv.org/pdf/2206.12328), Section 2, starts from specified background bin means and propagates independent bin fluctuations through the signed-root scan. A fixed GP-derived or regional expectation is compatible with that construction. The paper's centered, unit-normal marginal premise still requires checking for the chosen inference and null; the covariance calculation does not establish it.

## Why a same-GP reference can look especially good

Generating pseudo-data from a smoothed mean obtained using the same GP family and then analyzing them with that family is a useful internal recovery control. It preferentially probes shapes that the analysis can represent. If the source was the observed spectrum, the construction also conditions on the data being tested; smoothing does not turn it into an independent background measurement. A sufficiently flexible all-data source fit can absorb a real narrow excess into its generating background and reduce apparent significance.

Source agreement alone has this limitation too. Source residuals, deterministic fit responses, signal recovery, and independently held-out predictive checks answer complementary questions. New Poisson random seeds provide independent fluctuations conditional on the fixed source, not independent evidence that the source is the physical continuum. The 20-minute study can establish conditional numerical behavior and identify promising candidates. It cannot certify a replacement particle-discovery calibration from favorable centering alone.

## Scope of the bounded tests

Keep GP-derived mean curves fixed while generating Poisson experiments. This clearly conditions on their estimated background shapes and omits uncertainty in the source fit. If posterior GP realizations are later used to vary the generating background, their induced correlations between bins must be propagated coherently through whole scans; independent one-bin Poisson directions alone would omit that extra covariance. Avoid confusing such generating uncertainty with the GP nuisance covariance already profiled by the analysis.

For the present comparison, retain the same histogram support, resolution, fit masks, signal normalization, kernel coordinates, toy sizes and injected physical yields across truths. The observed signed root is then unchanged, while the null offset, width and resulting conditional tail can change. The new probabilities should be named by their generating reference. Any future model-selection rule, tuning or regional-boundary choice must be frozen and included in the validation of the complete eventual analysis.

## Construction audit

The frozen `scripts/build_truths.py` and `truths/protocol.json` were inspected at 20:38 UTC. The five supplied arrays are coherent fixed references. They retain the exact observed support and edges, contain finite positive counts, and are not redefined during the mass scan. All GP references use the archived 76 MeV kernel coordinates; the half-length reference changes that length by exactly 0.5 and retains its amplitude. This is a declared source-model choice, not a source-optimized global kernel.

The blocked GP uses centers every 5 MeV, positive compact C2 partition weights over ±5 MeV, and fixed excluded intervals that surround each positively weighted query by at least the analysis's 2.25-resolution exclusion over the search. These predictions are assembled once into a fixed curve. The absence of postfit total-count renormalization avoids reintroducing excluded observations through a shared normalization. This remains a source-conditioned construction with previously fitted kernel coordinates, rather than an independent control sample.

The regional reference fits separate four-coefficient Poisson log-cubic count models over 30–75 and 50–210 MeV. A predetermined quintic smoothstep blends their log means over 50–75 MeV, giving positive C2-joined expectations. Numerical gradients are small and both fits report optimizer success. The joining rule is explicit and does not inspect local fit roots.

Construction correctness should not be confused with source agreement. The saved Pearson residual sum divided by the number of bins is 1.085 and 1.067 for the full-data GP references, **5,028.94 for the blocked reference**, and **349.43 for the regional reference**. The blocked expected count total is 1.09350 times the observed total; the regional total is 1.00265 times it. Therefore agreement in total counts does not rescue the regional shape, and the blocked curve has both normalization and shape problems. These are source-fit diagnostics, not formally calibrated chi-square probabilities; the full-data GP values are in-sample. The poor blocked/regional examples do not rule out better independently specified blocked or regional models.

A source-level injected-signal reconstruction would add a useful complementary check if available: rebuild the generating curve after adding a fixed signal to the source and measure how much of it enters the source mean. This tests an issue that good recovery from injections into an already frozen mean does not address. The latter tests the main extractor; the former tests whether fitting the source has already explained a candidate away.

The archived reference should be described as a conditional hybrid stress histogram with a failed broad-component source-fit flag. Its low-mass component passed source-shape gates; the full hybrid was not qualified by the failed full-2016 support study. This labeling correction was requested from the builder author before packaging.
