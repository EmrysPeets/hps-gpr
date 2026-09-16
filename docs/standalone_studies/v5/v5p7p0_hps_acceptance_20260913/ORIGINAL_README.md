# HPS v5.7: angular acceptance and selected mass spectra

The main figure has three columns (2015, 2016, 2021) and three rows: a vertical angular diagram, a calculated angular acceptance curve, and the existing selected-pair mass spectrum. It explains the beam-energy dependence with a common geometry model. It does not establish the full signal efficiency of the three detector configurations.

## What the mass edges mean

For an on-axis parent of energy E(A') = x E(beam) decaying to equal-energy, effectively massless daughters, the pair mass is m = x E(beam) sin(theta), where theta is each daughter's angle from the beam. The opening angle between the two tracks is 2 theta, not theta. At x = 1:

| Run | Beam energy (GeV) | Mass at 15 mrad (MeV) | Model upper edge at 70 mrad (MeV) |
|---|---:|---:|---:|
| 2015 | 1.056 | 15.84 | 73.86 |
| 2016 | 2.300 | 34.50 | 160.87 |
| 2021 | 3.740 | 56.10 | 261.59 |

**The 15 mrad vertical gap alone gives no upper mass cutoff.** The last column assumes an additional 70 mrad polar cap. That cap is inspired by the historical HPS/test-detector design requirement of approximately 15-70 mrad; it is not a verified outer boundary for any complete run configuration. The diagram is a vertical slice of that model, with angles enlarged, not an engineering drawing. Boundary configurations occupy zero measure: acceptance rises above the inner edge and falls to zero at the capped outer edge.

The 15 mrad numbers are also conditional, not the lowest masses that HPS can ever record. Lower parent energy moves the edges down: x = 0.8 gives inner scales of 12.67, 27.60, and 44.88 MeV. Actual events have different parent energies and directions. The 2021 apparatus also had a seventh tracking layer and a positron hodoscope; using the same window in each model panel isolates the beam-energy effect rather than reproducing those detector differences. The electron mass is neglected; its correction to the x=1 symmetric reference edges is below 0.04 MeV.

## Calculated curves

Every point is the fraction of all decays in a fixed-mass, fixed-energy model that pass both daughter cuts. The parent is prompt and travels along +z. The rest-frame transverse-vector decay density is 3/8 (1 + c^2), where c = cos(theta*), with uniform azimuth. No physical production spectrum or polarization mixture is inferred.

With E = x E(beam), beta = sqrt(1 - m^2/E^2), and negligible electron mass, pT = (m/2) sqrt(1-c^2) and pz(+/-) = (E/2)(beta +/- c). Both daughters must be forward and satisfy |atan(py/pz)| >= 0.015. The colored curves additionally require atan(pT/pz) <= 0.070 for both daughters. The azimuth is integrated analytically and c by deterministic quadrature. The grey dotted curve removes the outer cap. Solid color uses x=1; dashed color uses x=0.8. Neither is averaged over an assumed x distribution.

These fractions omit the production angular distribution, target material and multiple scattering, magnetic transport, finite sensor geometry, hit requirements, trigger, reconstruction, and displaced-decay acceptance. A calibrated signal curve would need matched generated and selected signal samples with verified denominators and run-specific selections. No such three-run set was established in the bounded source search. The local HPSTR SIMP radiative acceptance products use different control selections and SLIC-level denominators and were not substituted for nominal A' efficiency.

## Data row and provenance

The bottom row gives raw selected pairs per MeV in 1 MeV bins, without luminosity, beam-current, production-cross-section, or acceptance scaling. The samples retain their existing labels: 2015 full, 2016 full, 2021 10%. The background-dominated distributions therefore show where these samples have selected pairs, not an A' production rate or efficiency. Their selections and exposures are not matched across years.

The display retains the prior 15/30/36 MeV crops and ends at 300 MeV. The 2015 source histogram ends at 150 MeV; the grey region beyond it denotes unavailable bins, not zero physical acceptance. Displayed totals are 21,442,838; 73,218,251; and 141,305,897 pairs. Crops and histogram endpoints are not detector cutoffs. Missing/cropped bins are explicitly marked in the CSV.

`inputs/` holds unchanged ROOT snapshots and original paths. `derived/input_provenance.json` records source hashes, histogram names and ranges; the three CSVs save the exact plotted numbers. `references.json` gives primary sources and claim boundaries. Existing studies and inputs were not edited.

## Reproduce and inspect

Run `python3 scripts/make_acceptance.py` then `python3 scripts/make_methods.py` from this folder. Required packages: NumPy, SciPy, matplotlib, uproot, ReportLab and pypdf. The script sets BLAS/OMP to one thread and uses no multiprocessing, new toys or detector simulation. CPU use was coordinated with the existing v5.5 and v5.6 tasks.

The numerical checks cover normalization/bounds, inner/outer endpoints, beam-energy scaling, count-preserving rebinning, and agreement with independent two-dimensional integration of boosted daughter momenta. The representative direct integration differs by at most 0.000105 in absolute acceptance. The rendered main figure and the two explanatory pages were visually checked; source/figure text was checked separately. See `qa/` for results. These checks validate the stated model, not detector efficiency.

## Primary references

1. [Baltzell et al., The Heavy Photon Search Experiment, arXiv:2203.08324](https://arxiv.org/abs/2203.08324), Table I and detector-upgrade description: campaign energies and apparatus changes.
2. [Adrian et al., original 2015 search, arXiv:1807.11530](https://arxiv.org/abs/1807.11530), p. 2: 1.056 GeV. Use this original value rather than the inconsistent retrospective value in the later 2016 paper.
3. [Adrian et al., 2016 prompt/displaced search, arXiv:2212.10629](https://arxiv.org/abs/2212.10629), pp. 2 and 5: 2.3 GeV and active silicon at 1.5 mm for a first layer 10 cm downstream (15 mrad). The passive 0.5 mm edge is not the active acceptance edge.
4. [Battaglieri et al., HPS Test Detector, arXiv:1406.6115](https://arxiv.org/abs/1406.6115): approximately 15-70 mrad design requirement, used only to motivate the optional model cap.
5. [HPS run wiki](https://wiki.jlab.org/hps-run/index.php/The_HPS_Run_Wiki): 2021 beam energy cross-check.
6. [Kubarovsky, 15 November 2021 trigger presentation](https://indico.jlab.org/event/496/contributions/9081/attachments/7389/10202/Kubarovsky_2021_11_15_HPS_trigger.pdf), slide 4: additional acceptance/trigger context. Not digitized or used as the plotted efficiency.
