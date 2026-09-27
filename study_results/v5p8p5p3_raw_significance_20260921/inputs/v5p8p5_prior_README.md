# HPS-GPR v5.8.5 consolidated significance review

21 September 2026. A 17-page LaTeX report with an opening schematic, reviewed jointly by statistics, particle-physics analysis, and study-history specialists. The primary PDF is `source/report.pdf`; released copies are under the corresponding `output/pdf/` directory. Previous releases remain unchanged.

## Findings

- The observed 2016 raw root remains 3.4525 at 90.5 MeV. The 2.886 maximum at 91.5 MeV includes mass-dependent reference centering/scaling before the look-elsewhere calculation. It is not adopted as a production replacement.
- Keeping raw-maximum ordering gives fixed-source Gaussian global p=0.067795 (18/256 direct Poisson exceedances); reference ordering gives p=0.104875 (36/256). These are different conditional tests, not options to select for the smallest observed p-value.
- On the same 2016 field and threshold, the upcrossing bound is 0.120378, above the GP maximum probability. This matches the paper's less-conservative comparison. Resolution-count Sidak tests a different approximation.
- The proposed |rho|<0.1 rule does not establish independent regions. A 24-point construction retains distant correlations of 0.835 and omits within-region maxima. Point-Sidak gives 0.04579, an independence product of actual block maxima gives 0.14341, and the joint maximum is 0.10640 in 500,000 new fields.
- The genuine 2021 high-psum 1% histogram is available, but its selection/exposure/overlap transfer is not qualified. Nominal scaling by ten gives only 13–52% of the native10 source across the search. Source rebuilding still absorbs 78–84% of injected window yield even with small deterministic offsets; fixed-source extraction recovers 96.7%.
- All new fits retain the ±2.25 sigma moving exclusion. No new width scan, physical significance calibration, exclusion curve, or discovery-reach claim is supplied.

## Reading guide

- Page 1: opening schematic.
- Pages 2–4: decisions, v5.8.0–.5 ledger, v5.0.5 §§4.20/6.11, exact 2016 remapping.
- Pages 4–6: explanation of v5.8.3 Figures 4 and 5.
- Pages 7–8: deterministic offset and v5.8.2 Figure 3 unpacked.
- Pages 9–11: genuine high-psum source, bounded toys, source rebuilding and uncertainty.
- Pages 12–14: raw ordering, same-field upcrossing bound, correlation regions.
- Pages 15–17: next calibration, discovery versus exclusion reach, combinations, provenance.

## Rebuild the PDF from included vector figures

Requires Tectonic. From this directory:

```bash
bash scripts/build_report.sh
```

## Reproduce the new numerical calculations

Requires Python with NumPy, SciPy, pandas, Matplotlib, uproot, and pypdf. Validated here with `/Applications/Xcode.app/Contents/Developer/usr/bin/python3`; use a suitable local interpreter on another machine. The scripts prefer bundled inputs and run one numerical thread. Relative per-run timeouts and the `STOP` file protect computations; the development-only `resource_guard.py` is not part of reproduction.

```bash
python3 scripts/correlation_diagnostic.py
python3 scripts/physics_source_diagnostic.py
python3 scripts/physics_signal_transfer.py
python3 scripts/physics_figures.py
python3 scripts/make_overview.py
bash scripts/build_report.sh
python3 scripts/validate_report.py
```

The statistical script reuses saved 200,000-draw maxima and 256 full Poisson scans, and generates a reproducible 500,000-field 2016 block diagnostic. The physics script performs 603 deterministic source/mass fits and 768 Poisson fits (64 complete spectra per source, six anchors), followed by two injection anchors. These are not full new discovery-tail calibrations. New numerical outputs have no dependency on a live parent checkout. Both physics and report reproduction were checked in separate temporary directories.

## Contents and provenance

- `source/`: LaTeX sections and rendered PDF.
- `figures/`: original archived figures and newly generated vector PDF/PNG pairs.
- `results/`: exact new results, inherited study ledger, fit and sampling QA.
- `reviews/`: the three specialists' audits and cross-review findings.
- `inputs/`: immutable copied fields, source/code/result snapshots, the actual high-psum ROOT histogram and its selection audit.
- `inputs/parent_manifest.json`: initial hashes of 3,782 original files.
- `qa/`: text/numerical validation, final page renders, external rebuild checks and parent immutability audit.
- `DELIVERY.json`: completion status, resource snapshots and work left unqualified.
- `SHA256SUMS.txt`: this bundle's file identities.

The package is self-contained for rendering the report and rerunning the **new** numerical work. Historical studies are included as selected code/data/field snapshots and results; the package does not claim to include all prerequisites for every historical fit. The original v5.8.5 mass-coherence release remains separate and unchanged.

Gaussian/global sampling intervals are Monte Carlo intervals only. The source construction, exposure/cut equivalence, possible learned signal, and method/domain selection remain inference qualifications. The report specifies the prospective full-procedure validation rather than claiming it was completed within this bounded review.
