# HPS-GPR v5.1.1: covariance and signed injection pilot

Support study for the frozen v5.0.2 analysis note, completed 10 September 2026.
Read `report.pdf` (six pages); numerical ledgers are under `derived/`.

- Conditional covariance grids: 2015 19–100 MeV (82 nodes); 2016 39–180 MeV plus 70.5–80.5 half-integers (153 nodes); 2021 50–250 MeV (201 nodes).
- Each empirical matrix uses the same first 256 archived whole-spectrum Poisson experiments at every node. The original 1,000-experiment native matrices remain separate pinned inputs.
- Nine declared injection masses, two generating backgrounds, strengths 0, 2 and 5 reference errors, and 64 paired experiments: 3,456 scenario labels from 1,152 background spectra and 2,304 independent signal-increment arrays.
- Full signed roots, unconstrained amplitudes, deterministic signal scans, negative neighboring echoes, template-weighted response predictions, and fixed-training controls are saved.
- Five figures develop correlations, modes, bootstrap slices, input-bin response directions and injected-signal behavior.

The 2016 stress offset remains substantial. This is fixed-mass conditional recovery and covariance validation, not calibrated global significance, discovery power, or an enlargement of the released observed search. The original v5.0.2 note and released results are preserved. Read the report for the connections to v5.1.0 and v5.2.0–2.

## Reproduce the figures and report

Requires Python with numpy, scipy, pandas, matplotlib and pypdf; Tectonic with its cached LaTeX resources. No external HPS runtime is needed for these saved-array steps:

```sh
python3 scripts/summarize.py
python3 scripts/make_report.py
bash scripts/build.sh
```

`python3 scripts/validate.py` additionally checks the recorded original HPS archives. Full refitting with `scripts/run_pilot.py` needs the HPS runtime and original data paths in `scripts/common.py`; these are disclosed dependencies, not bundled portable inputs. Every copied input is SHA-256 pinned in `inputs/manifest.json`. The fit scripts, seed convention and source hashes are included.

## Resource and provenance record

One worker and one linear-algebra thread were used, at reduced scheduling priority. Peak resident memory recorded in the final continuation was 835 MiB. `initial_protocol.json` preserves the initial optional grid; `protocol.json` records a timing-only amendment reducing optional half-steps and covariance validation to 256 coherent experiments. Completed initial-plan checkpoints are retained, but only nodes listed in each final `field.npz` enter the reported results. No coordinate was selected using a favorable response or probability.

`qa/run.log` records the initial phase and its intentional interruption; `qa/run_bounded.log` records successful completion. The 312-second execution duration is the final checkpoint continuation, not the total pilot runtime. `qa/validation.json` contains independent numerical and PDF checks; `qa/visual_review.json` records inspection of the rendered six-page report. `qa/portable_build.json` verifies a clean report build from source and figure assets.

The companion note derivative adds Section 6.11 material and three figures while preserving the previous figures and result state. Its package is separate from this study archive.
