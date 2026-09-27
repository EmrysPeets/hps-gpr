# HPS-GPR v6.3.7: v16 signal templates and fit windows

This package studies the 2021 10% TC analysis using the verified v16 signal-MC inputs. It compares a shifted Gaussian, a common aligned empirical distribution, interpolation between neighboring signal-MC samples, and a direct signal-MC benchmark. The complete signal distribution generates injected counts; selecting a shorter fit interval never renormalizes its probabilities.

The study is conditional on the archived background sources, GP kernel settings and v16 selected signal samples. It does not update the observed-data limits in the main v6.3.6 note, claim unconditional coverage, or perform a UC limit analysis.

## Rebuilding

Use Python 3 with NumPy, SciPy, pandas and matplotlib, and Tectonic for LaTeX. The local verified scientific runtime is `/Applications/Xcode.app/Contents/Developer/usr/bin/python3`. Set numerical thread counts to one. All required numerical inputs are included; S3DF access is unnecessary.

The final release records exact commands, cohort sizes, hashes and numerical checks in the protocol, results and QA files. Figures and report are rebuilt from saved numerical results; rerunning toys is a separate explicit command.

## Contents

- `inputs/`: immutable signal-MC histograms, fitted core summaries, background source and archived numerical implementation.
- `scripts/`: template construction, numerical study, figure generation and report construction.
- `results/`: candidate geometry, selection and independent toy-evaluation records.
- `provenance/`: input hashes and study configuration.
- `qa/`: numerical, portable rebuild and rendered-page checks.

The same study is embedded in the appendices of the v6.3.6 note. The original v6.3.6 release with appendices A-C is preserved under the output directory's `history/before_v637_appendix/`.

## Validated rebuild commands

Run `V637_PYTHON=/Applications/Xcode.app/Contents/Developer/usr/bin/python3 bash scripts/rebuild.sh` from this package. The default `python3` is usable when the listed scientific packages are installed. Saved checkpoint hashes are checked before reusing fits. To regenerate all toys, copy the package, preserve the existing results as provenance, then remove the copied checkpoint directory; the frozen protocol must remain unchanged. The serialization correction and complete numerical reproduction are recorded in `provenance/resume_fix/` and `qa/resume_fix_reproduction.json`.

For a report-only build, run `python3 scripts/make_figures.py` followed by `python3 scripts/build_report.py --build`. Tectonic is found on PATH, can be set with `TECTONIC`, or falls back to the verified Homebrew path. The compiler uses cached TeX resources; on a new machine, populate the Tectonic cache once before using `--only-cached`. `scripts/replay_fits.py` independently reconstructs six spectra from saved counts and repeats their numerical fits.

The numerical conclusion is improved yield response and modest average precision changes, not a universal upper-limit improvement. The proposed starter exclusion retains appreciable signal in GP training. Endpoint medians explicitly omit empty accepted sets; all rejection fractions retain denominator 100.
