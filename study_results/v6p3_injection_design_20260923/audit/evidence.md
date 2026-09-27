# File and line evidence

All paths below are original source paths. Copied text retains the same line numbering under `inputs/`. SHA-256 values are recorded in `source_hashes.json`; the portable input derivatives are separately hashed in `input_hashes.json`.

| Finding | Exact source and lines | Portable evidence |
|---|---|---|
| Background-only fit uses actual toy counts; separate Asimov error is also computed | `/Users/emryspeets/Desktop/gp_mods/hps-gpr-v5-analysis-note-20260908/study_results/v4p6_2021_exposure_refmatched_100toy_20260812/run_study.py`, 289–310 | `inputs/01_run_study.py` |
| Per-toy yield uses actual selected sigma_A; signal seed depends on strength | Same file, 340–343 | Same copy |
| Same fixed background plus Poisson signal | Same file, 367–378 | Same copy |
| Pull uses expected injected yield and post-injection sigma | Same file, 391–409 | Same copy |
| Accepted sigmaA_ref stores selected reference sigma_A | Same file, 463–478 | Same copy |
| Helper draws Poisson in each signal bin | `/Users/emryspeets/Desktop/gp_mods/hps-gpr-v5-analysis-note-20260908/hps_gpr/injection.py`, 78–85 | `inputs/03_injection.py` |
| Older recovery display uses raw Ahat/Ainj | `/Users/emryspeets/Desktop/gp_mods/hps-gpr-v5-analysis-note-20260908/study_results/v4p6_2021_exposure_refmatched_100toy_20260812/make_figures.py`, 239–246; source rows loaded at 442 | `inputs/02_make_figures.py` |
| v4.9.5 descendant still uses actual reference sigma, Poisson signal, postfit pull | `/Users/emryspeets/Desktop/gp_mods/hps-gpr-note-v505-20260916/study_results/v4p9p5_2021_gp_support_edge_optimization_20260820/run_support_scan.py`, 1074–1104, 1173–1217, 1288–1297 | `inputs/05_run_support_scan.py` |
| v5.0.5 defines source/toy/mass matched reference and paired subtraction | Repository `study_results/v5p0p5_analysis_note_20260916/source/sections/04_methodology.tex`, 898–929 | `inputs/04_04_methodology.tex` |
| Audited baseline is represented in v5.0.5, distinct from targeted replacements | Repository `study_results/v5p0p5_analysis_note_20260916/source/sections/05_toys_validation.tex`, 430–435, 472–476, 497–505; reproduced baseline means0.7890 and0.7160 | `inputs/07_05_toys_validation.tex` |
| Consolidation explicitly archives v4.6 historical baseline | `/Users/emryspeets/Desktop/gp_mods/hps-gpr-note-v505-20260916/study_results/v4p9p1_2021_background_validation_consolidation_20260817/README.md`, 3, 48–54 | `inputs/08_consolidation_README.md` |
| v6.2 fixed N, fixed archived kernel, Poisson background and exact-N multinomial signal | Repository `study_results/v6p2_mc_injection_20260923/scripts/injection_core.py`, 21, 69–73, 142–160 | `inputs/06_injection_core.py` |

Numerical source: `/Users/emryspeets/Desktop/gp_mods/hps-gpr-v5-analysis-note-20260908/study_results/v4p6_2021_exposure_refmatched_100toy_20260812/derived/accepted_extraction_rows.csv`. Its original header is line1, data span lines2–7995. Portable input retains only the13 columns necessary for this audit, including original `_source_line`; all row counts, source grouping and pairing are retained.
