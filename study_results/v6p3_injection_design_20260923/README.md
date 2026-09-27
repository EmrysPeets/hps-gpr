# HPS-GPR v6.3: Reference matching and fixed-yield injections

Read `pdf/HPS_GPR_v6p3_Injection_Design.pdf` for the mathematical and plain-language report.

Per-toy matching, A_t = z sigma_ref,t, is a legitimate adaptive injection–recovery diagnostic. A common expected yield, A = z s0, is the clearer primary design for testing performance at a fixed physical signal rate. Choose s0 from an independent background-only pilot ensemble within each dataset/mass/template cell, freeze it, and use each injected fit's own returned error in the pull. Matching is not itself a cure for bias, and both designs coincide at zero injection.

The report separates fixed expected yield from exact-N injection, derives the response-covariance difference between designs, and explains raw recovery versus paired response. The historical v4.6 full-100 baseline retained in v5.0.5 uses Poisson signal counts and has a 1.17–3.00% reference-error CV across 20 cells. This is a scoped audit of saved rows, not a new fixed-yield HPS fit or coverage calibration. Later targeted threshold replacements are not pooled.

## Contents

- `source/report.tex`, `source/archive_table.tex`: editable LaTeX and generated numerical table.
- `figures/injection_design.pdf`: vector illustration; the illustrative CV is 20%, not the measured HPS spread.
- `data/`: exact two-state example and analytic distribution summaries.
- `audit/`: portable minimal archived rows, original row numbers, copied code evidence, reference summaries, paired/raw recovery summaries, exact source lines, hashes, and specialist review.
- `scripts/`: analytic illustration and table generation; no new HPS fitting.
- `provenance/`: sources and runtime identity.
- `qa/`: numerical checks, extracted text, rendered pages, and final QA record.
- `MANIFEST.sha256`: hashes of the delivered snapshot, excluding this manifest itself.

## Reproduce

Requires Python 3 with NumPy, SciPy, Matplotlib and pandas, plus Tectonic with cached TeX resources. No network data or HPS scientific-fit runtime is needed. From this directory:

```sh
PYTHON=/Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
TECTONIC=/opt/homebrew/bin/tectonic bash reproduce.sh
```

Omit overrides when the tools are on PATH. Tectonic runs in cached-only mode (`-C`); on another machine first populate its TeX resource cache or remove `-C` to allow the initial resource download. The scripts limit numerical libraries to one thread. All calculations use packaged inputs; the audit also checks original source hashes read-only if those original paths still exist.

Verify the delivered package before rebuilding:

```sh
shasum -a 256 -c MANIFEST.sha256
```

Rebuilding may change PDF metadata. The manifest records the delivered snapshot. `qa/final_qa.json` records semantic and rendered-page checks; rendering uses Poppler and text inspection uses pypdf. The root deliverable copy is also under `output/pdf/v6p3_injection_design_20260923/` in the originating checkout.
