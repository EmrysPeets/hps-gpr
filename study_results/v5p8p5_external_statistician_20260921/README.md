# A fresh statistician's reading of the HPS studies

A three-page perspective report addressing which regions deserve follow-up, the meaning of a fixed background, and what genuinely additional 2021 data can clarify. It is a role-based statistical reading of saved results, not an independent experimental certification or a new fit.

Main judgement: 90–93 MeV merits replication and rate-consistency tests; 65–67 MeV is the cleaner shared-coupling candidate but currently weak globally. At 92 MeV the 2021 sample supplies 82% of local null information yet has the smallest fitted coupling. More 2021 data can test the predicted rate and distinguish fluctuations from persistent structure. Reproducible model bias still needs separation from signal.

`source/report.pdf` is the report. `source/report.tex` is the LaTeX source. `figures/observed_evidence.pdf` is the original vector figure. `inputs/` holds the exact saved tables/note sections used, and `results/provenance.json` identifies them. `reviews/` contains the statistician's assessment and the separate fixed-background audit, including an auxiliary-ledger caution that was avoided in the figure.

Rebuild the PDF from this directory:

```bash
tectonic -X compile source/report.tex --keep-logs
```

Rebuild the figure and numerical excerpts (NumPy, pandas, Matplotlib required):

```bash
python3 scripts/make_evidence.py
```

The figure script uses bundled inputs outside the original checkout. No observed likelihoods, sources, masks, significances or releases were changed. All quoted significance mappings are identified as raw/asymptotic or source-conditional. The background fixed inside the likelihood is explicitly distinguished from a fixed source used for generating toys; a numerical fixed-covariance discovery scan was not recovered and is not inferred.
