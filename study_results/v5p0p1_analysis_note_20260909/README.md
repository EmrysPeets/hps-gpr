# HPS-GPR Analysis Note v5.0.1

A polished review draft built from the published v5.0.0 note at commit `3c557e39101c0f73040dc02c46b6555e85d5b4ca`. The user-adopted revision brief is preserved under `provenance/`.

The revision restores historical figures and comparisons, expands the profile-likelihood explanation, adds four method diagrams, combines the correlation matrices, and supplies individual dataset scans and a conditional full-2021 density projection. Every saved observed result remains unchanged, including the connected 232-point limit curve over 19–250 MeV. The historical 2016 95% comparison refers to a confidence level and the full 2016 exposure.

The note remains a draft for unblinding review. Conditional calibration, background-source qualifications and unresolved tails retain their original interpretation. No HPS data, fits or toys were added. The exposure projection retains current observed fluctuations and is not expected sensitivity.

## Deliverables

- `pdf/HPS_GPR_Analysis_Note_v5p0p1_Unblinding_Review_Draft.pdf`
- `editorial/REQUEST_CHECKLIST.md`: each requested change and its final figure/section/page.
- `editorial/MATHEMATICAL_REVIEW_V501_FINAL.md`: implementation-based mathematical review and resolved corrections.
- `MANIFEST.json` and `SHA256SUMS.txt`: artifact identities.
- `qa/final_validation.json`, `qa/visual_review.json`, `qa/portable_build.json`: semantic, visual and independent rebuild checks.

## Rebuild

With Tectonic and its TeX bundle cached, run `bash scripts/build_note.sh` from this folder. The resulting PDF is `qa/build/main.pdf`. Source, bibliography and every included plot are bundled. The `-C` switch prevents network access during the build; on a fresh installation the TeX bundle must first be cached.

The new display figures can be regenerated with Python, NumPy, SciPy, pandas, Matplotlib, scikit-learn and PyYAML using `python3 scripts/make_v501_figures.py`; the method diagrams use `python3 scripts/make_method_diagrams.py`. New numerical figure inputs and necessary implementation snapshots are pinned under `provenance/figure_inputs/`. Historical figures are preserved as archived assets. The complete underlying inference campaigns remain in the frozen repository study directories identified by the source ledgers; rebuilding this document does not rerun them.
