# HPS GPR recent-study logbook

**Start with the [32-page plotbook](../../output/pdf/study_logbook_20260927/HPS_GPR_Recent_Studies_Plot_Logbook.pdf).** It collects 26 selected figures with findings, reading instructions, speaking notes, scientific qualifications and source links. The main catalogue covers 40 study and presentation entries from 13-27 September 2026, with the previously pushed v5.0.5 note and its older study archive included in the publication history.

| Need | Open |
|---|---|
| Browse the figures in GitHub | [Illustrated plot guide](PLOTBOOK.md) |
| Search/filter the local figure collection | [Offline HTML gallery](index.html) - open locally after cloning; GitHub displays its source |
| Find what each series accomplished | [Complete study log](STUDY_LOG.md) |
| Prepare a short presentation | [12-minute route and speaking notes](PRESENTATION_NOTES.md) |
| Read the integrated 2021 explanation | [v6.3.6 note](../../study_results/v6p3p6_readability_2021_20260924/pdf/HPS_GPR_v6p3p6_2021_Signal_Extraction.pdf) |
| Read the latest local/global and correlation discussion | [v6.4.5 note, retaining v6.4.4](../../study_results/v6p4p5_mass_correlation_lee_20260926/pdf/report.pdf) |
| Revisit the latest saved presentation | [23 September follow-up](../../output/slides/unblind_meeting_RCmeet_20260923b/CHANGELOG.md) |
| Check exactly what was preserved | [Publication archive and restoration guide](../../publication/recent_studies_20260927/README.md) |

The results support a coherent story: broad and displaced signal-MC shapes matter; paired injection and clean-training controls expose response loss; reducing a raw upper limit is not sufficient to establish improved sensitivity; and global inference needs the correlations and mass-dependent local behavior of the specified search.

The v6.4.4 common-coupling local-first result has 75/1,024 independent validation exceedances: add-one global p = 0.07415, with exact two-sided 95% binomial interval [0.05804, 0.09095], conditional on the frozen local maps and background sources. This is one joint common-coupling search. Its distinct raw-maximum result, the earlier local asymptotic scans, independent-amplitude combinations, and the supplementary choice across searches must retain their own labels.

Selected originals are in [assets](assets), with vector PDF companions wherever available. [plot_cards.json](plot_cards.json) records each caption and source; [catalog.json](catalog.json) is the study index. [source_manifest.json](source_manifest.json) pins figure and evidence hashes. [qa/source_claim_checks.json](qa/source_claim_checks.json) records the numerical checks performed for this publication. Historical study QA remains inside each original package; publication checks do not rerun or certify every scientific ensemble.

Rebuild with Python 3, Pillow, pypdf and reportlab:

```bash
python3 docs/study_logbook_20260927/build_logbook.py
python3 docs/study_logbook_20260927/validate_logbook.py
```

This build reads saved results and figures only. It runs no HPS likelihood fits, remote jobs or new toy ensembles. Rebuilding the PDF may change document metadata; compare text, figure payloads and the documented source checks. The PDF's source links use the publication tag `studies-2026-09-27`.
