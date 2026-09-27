# Preserved delivery archive

The payload of `HPS_GPR_v6p2_MC_Injection_Recovery_Source_Data.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive study_results/v6p2_mc_injection_20260923/history/20_toy_release/delivered/HPS_GPR_v6p2_MC_Injection_Recovery_Source_Data.zip \
  --output /tmp/HPS_GPR_v6p2_MC_Injection_Recovery_Source_Data.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `1ddcbe2a429aafbc4740fdfeb7ed0ec196311ea82cb791cf611369596afe3673`. Full manifest: `publication/recent_studies_20260927/archives/1ddcbe2a429aafbc4740fdfeb7ed0ec196311ea82cb791cf611369596afe3673.json`.
