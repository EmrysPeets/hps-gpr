# Preserved delivery archive

The payload of `HPS_GPR_v6p3p6_Reproducible_Study.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v6p3p6_readability_2021_20260924/history/before_blind_panel_combination/HPS_GPR_v6p3p6_Reproducible_Study.zip \
  --output /tmp/HPS_GPR_v6p3p6_Reproducible_Study.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `70ed584e50c00471a13ae1bbd1f39b2158c9deb050fbe6dd2f39229b992c2a9f`. Full manifest: `publication/recent_studies_20260927/archives/70ed584e50c00471a13ae1bbd1f39b2158c9deb050fbe6dd2f39229b992c2a9f.json`.
