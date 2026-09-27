# Preserved delivery archive

The payload of `HPS_GPR_v6p3p9_Reproducible_Study.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v6p3p9_combined_morph_20260925/HPS_GPR_v6p3p9_Reproducible_Study.zip \
  --output /tmp/HPS_GPR_v6p3p9_Reproducible_Study.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `9002275c50a601e23b948585d887ba607368d96607069f22b56e6668a0ad8e99`. Full manifest: `publication/recent_studies_20260927/archives/9002275c50a601e23b948585d887ba607368d96607069f22b56e6668a0ad8e99.json`.
