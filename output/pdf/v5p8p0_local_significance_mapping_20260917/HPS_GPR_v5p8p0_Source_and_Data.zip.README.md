# Preserved delivery archive

The payload of `HPS_GPR_v5p8p0_Source_and_Data.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v5p8p0_local_significance_mapping_20260917/HPS_GPR_v5p8p0_Source_and_Data.zip \
  --output /tmp/HPS_GPR_v5p8p0_Source_and_Data.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `9ccaa191835319aeae0a5025aab952db518e6d66ea60b51c8d6ebc6f6b60c34c`. Full manifest: `publication/recent_studies_20260927/archives/9ccaa191835319aeae0a5025aab952db518e6d66ea60b51c8d6ebc6f6b60c34c.json`.
