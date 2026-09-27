# Preserved delivery archive

The payload of `HPS_GPR_v5p8p4_Source_and_Data.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v5p8p4_independent_combinations_windows_20260918/HPS_GPR_v5p8p4_Source_and_Data.zip \
  --output /tmp/HPS_GPR_v5p8p4_Source_and_Data.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `544a7d651e50d1828498d42ef56834725cb05ab2de136cdb4f2a336c1c25946e`. Full manifest: `publication/recent_studies_20260927/archives/544a7d651e50d1828498d42ef56834725cb05ab2de136cdb4f2a336c1c25946e.json`.
