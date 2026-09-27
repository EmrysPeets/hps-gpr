# Preserved delivery archive

The payload of `HPS_GPR_v6p3p8_Reproducible_Study.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v6p3p8_observed_morph_20260925/HPS_GPR_v6p3p8_Reproducible_Study.zip \
  --output /tmp/HPS_GPR_v6p3p8_Reproducible_Study.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `c9c6b7aec15f46fba94815627a2f2b39af5a439970f6fb5eef703c8059cf32ad`. Full manifest: `publication/recent_studies_20260927/archives/c9c6b7aec15f46fba94815627a2f2b39af5a439970f6fb5eef703c8059cf32ad.json`.
