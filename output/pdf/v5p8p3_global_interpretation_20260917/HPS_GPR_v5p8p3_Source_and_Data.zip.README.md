# Preserved delivery archive

The payload of `HPS_GPR_v5p8p3_Source_and_Data.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v5p8p3_global_interpretation_20260917/HPS_GPR_v5p8p3_Source_and_Data.zip \
  --output /tmp/HPS_GPR_v5p8p3_Source_and_Data.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `fe246a8e6b7f828bcfbe71cc81c81abb63f8fbad6691b757892eeea3f653b0c5`. Full manifest: `publication/recent_studies_20260927/archives/fe246a8e6b7f828bcfbe71cc81c81abb63f8fbad6691b757892eeea3f653b0c5.json`.
