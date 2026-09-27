# Preserved delivery archive

The payload of `HPS_GPR_v6p3p6_Reproducible_Study.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v6p3p6_readability_2021_20260924/history/before_v16_appendix/HPS_GPR_v6p3p6_Reproducible_Study.zip \
  --output /tmp/HPS_GPR_v6p3p6_Reproducible_Study.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `3293af9d2b88ccce1ae46218d14d3836fbe0ccf07670d5086c4ed17d983d22d3`. Full manifest: `publication/recent_studies_20260927/archives/3293af9d2b88ccce1ae46218d14d3836fbe0ccf07670d5086c4ed17d983d22d3.json`.
