# Preserved delivery archive

The payload of `HPS_GPR_v6p3p5_Reproducible_Study.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v6p3p5_unified_2021_20260924/HPS_GPR_v6p3p5_Reproducible_Study.zip \
  --output /tmp/HPS_GPR_v6p3p5_Reproducible_Study.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `dab34a4c366ff1b62332053093f1e0ba08cf80ee316473dfa0a5e8739783a692`. Full manifest: `publication/recent_studies_20260927/archives/dab34a4c366ff1b62332053093f1e0ba08cf80ee316473dfa0a5e8739783a692.json`.
