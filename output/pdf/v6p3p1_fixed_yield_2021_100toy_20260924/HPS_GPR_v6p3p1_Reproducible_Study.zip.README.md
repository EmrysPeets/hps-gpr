# Preserved delivery archive

The payload of `HPS_GPR_v6p3p1_Reproducible_Study.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v6p3p1_fixed_yield_2021_100toy_20260924/HPS_GPR_v6p3p1_Reproducible_Study.zip \
  --output /tmp/HPS_GPR_v6p3p1_Reproducible_Study.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `500b4927da8f8aa04ec92bf783935ad6837acc4f8a3f3f3fc259ad07ad636990`. Full manifest: `publication/recent_studies_20260927/archives/500b4927da8f8aa04ec92bf783935ad6837acc4f8a3f3f3fc259ad07ad636990.json`.
