# Preserved delivery archive

The payload of `HPS_GPR_v6p3p6_Reproducible_Study.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v6p3p6_readability_2021_20260924/history/before_v637_appendix/HPS_GPR_v6p3p6_Reproducible_Study.zip \
  --output /tmp/HPS_GPR_v6p3p6_Reproducible_Study.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `bce986eadb34c2e9fbb98bb25b314c62ccb19290094ba7994c4758c12752b7b3`. Full manifest: `publication/recent_studies_20260927/archives/bce986eadb34c2e9fbb98bb25b314c62ccb19290094ba7994c4758c12752b7b3.json`.
