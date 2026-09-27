# Preserved delivery archive

The payload of `HPS_GPR_v5p0p4_Figure2_Source_and_Results.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v5p0p4_figure2_publication_20260913/HPS_GPR_v5p0p4_Figure2_Source_and_Results.zip \
  --output /tmp/HPS_GPR_v5p0p4_Figure2_Source_and_Results.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `cef86fe0224da33de50b6c0f37089f7e578d0a1395ef82a3f69bd4218ac60a52`. Full manifest: `publication/recent_studies_20260927/archives/cef86fe0224da33de50b6c0f37089f7e578d0a1395ef82a3f69bd4218ac60a52.json`.
