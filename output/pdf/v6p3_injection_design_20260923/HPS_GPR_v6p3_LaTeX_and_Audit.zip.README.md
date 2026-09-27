# Preserved delivery archive

The payload of `HPS_GPR_v6p3_LaTeX_and_Audit.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v6p3_injection_design_20260923/HPS_GPR_v6p3_LaTeX_and_Audit.zip \
  --output /tmp/HPS_GPR_v6p3_LaTeX_and_Audit.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `bf071118149931a800a7eb5e466bb9277a2491c207ede59dbd7eaf9d416e31fa`. Full manifest: `publication/recent_studies_20260927/archives/bf071118149931a800a7eb5e466bb9277a2491c207ede59dbd7eaf9d416e31fa.json`.
