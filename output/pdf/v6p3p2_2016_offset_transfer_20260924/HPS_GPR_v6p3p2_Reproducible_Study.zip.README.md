# Preserved delivery archive

The payload of `HPS_GPR_v6p3p2_Reproducible_Study.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v6p3p2_2016_offset_transfer_20260924/HPS_GPR_v6p3p2_Reproducible_Study.zip \
  --output /tmp/HPS_GPR_v6p3p2_Reproducible_Study.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `9583265f90ce1eebd0d5b9062cdf515147294b0366e51d397b75d7d047b33fc0`. Full manifest: `publication/recent_studies_20260927/archives/9583265f90ce1eebd0d5b9062cdf515147294b0366e51d397b75d7d047b33fc0.json`.
