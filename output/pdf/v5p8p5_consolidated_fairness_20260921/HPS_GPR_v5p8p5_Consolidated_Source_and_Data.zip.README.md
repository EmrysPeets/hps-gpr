# Preserved delivery archive

The payload of `HPS_GPR_v5p8p5_Consolidated_Source_and_Data.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v5p8p5_consolidated_fairness_20260921/HPS_GPR_v5p8p5_Consolidated_Source_and_Data.zip \
  --output /tmp/HPS_GPR_v5p8p5_Consolidated_Source_and_Data.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `4ca264dded3e5122ca905d26ec6d9fa9b82bfc95cb50bfe25e4177ec798b53e4`. Full manifest: `publication/recent_studies_20260927/archives/4ca264dded3e5122ca905d26ec6d9fa9b82bfc95cb50bfe25e4177ec798b53e4.json`.
