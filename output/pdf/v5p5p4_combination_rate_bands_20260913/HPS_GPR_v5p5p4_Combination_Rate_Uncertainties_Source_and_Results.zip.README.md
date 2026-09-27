# Preserved delivery archive

The payload of `HPS_GPR_v5p5p4_Combination_Rate_Uncertainties_Source_and_Results.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v5p5p4_combination_rate_bands_20260913/HPS_GPR_v5p5p4_Combination_Rate_Uncertainties_Source_and_Results.zip \
  --output /tmp/HPS_GPR_v5p5p4_Combination_Rate_Uncertainties_Source_and_Results.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `b13ccdf46cdb65168619d2ee3abdbf2c899e4e9bc3cc71aa1fcef397f05f1b4b`. Full manifest: `publication/recent_studies_20260927/archives/b13ccdf46cdb65168619d2ee3abdbf2c899e4e9bc3cc71aa1fcef397f05f1b4b.json`.
