# Preserved delivery archive

The payload of `HPS_GPR_v5p8p2_Source_and_Data.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v5p8p2_nominal_gp_significance_20260917/HPS_GPR_v5p8p2_Source_and_Data.zip \
  --output /tmp/HPS_GPR_v5p8p2_Source_and_Data.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `e82f926261910ae43705a4e455b4a2013083e7ce8b682111c7479640fa6262ba`. Full manifest: `publication/recent_studies_20260927/archives/e82f926261910ae43705a4e455b4a2013083e7ce8b682111c7479640fa6262ba.json`.
