# Preserved delivery archive

The payload of `HPS_GPR_v6p2_MC_Injection_Recovery_Source_Data.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v6p2_mc_injection_20260923/HPS_GPR_v6p2_MC_Injection_Recovery_Source_Data.zip \
  --output /tmp/HPS_GPR_v6p2_MC_Injection_Recovery_Source_Data.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `e5f3be919e47e7c070ac4050e6167345a70074fe8f559f2dc41dcaeee2c35ffb`. Full manifest: `publication/recent_studies_20260927/archives/e5f3be919e47e7c070ac4050e6167345a70074fe8f559f2dc41dcaeee2c35ffb.json`.
