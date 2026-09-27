# Preserved delivery archive

The payload of `HPS_GPR_v5p8p5p3_New_Plots.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v5p8p5p3_raw_significance_20260921/HPS_GPR_v5p8p5p3_New_Plots.zip \
  --output /tmp/HPS_GPR_v5p8p5p3_New_Plots.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `7190247e70007e010913dc5389371463cdc766f7d50843e4ff32476b1445c39f`. Full manifest: `publication/recent_studies_20260927/archives/7190247e70007e010913dc5389371463cdc766f7d50843e4ff32476b1445c39f.json`.
