# Preserved delivery archive

The payload of `HPS_GPR_v6p4_2016_MC_Shapes_Reproducible_Study.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v6p4_2016_mc_shapes_20260925/HPS_GPR_v6p4_2016_MC_Shapes_Reproducible_Study.zip \
  --output /tmp/HPS_GPR_v6p4_2016_MC_Shapes_Reproducible_Study.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `4b537a77a85eb190cb7e78763c697217a945e540d1963765348726d0f7cd07f7`. Full manifest: `publication/recent_studies_20260927/archives/4b537a77a85eb190cb7e78763c697217a945e540d1963765348726d0f7cd07f7.json`.
