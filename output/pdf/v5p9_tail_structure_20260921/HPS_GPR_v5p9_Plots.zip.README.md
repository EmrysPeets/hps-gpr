# Preserved delivery archive

The payload of `HPS_GPR_v5p9_Plots.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v5p9_tail_structure_20260921/HPS_GPR_v5p9_Plots.zip \
  --output /tmp/HPS_GPR_v5p9_Plots.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `dd75dbb1ad6f4156ead46bb7498e6af6b7a7e15c0df23c0e26d2ea7a935914ad`. Full manifest: `publication/recent_studies_20260927/archives/dd75dbb1ad6f4156ead46bb7498e6af6b7a7e15c0df23c0e26d2ea7a935914ad.json`.
