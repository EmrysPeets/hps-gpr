# Preserved delivery archive

The payload of `HPS_v5p7p1_LaTeX_Package.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v5p7p1_geometry_acceptance_20260913/HPS_v5p7p1_LaTeX_Package.zip \
  --output /tmp/HPS_v5p7p1_LaTeX_Package.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `3d7957b5d501806e609e5782d1267f293b23cc14997eb1509f14c707e70527ec`. Full manifest: `publication/recent_studies_20260927/archives/3d7957b5d501806e609e5782d1267f293b23cc14997eb1509f14c707e70527ec.json`.
