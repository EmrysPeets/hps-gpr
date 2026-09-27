# Preserved delivery archive

The payload of `HPS_v5p7p1_Source_and_Data.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v5p7p1_geometry_acceptance_20260913/HPS_v5p7p1_Source_and_Data.zip \
  --output /tmp/HPS_v5p7p1_Source_and_Data.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `44d2f65954427a7518b3a363e3d58300d80fb3e09227d91e49c7a8721a54aa9b`. Full manifest: `publication/recent_studies_20260927/archives/44d2f65954427a7518b3a363e3d58300d80fb3e09227d91e49c7a8721a54aa9b.json`.
