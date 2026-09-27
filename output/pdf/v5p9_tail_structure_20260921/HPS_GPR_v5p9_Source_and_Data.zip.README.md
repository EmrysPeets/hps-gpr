# Preserved delivery archive

The payload of `HPS_GPR_v5p9_Source_and_Data.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v5p9_tail_structure_20260921/HPS_GPR_v5p9_Source_and_Data.zip \
  --output /tmp/HPS_GPR_v5p9_Source_and_Data.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `6e4f1b32e9fed22b44963dd7ba3369d1b2bcd3d9449c5595547742caab4bb0b6`. Full manifest: `publication/recent_studies_20260927/archives/6e4f1b32e9fed22b44963dd7ba3369d1b2bcd3d9449c5595547742caab4bb0b6.json`.
