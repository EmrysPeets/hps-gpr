# Preserved delivery archive

The payload of `HPS_GPR_v6p4p4_Calibrated_Local_Global_Source.zip` is preserved in Git without storing a duplicate ZIP container. Every member has a recorded SHA-256 and stored source.

From the repository root, run:

```bash
python3 publication/recent_studies_20260927/scripts/restore.py \
  --archive output/pdf/v6p4p4_calibrated_local_global_20260925/HPS_GPR_v6p4p4_Calibrated_Local_Global_Source.zip \
  --output /tmp/HPS_GPR_v6p4p4_Calibrated_Local_Global_Source.zip
```

The rebuilt ZIP has identical member names and bytes; container metadata or compression may differ. Original ZIP SHA-256: `64faecd27123ca753c071f1de97e29c142dc0f44712013dafda1381b227f66df`. Full manifest: `publication/recent_studies_20260927/archives/64faecd27123ca753c071f1de97e29c142dc0f44712013dafda1381b227f66df.json`.
