# v4.9.10 2016-full low-control confirmation: terminal at Phase A

This separately versioned prospective follow-up is complete and terminal.  It
does not supersede or modify v4.9.9.

The frozen protocol first checked the production-default upper length-scale
factor 12 on the already-open 2016 10% source for the fixed shortlist
29--210 and 30--210 MeV.  Fifteen of the sixteen support/anchor cells passed.
The 29 MeV support at the 90 MeV anchor had only one eligible warning-free
repeat; its other two deterministic repeats emitted scikit-learn
`ABNORMAL_TERMINATION_IN_LNSRCH` warnings.  Its selected length scale was
0.441704 within the nonbinding interval [0.036243, 0.483242], and its
covariance and other bound checks passed.

This is therefore a non-upper-bound technical failure.  The frozen rule
forbids factor expansion, optimizer retry, or a rule change, so the canonical
decision is `stopped_non_upper_bound_technical_failure`.  No length factor or
support was selected.  The full-2016 low controls were not loaded, fitted, or
scored; no Phase-B confirmation, support freeze, observed scan, signal
extraction, p-value, or limit exists in this release.

Canonical artifacts:

- `STUDY_PROTOCOL.md` and the matching config-side `study_spec.json`;
- `derived/length_factor_12/optimizer_attempts.csv`;
- `derived/length_factor_12/selected_cells.csv`;
- `derived/length_factor_12/run_manifest.json`;
- `derived/length_factor_decision.json`;
- `PHASE_A_TERMINAL_SHA256`;
- `qa/final_validation.json` after running `validate_release.py`.

Reproduce and validate from the repository root with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python3 study_results/v4p9p10_2016_full_low_control_confirmation_20260902/run_qualification.py \
  --mode length --factor 12
python3 study_results/v4p9p10_2016_full_low_control_confirmation_20260902/analyze_qualification.py \
  factor --factor 12
python3 study_results/v4p9p10_2016_full_low_control_confirmation_20260902/validate_release.py
```

Re-running the first two commands will reproduce numerical ledgers but update
timestamp-bearing JSON files and therefore their SHA-256 values.  The canonical
terminal hashes are the ones recorded in `PHASE_A_TERMINAL_SHA256`.
