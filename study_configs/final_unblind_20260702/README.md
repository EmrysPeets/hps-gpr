# Final Unblind Combined Upper-Limit Configs

These configs are for the final pre-unblinding SDF upper-limit pass using the
review-committee closure-study settings as of 2026-07-02.

Primary choices:

- 90% CL observed upper limits: `cls_alpha: 0.1`.
- Corrected combined coupling coordinate: `combined_mode: count_scale`.
- 2021 input: `/sdf/home/e/epeets/run/2021_bump/final_1pct_invM.root`.
- 2021 histogram: `preselection/h_invM_8000`.
- 2021 scaled mass resolution: `1.25 * [0.0014786, -0.0011, 0.0687]`.
- Scan/support ranges: 2015 `19-90 MeV`, 2016 `39-180 MeV`, 2021 `50-250 MeV`.
- Length-scale lower bounds: 2015 `1.0 sigma`, 2016 `0.9 sigma`, 2021 `1.1 sigma`.
- Radiative penalties: 2015 `7%`, 2016 `7%`, 2021 `4.6%`.
- Expected bands: 2021 individual bands plus combined bands only. No 2015 or
  2016 individual bands are produced by the primary combined config.

## Configs

- `config_obsUL90_combined_finalpass_search50_countscale_bands2021_combined.yaml`
  - Main SDF run.
  - Enables 2015, 2016, and 2021.
  - Produces single-dataset observed scans, the corrected combined observed
    scan, 2021 individual expected bands, and combined expected bands.
  - Uses `run_limit_bands_on: '2021'` and `do_combined_bands: true`.

- `config_obsUL90_2021_finalpass_search50_bands.yaml`
  - Optional cheaper 2021-only cross-check.
  - Uses the same 2021 path, range, scaled resolution, length-scale bound, and
    4.6% radiative penalty.

## SDF Update

Use this on the SDF checkout to preserve local SDF edits while taking the final
config branch:

```bash
cd /path/to/hps-gpr
git status --short
git stash push -u -m "before final unblind UL configs"
git fetch origin
git checkout codex/final-unblind-combined-ul-configs-20260702
git pull --ff-only origin codex/final-unblind-combined-ul-configs-20260702
git stash pop
```

If `git stash pop` reports conflicts, resolve them before generating SLURM
jobs. If there were no local SDF edits, the stash command may report that there
is nothing to save; in that case skip `git stash pop`.

## Generate Jobs

From the repo root on SDF:

```bash
mkdir -p slurm/final_unblind_20260702/combined
CONFIG_ABS="$PWD/study_configs/final_unblind_20260702/config_obsUL90_combined_finalpass_search50_countscale_bands2021_combined.yaml"

hps-gpr slurm-gen \
  --config "$CONFIG_ABS" \
  --n-jobs 232 \
  --job-name hps_obsUL90_final_comb \
  --partition roma \
  --account hps:hps-prod \
  --time 24:00:00 \
  --memory 8G \
  --output slurm/final_unblind_20260702/combined/submit_obsUL90_final_combined_20260702.slurm

slurm/final_unblind_20260702/combined/submit_all.sh
```

The `232` jobs cover the union of the enabled scan ranges, `19-250 MeV`, at
`1 MeV` spacing.

Optional 2021-only cross-check:

```bash
mkdir -p slurm/final_unblind_20260702/2021
CONFIG_ABS="$PWD/study_configs/final_unblind_20260702/config_obsUL90_2021_finalpass_search50_bands.yaml"

hps-gpr slurm-gen \
  --config "$CONFIG_ABS" \
  --n-jobs 201 \
  --job-name hps2021_obsUL90_final \
  --partition roma \
  --account hps:hps-prod \
  --time 24:00:00 \
  --memory 8G \
  --output slurm/final_unblind_20260702/2021/submit_obsUL90_final_2021_20260702.slurm

slurm/final_unblind_20260702/2021/submit_all.sh
```

## Combine Finished Jobs

Run these after the corresponding SLURM jobs finish:

```bash
hps-gpr slurm-combine \
  --output-dir /sdf/data/hps/users/epeets/hps_gpr/final_unblind_20260702/obsUL90_combined_search50_countscale_sig2021x1p25_lslb2015_1p0_2016_0p9_2021_1p1_rpen7_7_4p6

hps-gpr slurm-combine \
  --output-dir /sdf/data/hps/users/epeets/hps_gpr/final_unblind_20260702/obsUL90_2021_search50_sig2021x1p25_lslb1p1_rpen4p6
```

For the primary run, inspect `summary_combined_all/` for the combined
observed/expected upper-limit suite, `combined_ul_bands_combined_all.csv` for
the combined expected-band table, and `combined_ul_bands_2021.csv` plus the
`ul_bands_eps2_2021.png` task/merged plots for the 2021-only band products.
