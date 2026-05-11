# Observed Upper-Limit Scan: blind 2.25, LS lower bound 1.0

This run mode is the observed upper-limit production after the length-scale and
signal-leakage lock-ins.

Locked choices:

- 10k background toys for expected bands and toy-tail diagnostics.
- Observed-data CLs evaluation uses `ul_bands_cls_mode: asymptotic`.
- Radiative-fraction penalty is enabled for 2015, 2016 10%, and 2021 1%.
- Blind/extraction window is `m +/- 2.25 sigma(m)`.
- No additional GP-training guard band: `gp_train_exclude_nsigma` and
  `ul_bands_train_exclude_nsigma` are also `2.25`.
- GP length-scale lower bound is `1.0 sigma_x` for all datasets.
- The prompt-density/rate window for `A <-> epsilon^2` conversion is
  `eps2_density_nsigma: 1.64`.

YAMLs:

- `study_configs/config_2015_obsUL_blind2p25_lslb1p0_rpen7_dens1p64_10k.yaml`
- `study_configs/config_2016_10pct_obsUL_blind2p25_lslb1p0_rpen7_dens1p64_10k.yaml`
- `study_configs/config_2021_1pct_obsUL_blind2p25_lslb1p0_rpen7_dens1p64_10k.yaml`
- `study_configs/config_2015_2016_10pct_2021_1pct_obsUL_blind2p25_lslb1p0_rpen7_dens1p64_10k.yaml`

The combined YAML uses `run_limit_bands_on: all`, so its SLURM jobs write
dataset-level band CSVs for all active datasets as well as
`ul_bands_combined_all.csv`. `slurm-combine` then writes
`summary_combined_all/ul_observed_overlay_eps2.png`, overlaying the three observed
dataset upper limits and the combined observed upper limit on a log scale.

All four YAMLs write under `/sdf/data/hps/users/epeets/hps_gpr/observed_ul/` so the heavy task outputs land in the HPS data area rather than under `outputs/` in the source checkout.

## SDF Sync

Run this in your SDF checkout before generating jobs:

```bash
cd /path/to/hps-gpr
git status --short
git stash push -u -m "before observed UL blind2p25 lslb1p0 rpen7"
git fetch origin
git checkout codex/observed-ul-blind2p25-lslb1p0-rpen7
git pull --ff-only origin codex/observed-ul-blind2p25-lslb1p0-rpen7
git stash pop
```

If `git stash pop` reports conflicts, resolve those before submitting jobs.

## Generate And Submit

From the repo root on SDF:

```bash
# 2015-only observed UL scan: 20-130 MeV, 111 jobs
hps-gpr slurm-gen \
  --config study_configs/config_2015_obsUL_blind2p25_lslb1p0_rpen7_dens1p64_10k.yaml \
  --n-jobs 111 \
  --job-name hps2015_obsUL_b225_lslb1_rpen7 \
  --partition roma \
  --account hps:hps-prod \
  --time 24:00:00 \
  --memory 8G \
  --output submit_2015_obsUL_b225_lslb1_rpen7.slurm

./submit_all.sh submit_2015_obsUL_b225_lslb1_rpen7.slurm

# 2016 10%-only observed UL scan: 35-210 MeV, 176 jobs
hps-gpr slurm-gen \
  --config study_configs/config_2016_10pct_obsUL_blind2p25_lslb1p0_rpen7_dens1p64_10k.yaml \
  --n-jobs 176 \
  --job-name hps2016_10pct_obsUL_b225_lslb1_rpen7 \
  --partition roma \
  --account hps:hps-prod \
  --time 24:00:00 \
  --memory 8G \
  --output submit_2016_10pct_obsUL_b225_lslb1_rpen7.slurm

./submit_all.sh submit_2016_10pct_obsUL_b225_lslb1_rpen7.slurm

# 2021 1%-only observed UL scan: 30-250 MeV, 221 jobs
hps-gpr slurm-gen \
  --config study_configs/config_2021_1pct_obsUL_blind2p25_lslb1p0_rpen7_dens1p64_10k.yaml \
  --n-jobs 221 \
  --job-name hps2021_1pct_obsUL_b225_lslb1_rpen7 \
  --partition roma \
  --account hps:hps-prod \
  --time 24:00:00 \
  --memory 8G \
  --output submit_2021_1pct_obsUL_b225_lslb1_rpen7.slurm

./submit_all.sh submit_2021_1pct_obsUL_b225_lslb1_rpen7.slurm

# Three-way observed UL scan: union 20-250 MeV, 231 jobs
hps-gpr slurm-gen \
  --config study_configs/config_2015_2016_10pct_2021_1pct_obsUL_blind2p25_lslb1p0_rpen7_dens1p64_10k.yaml \
  --n-jobs 231 \
  --job-name hps2015_2016_2021_obsUL_b225_lslb1_rpen7 \
  --partition roma \
  --account hps:hps-prod \
  --time 24:00:00 \
  --memory 8G \
  --output submit_2015_2016_2021_obsUL_b225_lslb1_rpen7.slurm

./submit_all.sh submit_2015_2016_2021_obsUL_b225_lslb1_rpen7.slurm
```

## Compile

Run each compile after its jobs finish:

```bash
hps-gpr slurm-combine --output-dir /sdf/data/hps/users/epeets/hps_gpr/observed_ul/obsUL_2015_blind2p25_lslb1p0_rpen7_dens1p64_10k
hps-gpr slurm-combine --output-dir /sdf/data/hps/users/epeets/hps_gpr/observed_ul/obsUL_2016_10pct_blind2p25_lslb1p0_rpen7_dens1p64_10k
hps-gpr slurm-combine --output-dir /sdf/data/hps/users/epeets/hps_gpr/observed_ul/obsUL_2021_1pct_blind2p25_lslb1p0_rpen7_dens1p64_10k
hps-gpr slurm-combine --output-dir /sdf/data/hps/users/epeets/hps_gpr/observed_ul/obsUL_2015_2016_10pct_2021_1pct_blind2p25_lslb1p0_rpen7_dens1p64_10k
```

Key plots to inspect in each `summary_combined_*` folder:

- `ul_bands_eps2_obsexp.png`: expected bands plus observed limit, log scale.
- `p0_analytic_local_global.png`: asymptotic local and global-reference p-values,
  log scale.
- `ul_pvalues.png`: toy-derived `p_strong`, `p_weak`, and `p_two`, log scale.
- `ul_pvalues_components_local_global_refs.png`: toy tails with local/global
  reference lines.
- `ul_observed_overlay_eps2.png`: combined-run overlay of the three observed
  dataset limits plus the combined observed limit, log scale.
