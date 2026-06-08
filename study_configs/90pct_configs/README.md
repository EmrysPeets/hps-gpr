# 90% CL Observed Upper-Limit Rerun Configs

These configs reproduce the nominal observed upper-limit settings from the latest
analysis-note workflow, changing only the CLs threshold from 95% CL to 90% CL:

- `cls_alpha: 0.1`
- blind/extraction window: `blind_nsigma: 2.25`
- GP training exclusion: `gp_train_exclude_nsigma: 2.25`
- UL-band toy training exclusion: `ul_bands_train_exclude_nsigma: 2.25`
- GP length-scale lower bound: `1.0 sigma_x`
- radiative penalty enabled with `radiative_penalty_frac_*: 0.07`
- prompt-density window: `eps2_density_nsigma: 1.64`
- 10k expected-band toys

The older 90% CL YAMLs in `study_configs/` are legacy injection-study configs
with different nominal settings. Use the YAMLs in this folder for the current
observed-UL rerun on SDF.

## YAMLs

- `study_configs/90pct_configs/config_2015_obsUL90_blind2p25_lslb1p0_rpen7_dens1p64_10k.yaml`
- `study_configs/90pct_configs/config_2016_10pct_obsUL90_blind2p25_lslb1p0_rpen7_dens1p64_10k.yaml`
- `study_configs/90pct_configs/config_2021_1pct_obsUL90_blind2p25_lslb1p0_rpen7_dens1p64_10k.yaml`
- `study_configs/90pct_configs/config_2015_2016_10pct_2021_1pct_obsUL90_blind2p25_lslb1p0_rpen7_dens1p64_10k.yaml`

Outputs go under:

```bash
/sdf/data/hps/users/epeets/hps_gpr/observed_ul_90CL/
```

## Push And Merge From Local

From the local checkout where these files were created:

```bash
git checkout -b codex/90pct-observed-ul-configs
git add study_configs/90pct_configs
git commit -m "Add 90 percent observed UL configs"
git push -u origin codex/90pct-observed-ul-configs
gh pr create --base main --head codex/90pct-observed-ul-configs --fill
gh pr merge --squash --delete-branch
```

If the target branch is not `main`, replace `main` in the `gh pr create`
command.

## Update The SDF Checkout

After the PR is merged, update the SDF checkout while preserving local SDF files:

```bash
cd /path/to/hps-gpr
git status --short
git stash push -u -m "before 90pct observed UL rerun"
git fetch origin
git checkout main
git pull --ff-only origin main
git stash pop
```

If `git stash pop` reports conflicts, resolve them before generating jobs.

## Generate And Submit On SDF

From the repo root on SDF:

```bash
# 2015-only observed UL scan: 20-130 MeV, 111 jobs
hps-gpr slurm-gen \
  --config study_configs/90pct_configs/config_2015_obsUL90_blind2p25_lslb1p0_rpen7_dens1p64_10k.yaml \
  --n-jobs 111 \
  --job-name hps2015_obsUL90_b225_lslb1_rpen7 \
  --partition roma \
  --account hps:hps-prod \
  --time 24:00:00 \
  --memory 8G \
  --output submit_2015_obsUL90_b225_lslb1_rpen7.slurm

./submit_all.sh submit_2015_obsUL90_b225_lslb1_rpen7.slurm

# 2016 10%-only observed UL scan: 35-210 MeV, 176 jobs
hps-gpr slurm-gen \
  --config study_configs/90pct_configs/config_2016_10pct_obsUL90_blind2p25_lslb1p0_rpen7_dens1p64_10k.yaml \
  --n-jobs 176 \
  --job-name hps2016_10pct_obsUL90_b225_lslb1_rpen7 \
  --partition roma \
  --account hps:hps-prod \
  --time 24:00:00 \
  --memory 8G \
  --output submit_2016_10pct_obsUL90_b225_lslb1_rpen7.slurm

./submit_all.sh submit_2016_10pct_obsUL90_b225_lslb1_rpen7.slurm

# 2021 1%-only observed UL scan: 30-250 MeV, 221 jobs
hps-gpr slurm-gen \
  --config study_configs/90pct_configs/config_2021_1pct_obsUL90_blind2p25_lslb1p0_rpen7_dens1p64_10k.yaml \
  --n-jobs 221 \
  --job-name hps2021_1pct_obsUL90_b225_lslb1_rpen7 \
  --partition roma \
  --account hps:hps-prod \
  --time 24:00:00 \
  --memory 8G \
  --output submit_2021_1pct_obsUL90_b225_lslb1_rpen7.slurm

./submit_all.sh submit_2021_1pct_obsUL90_b225_lslb1_rpen7.slurm

# Three-way observed UL scan: union 20-250 MeV, 231 jobs
hps-gpr slurm-gen \
  --config study_configs/90pct_configs/config_2015_2016_10pct_2021_1pct_obsUL90_blind2p25_lslb1p0_rpen7_dens1p64_10k.yaml \
  --n-jobs 231 \
  --job-name hps2015_2016_2021_obsUL90_b225_lslb1_rpen7 \
  --partition roma \
  --account hps:hps-prod \
  --time 24:00:00 \
  --memory 8G \
  --output submit_2015_2016_2021_obsUL90_b225_lslb1_rpen7.slurm

./submit_all.sh submit_2015_2016_2021_obsUL90_b225_lslb1_rpen7.slurm
```

## Combine Finished Jobs

Run each combine after that scan's jobs finish:

```bash
hps-gpr slurm-combine --output-dir /sdf/data/hps/users/epeets/hps_gpr/observed_ul_90CL/obsUL90_2015_blind2p25_lslb1p0_rpen7_dens1p64_10k
hps-gpr slurm-combine --output-dir /sdf/data/hps/users/epeets/hps_gpr/observed_ul_90CL/obsUL90_2016_10pct_blind2p25_lslb1p0_rpen7_dens1p64_10k
hps-gpr slurm-combine --output-dir /sdf/data/hps/users/epeets/hps_gpr/observed_ul_90CL/obsUL90_2021_1pct_blind2p25_lslb1p0_rpen7_dens1p64_10k
hps-gpr slurm-combine --output-dir /sdf/data/hps/users/epeets/hps_gpr/observed_ul_90CL/obsUL90_2015_2016_10pct_2021_1pct_blind2p25_lslb1p0_rpen7_dens1p64_10k
```

For the combined scan, inspect `summary_combined_all/` for the combined and
per-dataset observed-limit overlays.
