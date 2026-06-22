# 2021 Resolution-Scale 90% CL Variant Configs

These configs repeat the corrected 90% CL observed-limit setup while scaling
the 2021 mass-resolution parameterization by 25%:

```yaml
sigma_coeffs_2021:
- 0.00184825
- -0.001375
- 0.085875
```

This is exactly `1.25 * [0.0014786, -0.0011, 0.0687]`, so
`sigma_2021(m)` is scaled by 1.25 at every mass. The baseline 2015 and 2016
settings are unchanged.

Two 2021 systematic variants are included:

- `sig2021x1p25`: 2021 resolution scaled by 1.25, with the nominal 7% radiative-fraction penalty.
- `sig2021x1p25_rpen2021_4p6pct`: 2021 resolution scaled by 1.25, with `radiative_penalty_frac_2021: 0.046`; 2015 and 2016 remain at 7%.

Each variant has a combined-only config and a 2021-only config. The combined-only
configs set `run_limit_bands_on: ""`, so they produce the combined observed
limit and `ul_bands_combined_all.csv` while skipping individual expected-band
CSVs. The 2021-only configs keep `run_limit_bands_on: '2021'` and
`make_eps2_bands: true`, so they produce the 2021 observed/expected upper-limit
CSVs and plots for direct comparison.

## Configs

- `config_2015_2016_10pct_2021_1pct_obsUL90_combined_only_sig2021x1p25_blind2p25_lslb1p0_rpen7_dens1p64_10k.yaml`
- `config_2021_1pct_obsUL90_sig2021x1p25_blind2p25_lslb1p0_rpen7_dens1p64_10k.yaml`
- `config_2015_2016_10pct_2021_1pct_obsUL90_combined_only_sig2021x1p25_rpen2021_4p6pct_blind2p25_lslb1p0_dens1p64_10k.yaml`
- `config_2021_1pct_obsUL90_sig2021x1p25_rpen2021_4p6pct_blind2p25_lslb1p0_dens1p64_10k.yaml`

## Update The SDF Checkout

```bash
cd /path/to/hps-gpr

# If you have local SDF edits:
git stash push -u -m "before 2021 resolution-scale 90CL study"

git fetch origin
git checkout codex/observed-ul-blind2p25-lslb1p0-rpen7
git pull --ff-only origin codex/observed-ul-blind2p25-lslb1p0-rpen7

# Only run this if you actually stashed changes above:
git stash pop
```

## Resolution Scale Only

```bash
hps-gpr slurm-gen \
  --config study_configs/90pct_configs/2021_resolution_variants/config_2015_2016_10pct_2021_1pct_obsUL90_combined_only_sig2021x1p25_blind2p25_lslb1p0_rpen7_dens1p64_10k.yaml \
  --n-jobs 231 \
  --job-name hps_obsUL90_comb_sig2021x1p25 \
  --partition roma \
  --account hps:hps-prod \
  --time 24:00:00 \
  --memory 8G \
  --output submit_obsUL90_comb_sig2021x1p25.slurm

./submit_all.sh submit_obsUL90_comb_sig2021x1p25.slurm
```

Optional 2021-only comparison products:

```bash
hps-gpr slurm-gen \
  --config study_configs/90pct_configs/2021_resolution_variants/config_2021_1pct_obsUL90_sig2021x1p25_blind2p25_lslb1p0_rpen7_dens1p64_10k.yaml \
  --n-jobs 221 \
  --job-name hps2021_obsUL90_sig2021x1p25 \
  --partition roma \
  --account hps:hps-prod \
  --time 24:00:00 \
  --memory 8G \
  --output submit_2021_obsUL90_sig2021x1p25.slurm

./submit_all.sh submit_2021_obsUL90_sig2021x1p25.slurm
```

After jobs finish:

```bash
hps-gpr slurm-combine --output-dir /sdf/data/hps/users/epeets/hps_gpr/observed_ul_90CL/obsUL90_combined_only_2015_2016_10pct_2021_1pct_sig2021x1p25_blind2p25_lslb1p0_rpen7_dens1p64_10k_corrected
hps-gpr slurm-combine --output-dir /sdf/data/hps/users/epeets/hps_gpr/observed_ul_90CL/obsUL90_2021_1pct_sig2021x1p25_blind2p25_lslb1p0_rpen7_dens1p64_10k
```

## Resolution Scale Plus 2021 Radiative Penalty 4.6%

```bash
hps-gpr slurm-gen \
  --config study_configs/90pct_configs/2021_resolution_variants/config_2015_2016_10pct_2021_1pct_obsUL90_combined_only_sig2021x1p25_rpen2021_4p6pct_blind2p25_lslb1p0_dens1p64_10k.yaml \
  --n-jobs 231 \
  --job-name hps_obsUL90_comb_sig2021x1p25_rpen4p6 \
  --partition roma \
  --account hps:hps-prod \
  --time 24:00:00 \
  --memory 8G \
  --output submit_obsUL90_comb_sig2021x1p25_rpen4p6.slurm

./submit_all.sh submit_obsUL90_comb_sig2021x1p25_rpen4p6.slurm
```

Optional 2021-only comparison products:

```bash
hps-gpr slurm-gen \
  --config study_configs/90pct_configs/2021_resolution_variants/config_2021_1pct_obsUL90_sig2021x1p25_rpen2021_4p6pct_blind2p25_lslb1p0_dens1p64_10k.yaml \
  --n-jobs 221 \
  --job-name hps2021_obsUL90_sig2021x1p25_rpen4p6 \
  --partition roma \
  --account hps:hps-prod \
  --time 24:00:00 \
  --memory 8G \
  --output submit_2021_obsUL90_sig2021x1p25_rpen4p6.slurm

./submit_all.sh submit_2021_obsUL90_sig2021x1p25_rpen4p6.slurm
```

After jobs finish:

```bash
hps-gpr slurm-combine --output-dir /sdf/data/hps/users/epeets/hps_gpr/observed_ul_90CL/obsUL90_combined_only_2015_2016_10pct_2021_1pct_sig2021x1p25_rpen2021_4p6pct_blind2p25_lslb1p0_dens1p64_10k_corrected
hps-gpr slurm-combine --output-dir /sdf/data/hps/users/epeets/hps_gpr/observed_ul_90CL/obsUL90_2021_1pct_sig2021x1p25_rpen2021_4p6pct_blind2p25_lslb1p0_dens1p64_10k
```

For the 2021-only runs, compare `ul_bands_eps2_2021.png`,
`ul_bands_2021.csv`, and the merged `summary_combined_2021/` products against
the nominal 2021 output directory.
