# 2015 Toy-Study Workflows

This runbook reproduces the corrected 2015 toy-study workflow in four stages:

1. regenerate the functional-form toy ROOT files
2. run the 2015 functional-form closure ladder
3. run the 2015 GP-mean/global-fit pseudoexperiments
4. run the full 2015 scan/bands production

The baseline configuration keeps the real-data-style full-range GP training.
The audit configuration clips 2015 GP training to `20-130 MeV` so you can
measure the sub-20 MeV impact directly.

Runtime notes:

- `toy-scan` runs one full mass scan per toy histogram, so its wall time is not directly comparable to a single observed-data scan chunk or one array-task slice.
- `inject` with `inj_refit_gp_on_toy: true` does one full-range toy generation plus one GP refit per `(mass, strength, toy)` point.
- On SDF, prefer outer job parallelism over nested joblib workers. The shipped production configs now default the inner scan/toy workers to serial settings for stability.

## 1. Regenerate The Toy ROOT Files

Run the dataset macros from the repo root:

```bash
root -l -b -q 'root_macros/funcform/make_func_data_output_2015.C()'
root -l -b -q 'root_macros/funcform/make_func_data_output_2016.C()'
root -l -b -q 'root_macros/funcform/make_func_data_output_2021.C()'
```

Expected outputs:

- `outputs/funcform_toys/funcform_2015_dataset_mod_toys.root`
- `outputs/funcform_toys/funcform_2016_dataset_mod_toys.root`
- `outputs/funcform_toys/funcform_2021_dataset_mod_toys.root`
- `outputs/funcform_toys/funcform_2015_dataset_mod_toys.root.metadata.json`
- `outputs/funcform_toys/funcform_2016_dataset_mod_toys.root.metadata.json`
- `outputs/funcform_toys/funcform_2021_dataset_mod_toys.root.metadata.json`
- `hps_gpr_analysis_note/toy_generation_figs/funcform_fit_2015.{png,pdf}`
- `hps_gpr_analysis_note/toy_generation_figs/funcform_fit_2016.{png,pdf}`
- `hps_gpr_analysis_note/toy_generation_figs/funcform_fit_2021.{png,pdf}`
- `hps_gpr_analysis_note/toy_generation_figs/funcform_primary_fit_summary.{png,pdf}`

Each ROOT file now contains:

- `input_hist`
- one directory per function family with the generated toys
- `fit_functions/`
- `fit_metadata/`
- `validation/`

Use the sidecar `*.metadata.json` file to confirm:

- the full support range used for toy normalization
- the fit range used for parameter determination
- the intended scan range
- the normalization target event count
- the selected primary family and validation summary

Generation rule:

- each toy bin is sampled from a Poisson distribution whose mean comes from the
  support-range-normalized bin integral of the selected analytic fit

Refresh the note-local comparison figure after regenerating the ROOT files:

```bash
python3 hps_gpr_analysis_note/scripts/generate_note_figures.py --funcform-only
```

## 2. Run The 2015 Functional-Form Closure Ladder

### Baseline Smoke Test

```bash
python -m hps_gpr.cli toy-scan \
  --config config_2015_10k.yaml \
  --dataset 2015 \
  --toy-root outputs/funcform_toys/funcform_2015_dataset_mod_toys.root \
  --container fShiftSigPowTail \
  --toy-pattern 'fShiftSigPowTail_toy_*' \
  --max-toys 1 \
  --output-dir outputs/funcform_toy_smoke_2015_corrected \
  --mass-min 0.020 \
  --mass-max 0.040

python -m hps_gpr.cli toy-scan-merge \
  --input-dir outputs/funcform_toy_smoke_2015_corrected/toy_scans/2015 \
  --output-dir outputs/funcform_toy_smoke_2015_corrected/merged
```

Expected outputs:

- `outputs/funcform_toy_smoke_2015_corrected/toy_scans/2015/toy_0000/`
- `outputs/funcform_toy_smoke_2015_corrected/merged/toy_scan_merged.csv`
- `outputs/funcform_toy_smoke_2015_corrected/merged/toy_scan_summary.csv`

### Baseline 10-Toy Pilot

```bash
python -m hps_gpr.cli toy-scan \
  --config config_2015_10k.yaml \
  --dataset 2015 \
  --toy-root outputs/funcform_toys/funcform_2015_dataset_mod_toys.root \
  --container fShiftSigPowTail \
  --toy-pattern 'fShiftSigPowTail_toy_*' \
  --max-toys 10 \
  --output-dir outputs/funcform_toy_pilot_2015_corrected

python -m hps_gpr.cli toy-scan-merge \
  --input-dir outputs/funcform_toy_pilot_2015_corrected/toy_scans/2015 \
  --output-dir outputs/funcform_toy_pilot_2015_corrected/merged
```

Inspect:

- `outputs/funcform_toy_pilot_2015_corrected/merged/toy_scan_summary.csv`
- `outputs/funcform_toy_pilot_2015_corrected/merged/toy_scan_validation_2015_fShiftSigPowTail_local_significance.png`

### Baseline Full 100-Toy Study

```bash
python -m hps_gpr.cli slurm-gen-toy-scan \
  --config config_2015_10k.yaml \
  --dataset 2015 \
  --toy-root outputs/funcform_toys/funcform_2015_dataset_mod_toys.root \
  --container fShiftSigPowTail \
  --toy-pattern 'fShiftSigPowTail_toy_*' \
  --job-name hps2015_toyscan_corrected \
  --partition roma \
  --cpus-per-task 10 \
  --account hps:hps-prod \
  --time 4:00:00 \
  --memory 8G \
  --output submit_2015_toyscan_corrected.slurm

bash submit_toy_scan_all.sh --account hps:hps-prod --qos <valid-qos>

python -m hps_gpr.cli toy-scan-merge \
  --input-dir outputs/prod_2015_10k_3/jobs \
  --output-dir outputs/prod_2015_10k_3/merged
```

Expected outputs:

- `outputs/prod_2015_10k_3/jobs/<toy_name>/toy_scans/2015/toy_XXXX/`
- `outputs/prod_2015_10k_3/merged/toy_scan_merged.csv`
- `outputs/prod_2015_10k_3/merged/toy_scan_summary.csv`

### Audit Variant With 20-130 MeV GP Training

Use the dedicated audit config:

- `study_configs/config_2015_toyscan_audit_20to130.yaml`

Example smoke run:

```bash
python -m hps_gpr.cli toy-scan \
  --config study_configs/config_2015_toyscan_audit_20to130.yaml \
  --dataset 2015 \
  --toy-root outputs/funcform_toys/funcform_2015_dataset_mod_toys.root \
  --container fShiftSigPowTail \
  --toy-pattern 'fShiftSigPowTail_toy_*' \
  --max-toys 10 \
  --output-dir outputs/funcform_toy_audit_2015_20to130

python -m hps_gpr.cli toy-scan-merge \
  --input-dir outputs/funcform_toy_audit_2015_20to130/toy_scans/2015 \
  --output-dir outputs/funcform_toy_audit_2015_20to130/merged
```

Compare the baseline and audit merges to isolate the contribution from
sub-20 MeV GP training.

## 3. Run The 2015 GP-Mean / Global-Fit Pseudoexperiments

```bash
hps-gpr inject \
  --config study_configs/config_2015_blind1p64_95CL_10k_injection_gpmean_pseudoexp.yaml \
  --dataset 2015 \
  --masses 0.025,0.030,0.040,0.050,0.065,0.080,0.095,0.115,0.135 \
  --strengths 1,2,3,5 \
  --n-toys 10000

hps-gpr inject-plot \
  --input-dir outputs/study_2015_w1p64_95CL_gpmean_pseudoexp/injection_flat \
  --output-dir outputs/study_2015_w1p64_95CL_gpmean_pseudoexp/injection_summary
```

Expected outputs:

- `outputs/study_2015_w1p64_95CL_gpmean_pseudoexp/injection_flat/`
- `outputs/study_2015_w1p64_95CL_gpmean_pseudoexp/injection_summary/`

Check these summary products:

- pull mean and width
- coverage
- Z-calibration residuals
- fixed-GP vs refit-GP summaries

These GP-mean pseudoexperiments are intentionally heavier than a plain observed-data scan because each toy point regenerates full-range pseudo-data and refits the GP on toy sidebands.

## 4. Run The Full 2015 Scan / Bands Production

```bash
hps-gpr slurm-gen \
  --config config_2015_10k.yaml \
  --n-jobs 111 \
  --job-name hps2015_bands_10k \
  --partition roma \
  --account hps:hps-prod \
  --time 24:00:00 \
  --memory 8G \
  --output submit_2015_bands_10k.slurm

./submit_all.sh submit_2015_bands_10k.slurm

hps-gpr slurm-combine --output-dir outputs/prod_2015_10k_3
```

For these array-split scan productions, keep the inner scan parallelism off and let the job matrix provide the parallelism.

Expected outputs:

- `outputs/prod_2015_10k_3/task_XXXX/`
- `outputs/prod_2015_10k_3/results_single.csv`
- `outputs/prod_2015_10k_3/results_combined.csv`
- `outputs/prod_2015_10k_3/ul_bands_2015.csv`
- `outputs/prod_2015_10k_3/summary_plots/`

## 5. Compile The Note Locally

After the toy ROOT files and note-local figure exports are refreshed:

```bash
cd hps_gpr_analysis_note
tectonic main.tex
```

Expected output:

- `hps_gpr_analysis_note/main.pdf`

## Validation Checklist

- Confirm the regenerated `*.metadata.json` files report the expected full support range, scan range, and normalization target count.
- Confirm the toy ROOT validation summaries preserve the real-data full-range event total and report scan-range, sideband-fraction, and peak comparisons.
- Confirm the 100-toy baseline merge still shows the dominant closure feature at `24-25 MeV`, with the `32-34 MeV` shoulder remaining secondary.
- Compare the baseline and audit merges before changing any GP kernel settings.
