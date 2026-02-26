#!/bin/bash
#SBATCH --job-name=hps2016_10pct_bands_10k
#SBATCH --partition=milano
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --account=hps:hps-prod

mkdir -p logs

# TASK_ID and N_TASKS are passed via --export at submission time
hps-gpr scan \
    --config config_2016_10pct_10k.yaml \
    --array-task ${TASK_ID} \
    --n-tasks ${N_TASKS}
