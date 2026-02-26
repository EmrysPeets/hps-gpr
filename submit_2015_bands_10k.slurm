#!/bin/bash
#SBATCH --job-name=hps2015_bands_10k
#SBATCH --account hps:hps-prod
#SBATCH --partition milano
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --output=/sdf/scratch/hps/epeets/output/logs/%j.out
#SBATCH --error=/sdf/scratch/hps/epeets/output/logs/%j.err

mkdir -p logs

# TASK_ID and N_TASKS are passed via --export at submission time
hps-gpr scan \
    --config config_2015_10k.yaml \
    --array-task ${TASK_ID} \
    --n-tasks ${N_TASKS}
