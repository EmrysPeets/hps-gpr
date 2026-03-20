#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

PARTITION="${PARTITION:-roma}"
ACCOUNT="${ACCOUNT:-hps:hps-prod}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
MEMORY="${MEMORY:-6G}"
CONFIG="${CONFIG:-study_configs/config_2015_extraction_display_v15p8.yaml}"
DATASET="${DATASET:-2015}"
BASE_OUTPUT="${BASE_OUTPUT:-outputs/extraction_display_2015_batch}"
LOGDIR="${LOGDIR:-logs/extraction_display_2015}"
SIGMAS_STR="${SIGMAS:-3 5 7}"

mkdir -p "${LOGDIR}"
read -r -a SIGMA_LIST <<< "${SIGMAS_STR}"

for Z in "${SIGMA_LIST[@]}"; do
  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=hps_exdisp_2015_z${Z}
#SBATCH --partition=${PARTITION}
#SBATCH --account=${ACCOUNT}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --mem=${MEMORY}
#SBATCH --output=${LOGDIR}/%x_%j.out
#SBATCH --error=${LOGDIR}/%x_%j.err

set -euo pipefail
cd "${REPO_DIR}"
PYTHONPATH=. hps-gpr extract-display \
  --config "${CONFIG}" \
  --dataset "${DATASET}" \
  --strengths "${Z}" \
  --output-dir "${BASE_OUTPUT}/z${Z}"
EOF
done
