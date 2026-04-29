#!/bin/bash
# Perlmutter (NERSC) — DESI Y3 per-bin MCMC for cases: all/upper/lower

#SBATCH -A desi_g
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1

# 4 bins × 3 cases = 12 tasks
#SBATCH --array=0-11%4

#SBATCH -J y3-mcmc-cases
#SBATCH --chdir=/global/u1/z/zhaozhon/project

#SBATCH -o /global/u1/z/zhaozhon/project/slurm/cases_%A_%a.out
#SBATCH -e /global/u1/z/zhaozhon/project/slurm/cases_%A_%a.err

set -euo pipefail

# ------------------------------
# Environment
# ------------------------------
module use /global/common/software/desi/perlmutter/desiconda/current/modulefiles
module load desiconda

# Activate your env if needed:
# conda activate <YOUR_ENV>

# Keep math libs from oversubscribing CPUs
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# JAX niceties on shared GPUs
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

# Ensure python can import your package
export PYTHONPATH=$HOME/project/src:$PYTHONPATH

mkdir -p "$HOME/project/slurm"

echo "[$(date)] Host: $(hostname)"
echo "JOB_ID=$SLURM_JOB_ID ARRAY_ID=$SLURM_ARRAY_JOB_ID TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "WorkDir: $PWD"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
which python
python -V
nvidia-smi || true

# ------------------------------
# Map array index -> (bin, case)
# ------------------------------
# task_id in [0..14]
# bin_index = (task_id / 3) + 1   -> 1..5
# case_id   = task_id % 3         -> 0,1,2
# cases = [all, upper, lower]
TASK=${SLURM_ARRAY_TASK_ID}
BIN_INDEX=$(( TASK / 3 + 1 ))
CASE_ID=$(( TASK % 3 ))

if [ "$CASE_ID" -eq 0 ]; then
  CASE="all"
elif [ "$CASE_ID" -eq 1 ]; then
  CASE="upper"
else
  CASE="lower"
fi

echo "[dispatch] bin=${BIN_INDEX} case=${CASE}"

# ------------------------------
# Run
# ------------------------------
srun --cpu-bind=cores --gpu-bind=single:1 \
  python "$HOME/project/scripts/mcmc_perbin_case.py" \
    --bin-index "${BIN_INDEX}" \
    --case "${CASE}" \
    --num-warmup 10000 \
    --num-samples 10000 \
    --num-chains 4 \
    --seed 12345 \