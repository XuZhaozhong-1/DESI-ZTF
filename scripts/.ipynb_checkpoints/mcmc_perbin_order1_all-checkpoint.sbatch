#!/bin/bash
# Perlmutter (NERSC) — DESI Y3 per-bin MCMC
# all-case only, order=1 sigma(m), constant mu(z_center)

#SBATCH -A desi_g
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --array=1-4%4
#SBATCH -J y3-mcmc-o1-all
#SBATCH --chdir=/global/u1/z/zhaozhon/project

#SBATCH -o /global/u1/z/zhaozhon/project/slurm/order1_all_%A_%a.out
#SBATCH -e /global/u1/z/zhaozhon/project/slurm/order1_all_%A_%a.err

set -euo pipefail

# ------------------------------
# Environment
# ------------------------------
module use /global/common/software/desi/perlmutter/desiconda/current/modulefiles
module load desiconda

# If needed:
# source activate base
# conda activate desi-y3

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

export PYTHONPATH=$HOME/project/src:$PYTHONPATH

mkdir -p "$HOME/project/slurm"

echo "[$(date)] Host: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_ARRAY_JOB_ID=$SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "WorkDir: $PWD"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
which python
python -V
nvidia-smi || true

# ------------------------------
# Run per-bin order-1 MCMC
# ------------------------------
srun --cpu-bind=cores --gpu-bind=single:1 \
  python "$HOME/project/scripts/mcmc_perbin_order1_all.py" \
    --bin-index "${SLURM_ARRAY_TASK_ID}" \
    --num-warmup 10000 \
    --num-samples 10000 \
    --num-chains 4 \
    --target-accept 0.9 \
    --m-ref 22.0 \
    --sigma-nbins 20 \
    --sigma-min-per-bin 20
