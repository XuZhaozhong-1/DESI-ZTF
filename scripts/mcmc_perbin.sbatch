#!/bin/bash
# Perlmutter (NERSC) — DESI Y3 per-bin MCMC

#SBATCH -A desi_g                      # NERSC project/account
#SBATCH -q regular                     # QOS: debug | regular | premium
#SBATCH -C gpu                         # request GPU nodes (A100)
#SBATCH -t 04:00:00                    # wall clock time
#SBATCH -N 1                           # nodes
#SBATCH --ntasks=1                     # one MPI task per array item
#SBATCH --cpus-per-task=16             # CPU cores for that task
#SBATCH --gpus-per-task=1              # 1 GPU per task
#SBATCH --array=1-4%4                  # run bins 1–4; cap concurrency to 4
#SBATCH -J y3-mcmc-bins                # job name
#SBATCH --chdir=/global/u1/z/zhaozhon/project   # working dir

# Per-task logs (array-safe)
#SBATCH -o /global/u1/z/zhaozhon/project/slurm/bin%A_%a.out
#SBATCH -e /global/u1/z/zhaozhon/project/slurm/bin%A_%a.err

set -euo pipefail

# ------------------------------
# Environment
# ------------------------------
module use /global/common/software/desi/perlmutter/desiconda/current/modulefiles
module load desiconda

# If you use a specific conda env, uncomment and set it:
# source activate base
# conda activate desi-y3   # <--- change to your env name

# Keep math libs from oversubscribing CPUs
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# (Optional) JAX/TF niceties on shared GPUs
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

# Your code path
export PYTHONPATH=$HOME/project/src:$PYTHONPATH

echo "[$(date)] Host: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  SLURM_ARRAY_JOB_ID=$SLURM_ARRAY_JOB_ID  TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "WorkDir: $PWD"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-unset}"
which python ; python -V
nvidia-smi || true

# ------------------------------
# Run the per-bin MCMC
# ------------------------------
# Bind CPUs to cores and 1 GPU per task for clean affinity
srun --cpu-bind=cores --gpu-bind=single:1 \
  python "$HOME/project/scripts/mcmc_perbin.py" \
    --bin-index "${SLURM_ARRAY_TASK_ID}" \
    --num-warmup 10000 \
    --num-samples 10000 \
    --num-chains 4