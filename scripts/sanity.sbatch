#!/bin/bash
#SBATCH -A desi_g                     # account
#SBATCH -C gpu                        # request GPU node
#SBATCH -q regular                    # qos/queue
#SBATCH -t 04:00:00                   # walltime
#SBATCH -N 1                          # nodes
#SBATCH -c 16                         # CPU cores
#SBATCH --gpus 1                      # GPUs
#SBATCH --array=1-4                   # run bins 1–4 as separate tasks
#SBATCH -J y3-mcmc-bins               # job name
#SBATCH -o /global/u1/z/zhaozhon/project/slurm/bin%A_%a.out
#SBATCH -e /global/u1/z/zhaozhon/project/slurm/bin%A_%a.err

# ------------------------------
# Environment setup
# ------------------------------
module use /global/common/software/desi/perlmutter/desiconda/current/modulefiles
module load desiconda

export PYTHONPATH=$HOME/project/src:$PYTHONPATH

# ------------------------------
# Run the per-bin MCMC script
# ------------------------------
srun -n 1 python $HOME/project/scripts/mcmc_perbin.py \
     --bin-index $SLURM_ARRAY_TASK_ID \
     --num-warmup 10000 \
     --num-samples 10000 \
     --num-chains 4