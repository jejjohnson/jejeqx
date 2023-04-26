#!/bin/bash
#SBATCH --job-name=osse_n_sine                # name of job
#SBATCH --account=yrf@v100                   # for statistics
#SBATCH --nodes=1                            # we ALWAYS request one node
#SBATCH --ntasks-per-node=1                  # number of tasks per node
#SBATCH --cpus-per-task=10                   # number of cpus per task
#SBATCH -C v100-16g                          # V100 GPU + 16 GBs RAM
#SBATCH --gres=gpu:1                         # number of GPUs (1/4 of GPUs)
#SBATCH --qos=qos_gpu-t3                     # GPU partition (max 20 hrs)
#SBATCH --time=20:00:00                      # maximum execution time requested (HH:MM:SS)
#SBATCH --output=/gpfsscratch/rech/cli/uvo53rl/logs/nerf4ssh_dc20_nadir4_%j.log      # name of output file
#SBATCH --error=/gpfsscratch/rech/cli/uvo53rl/errs/nerf4ssh_dc20_nadir4_%j.err       # name of error file
#SBATCH --export=ALL
#SBATCH --signal=SIGUSR1@90


# loading of modules
module purge

module load git/2.31.1
module load github-cli/1.13.1
module load git-lfs/3.0.2

# go to appropriate directory
cd /gpfswork/rech/cli/uvo53rl/projects/jejeqx/scripts/dc20a/
export PYTHONPATH=/gpfswork/rech/cli/uvo53rl/projects/jejeqx:${PYTHONPATH}


# load modules
source activate jejeqx

# TRAIN NADIR Simulations
srun /gpfswork/rech/cli/uvo53rl/projects/jejeqx/scripts/dc20a/jeanzay/nadir4_dc20a/train_nadir4dc20a_siren.sh