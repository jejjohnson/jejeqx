#!/bin/bash
#SBATCH --job-name=ffn_s                 # name of job
#SBATCH --account=yrf@v100                   # for statistics
#SBATCH --nodes=1                            # we ALWAYS request one node
#SBATCH --ntasks-per-node=1                  # number of tasks per node
#SBATCH --cpus-per-task=10                   # number of cpus per task
#SBATCH -C v100-16g                          # V100 GPU + 16 GBs RAM
#SBATCH --gres=gpu:1                         # number of GPUs (1/4 of GPUs)
#SBATCH --hint=nomultithread                 # hyperthreading is deactivated
#SBATCH --qos=qos_gpu-t3                     # GPU partition (max 20 hrs)
#SBATCH --time=20:00:00                      # maximum execution time requested (HH:MM:SS)
#SBATCH --output=/gpfsscratch/rech/cli/uvo53rl/logs/nerf4ssh_dc20_swot1nadir5_ffn_%j.log      # name of output file
#SBATCH --error=/gpfsscratch/rech/cli/uvo53rl/errs/nerf4ssh_dc20_swot1nadir5_ffn_%j.err       # name of error file
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

#######################
# TRAIN
#######################
# srun /gpfswork/rech/cli/uvo53rl/projects/jejeqx/scripts/dc20a/jeanzay/swot1nadir5_dc20a/train_swotnadir_ffn.sh
srun python /gpfswork/rech/cli/uvo53rl/projects/jejeqx/scripts/dc20a/main.py \
    stage="train" \
    num_epochs=2000 \
    data=swot_dc20a \
    model=ffn \
    evaluation=natl60_dc20a \
    pretrained=default \
    ++data.batch_size=100_000 \
    ++logger.mode="offline" \
    ++optimizer.learning_rate=1e-4 \
    ++data.train_size=0.90 \
    dataset="swot1nadir5" \
    ++model.basis_net.ard=False
