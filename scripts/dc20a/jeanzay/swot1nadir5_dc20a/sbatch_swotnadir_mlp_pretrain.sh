#!/bin/bash
#SBATCH --job-name=osse_swot                # name of job
#SBATCH --account=yrf@v100                   # for statistics
#SBATCH --nodes=1                            # we ALWAYS request one node
#SBATCH --ntasks=1                           # number of tasks (analyses) to run
#SBATCH --cpus-per-task=10                   # number of cpus per task
#SBATCH -C v100-16g                          # V100 GPU + 16 GBs RAM
#SBATCH --gres=gpu:1                         # number of GPUs (1/4 of GPUs)
#SBATCH --qos=qos_gpu-t3                     # GPU partition (max 20 hrs)
#SBATCH --hint=nomultithread                 # hyperthreading is deactivated
#SBATCH --time=20:00:00                      # maximum execution time requested (HH:MM:SS)
#SBATCH --output=/gpfsscratch/rech/cli/uvo53rl/logs/nerf4ssh_dc20_swot1nadir5_mlp_%j_batch.log      # name of output file
#SBATCH --error=/gpfsscratch/rech/cli/uvo53rl/errs/nerf4ssh_dc20_swot1nadir5__mlp_%j_batch.err       # name of error file
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


########################
# TRAIN PRETRAIN
########################
# srun  --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
#     bash /gpfswork/rech/cli/uvo53rl/projects/jejeqx/scripts/dc20a/jeanzay/swot1nadir5_dc20a/swotnadir_ffn_pretrain.sh &
srun  --ntasks=1 python /gpfswork/rech/cli/uvo53rl/projects/jejeqx/scripts/dc20a/main.py \
    stage="train" \
    num_epochs=2000 \
    data=swot_dc20a \
    model=mlp \
    evaluation=natl60_dc20a \
    pretrained=mlp_nadir_dc20a \
    ++data.batch_size=50_000 \
    ++logger.mode="disabled" \
    ++optimizer.learning_rate=5e-5 \
    ++data.train_size=0.90 \
    dataset="swot1nadir5"
# wait
