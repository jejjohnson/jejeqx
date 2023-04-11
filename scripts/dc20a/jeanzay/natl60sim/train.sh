#!/bin/bash

#python /gpfswork/rech/cli/uvo53rl/projects/jejeqx/scripts/dc20a/main.py \
#    stage="train" \
#    num_epochs=1000 \
#    data=natl60 \
#    evaluation=natl60_sim \
#    ++logger.mode="offline" \
#    ++optimizer.learning_rate=5e-5

#python /gpfswork/rech/cli/uvo53rl/projects/jejeqx/scripts/dc20a/main.py \
#    stage="train" \
#    num_epochs=1 \
#    data=natl60 \
#    evaluation=natl60_sim \
#    ++logger.mode="disabled" \
#    ++optimizer.learning_rate=5e-5 \
#    ++data.subset_size=0.25
    
# ===============================================
# RANDOM FOURIER FEATURES
# ===============================================
python /gpfswork/rech/cli/uvo53rl/projects/jejeqx/scripts/dc20a/main.py \
    stage="train" \
    num_epochs=1000 \
    data=natl60 \
    model=ffn \
    evaluation=natl60_sim \
    pretrained=default \
    ++model.basis_net.depth=5 \
    ++logger.mode="offline" \
    ++optimizer.learning_rate=1e-4 \
    dataset="natl60" 