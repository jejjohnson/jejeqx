#!/bin/bash

python /gpfswork/rech/cli/uvo53rl/projects/jejeqx/scripts/dc20a/main.py \
    stage="inference" \
    num_epochs=1000 \
    data=natl60 \
    evaluation=natl60_sim \
    ++logger.mode="disabled" \
    ++optimizer.learning_rate=1e-4 \
    pretrained=siren_natl60_full

#python /gpfswork/rech/cli/uvo53rl/projects/jejeqx/scripts/dc20a/main.py \
#    stage=train \
#    num_epochs=1 \
#    ++logger.mode="disabled" \
#    data=natl60 \
#    evaluation=natl60_sim \
#    ++data.subset_size=0.25