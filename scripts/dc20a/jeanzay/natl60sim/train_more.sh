#!/bin/bash

python /gpfswork/rech/cli/uvo53rl/projects/jejeqx/scripts/dc20a/main.py \
    stage="train" \
    num_epochs=100 \
    data=natl60_sim \
    evaluation=natl60_sim \
    ++logger.mode="offline" \
    ++data.subset_size=0.50 \
    ++optimizer.learning_rate=5e-5 \
    pretrained=siren_natl60_full \
    dataset="natl60"

#python /gpfswork/rech/cli/uvo53rl/projects/jejeqx/scripts/dc20a/main.py \
#    stage="train" \
#    num_epochs=1 \
#    ++logger.mode="disabled" \
#    data=natl60 \
#    evaluation=natl60_sim \
#    ++data.subset_size=0.25 \
#    pretrained=siren_natl60_full \
#    ++optimizer.learning_rate=5e-5