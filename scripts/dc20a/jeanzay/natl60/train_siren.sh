#!/bin/bash

# ===============================================
# SIREN
# ===============================================
# ++logger.mode="offline"
# num_epochs=1000
python /gpfswork/rech/cli/uvo53rl/projects/jejeqx/scripts/dc20a/main.py \
    stage="train" \
    num_epochs=1000 \
    data=natl60 \
    model=siren \
    evaluation=natl60 \
    pretrained=default \
    ++data.batch_size=32 \
    ++logger.mode="disabled" \
    ++optimizer.learning_rate=1e-4 \
    ++data.train_size=0.90 \
    ++model.basis_net.ard=False \
    dataset="natl60" 