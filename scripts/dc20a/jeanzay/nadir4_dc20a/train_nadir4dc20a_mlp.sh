#!/bin/bash

# ===============================================
# MLP
# ===============================================

# CONFIG I
# NADIR4 | NO PRETRAINING | ReLU | EVAL AREA
# ++logger.mode="offline"
# num_epochs=1000
python /gpfswork/rech/cli/uvo53rl/projects/jejeqx/scripts/dc20a/main.py \
    stage="train" \
    num_epochs=200 \
    data=nadir_dc20a \
    model=mlp \
    evaluation=natl60_dc20a \
    pretrained=default \
    ++data.batch_size=32 \
    ++logger.mode="offline" \
    ++optimizer.learning_rate=1e-4 \
    ++data.train_size=0.90 \
    dataset="nadir4" \
    lr_scheduler=warmup_cosine
