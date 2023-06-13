#!/bin/bash


# ===============================================
# SIREN
# ===============================================
# CONFIG I
# NADIR4 | NO PRETRAINING | EVAL AREA
# ++logger.mode="offline"
# num_epochs=1000
python /gpfswork/rech/cli/uvo53rl/projects/jejeqx/scripts/dc20a/main.py \
    stage="train" \
    num_epochs=200 \
    data=nadir_dc20a \
    model=siren \
    evaluation=natl60_dc20a \
    pretrained=default \
    ++data.batch_size=32 \
    ++logger.mode="offline" \
    ++optimizer.learning_rate=5e-5 \
    ++data.train_size=0.90 \
    dataset="nadir4"


# ===============================================
# RANDOM FOURIER FEATURES
# ===============================================

# CONFIG I
# NADIR4 | NO PRETRAINING | RBF | EVAL AREA
# ++logger.mode="offline"
# num_epochs=1000
python /gpfswork/rech/cli/uvo53rl/projects/jejeqx/scripts/dc20a/main.py \
    stage="train" \
    num_epochs=200 \
    data=nadir_dc20a \
    model=ffn \
    evaluation=natl60_dc20a \
    pretrained=default \
    ++data.batch_size=32 \
    ++logger.mode="offline" \
    ++optimizer.learning_rate=5e-5 \
    ++data.train_size=0.90 \
    ++model.basis_net.ard=False \
    dataset="nadir4"

# # CONFIG II
# # NADIR4 | PRETRAINING - NATL60 | RBF | EVAL AREA
# # ++logger.mode="offline"
# # num_epochs=1000
# python /gpfswork/rech/cli/uvo53rl/projects/jejeqx/scripts/dc20a/main.py \
#     stage="train" \
#     num_epochs=50 \
#     data=nadir_dc20a \
#     model=ffn \
#     evaluation=natl60_dc20a \
#     pretrained=default \
#     ++data.batch_size=32 \
#     ++logger.mode="disabled" \
#     ++optimizer.learning_rate=1e-4 \
#     ++data.train_size=0.90 \
#     ++model.basis_net.ard=False \
#     dataset="nadir4"


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
    ++optimizer.learning_rate=5e-5 \
    ++data.train_size=0.90 \
    dataset="nadir4"
