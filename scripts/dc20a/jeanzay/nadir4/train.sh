#!/bin/bash


# ===============================================
# SIREN
# ===============================================
#python /gpfswork/rech/cli/uvo53rl/projects/jejeqx/scripts/dc20a/main.py \
#    stage="train" \
#    num_epochs=1000 \
#    data=nadir \
#    evaluation=natl60_dc20a \
#    pretrained=default \
#    ++data.batch_size=32 \
#    ++logger.mode="offline" \
#    ++optimizer.learning_rate=1e-4 \
#    ++data.train_size=0.90 \
#    dataset="nadir4"



# ===============================================
# RANDOM FOURIER FEATURES
# ===============================================

# CONFIG I
# NADIR4 | NO PRETRAINING | RBF | EVAL AREA
# ++logger.mode="offline"
# num_epochs=1000
python /gpfswork/rech/cli/uvo53rl/projects/jejeqx/scripts/dc20a/main.py \
    stage="train" \
    num_epochs=100 \
    data=nadir_dc20a \
    model=ffn \
    evaluation=natl60_dc20a \
    pretrained=default \
    ++data.batch_size=2048 \
    ++logger.mode="disabled" \
    ++optimizer.learning_rate=1e-4 \
    ++data.train_size=0.90 \
    ++model.basis_net.ard=False \
    dataset="nadir4"

# # CONFIG II
# # NDAIR4 | NO PRETRAINING | ARD | EVAL AREA
# python /gpfswork/rech/cli/uvo53rl/projects/jejeqx/scripts/dc20a/main.py \
#     stage="train" \
#     num_epochs=1000 \
#     data=nadir \
#     model=ffn \
#     evaluation=natl60_dc20a \
#     pretrained=default \
#     ++data.batch_size=2048 \
#     ++logger.mode="offline" \
#     ++optimizer.learning_rate=1e-4 \
#     ++data.train_size=0.90 \
#     ++model.basis_net.ard=True \
#     dataset="nadir4"
 
#python /gpfswork/rech/cli/uvo53rl/projects/jejeqx/scripts/dc20a/main.py \
#    stage="train" \
#    num_epochs=5 \
#    data=nadir \
#    model=ffn \
#    evaluation=natl60_dc20a \
#    pretrained=default \
#    ++logger.mode="disabled" \
#    ++optimizer.learning_rate=1e-4 \
#    ++data.train_size=0.90 \
#    dataset="nadir4"
    
    