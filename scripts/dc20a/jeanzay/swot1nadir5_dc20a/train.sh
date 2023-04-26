#!/bin/bash


# ===============================================
# SIREN 
# ===============================================

# CONFIG I - FROM SCRATCH
# NADIR4 | NO PRETRAINING | EVAL AREA
# ++logger.mode="offline"
# num_epochs=1000

python /gpfswork/rech/cli/uvo53rl/projects/jejeqx/scripts/dc20a/main.py \
    stage="train" \
    num_epochs=10 \
    data=swot_dc20a \
    model=siren \
    evaluation=natl60_dc20a \
    pretrained=default \
    ++data.batch_size=32 \
    ++logger.mode="disabled" \
    ++optimizer.learning_rate=1e-4 \
    ++data.subset_size=0.50 \
    ++data.train_size=0.90 \
    dataset="swot1nadir5"
    
# ===============================================
# SIREN (PRETRAINED)
# ===============================================

#python /gpfswork/rech/cli/uvo53rl/projects/jejeqx/scripts/dc20a/main.py \
#    stage="train" \
#    num_epochs=5000 \
#    data=swot \
#    evaluation=natl60_dc20a \
#    pretrained=natl60_sim \
#    ++logger.mode="offline" \
#    ++optimizer.learning_rate=1e-5 \
#    dataset="swot1nadir5"
    

# ===============================================
# RANDOM FOURIER FEATURES
# ===============================================
#python /gpfswork/rech/cli/uvo53rl/projects/jejeqx/scripts/dc20a/main.py \
#    stage="train" \
#    num_epochs=1000 \
#    data=swot \
#    model=ffn \
#    evaluation=natl60_dc20a \
#    pretrained=default \
#    ++data.batch_size=32 \
#    ++logger.mode="offline" \
#    ++model.basis_net.ard=False \
#    ++optimizer.learning_rate=1e-4 \
#    ++data.train_size=0.90 \
#    dataset="swot1nadir5"

#python /gpfswork/rech/cli/uvo53rl/projects/jejeqx/scripts/dc20a/main.py \
#    stage="train" \
#    num_epochs=1000 \
#    data=nadir \
#    evaluation=natl60_dc20a \
#    pretrained=siren_natl60_full \
#    ++logger.mode="disabled" \
#    ++optimizer.learning_rate=5e-5
    
    
    