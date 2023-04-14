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
python /gpfswork/rech/cli/uvo53rl/projects/jejeqx/scripts/dc20a/main.py \
    stage="train" \
    num_epochs=1000 \
    data=nadir \
    model=ffn \
    evaluation=natl60_dc20a \
    pretrained=default \
    ++data.batch_size=32 \
    ++logger.mode="offline" \
    ++optimizer.learning_rate=1e-5 \
    ++data.train_size=0.90 \
    ++model.basis_net.ard=False \
    dataset="nadir4"
 
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
    
    