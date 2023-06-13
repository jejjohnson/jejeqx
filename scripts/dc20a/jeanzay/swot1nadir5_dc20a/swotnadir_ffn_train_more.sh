#!/bin/bash
python /gpfswork/rech/cli/uvo53rl/projects/jejeqx/scripts/dc20a/main.py \
    stage="train" \
    num_epochs=2000 \
    data=swot_dc20a \
    model=mlp \
    evaluation=natl60_dc20a \
    pretrained=mlp_swotnadir_dc20a \
    ++data.batch_size=50_000 \
    ++logger.mode="disabled" \
    ++optimizer.learning_rate=5e-5 \
    ++data.train_size=0.90 \
    dataset="swot1nadir5" &
