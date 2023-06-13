#!/bin/bash
"""
Metrics Script for SWOT1NADIR5
"""

#python /gpfswork/rech/cli/uvo53rl/projects/jejeqx/scripts/dc20a/main.py \
#    stage="inference" \
#    data=swot \
#    evaluation=natl60_dc20a \
#    ++logger.mode="disabled" \
#    pretrained=siren_swot


python /gpfswork/rech/cli/uvo53rl/projects/jejeqx/scripts/dc20a/main.py \
    stage="inference" \
    data=swot \
    evaluation=natl60_dc20a \
    ++logger.mode="disabled" \
    pretrained=ffn_nadir




#python /gpfswork/rech/cli/uvo53rl/projects/jejeqx/scripts/dc20a/main.py \
#    stage=train \
#    num_epochs=1 \
#    ++logger.mode="disabled" \
#    data=natl60 \
#    evaluation=natl60_sim \
#    ++data.subset_size=0.25
