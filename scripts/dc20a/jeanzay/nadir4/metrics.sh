#!/bin/bash
"""
Metrics Script for NADIR4
"""

## SIREN
#python /gpfswork/rech/cli/uvo53rl/projects/jejeqx/scripts/dc20a/main.py \
#    stage="inference" \
#    data=nadir \
#    evaluation=natl60_dc20a \
#    ++logger.mode="disabled" \
#    pretrained=siren_nadir_natl60
    
#python /gpfswork/rech/cli/uvo53rl/projects/jejeqx/scripts/dc20a/main.py \
#    stage="inference" \
#    data=nadir \
#    evaluation=natl60_dc20a \
#    ++logger.mode="disabled" \
#    pretrained=siren_nadir

#python /gpfswork/rech/cli/uvo53rl/projects/jejeqx/scripts/dc20a/main.py \
#    stage="inference" \
#    data=nadir \
#    evaluation=natl60_dc20a \
#    ++logger.mode="disabled" \
#    pretrained=siren_nadir_more
    


python /gpfswork/rech/cli/uvo53rl/projects/jejeqx/scripts/dc20a/main.py \
    stage="inference" \
    data=nadir \
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