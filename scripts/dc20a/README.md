
# Download the Data


## NATL60 Simulations

### Full Dataset

```bash

```


### Temporally Resampled (1D)

```bash

```

# Run Inference

```bash
# MLP | NADIR
python main.py stage=inference pretrained=mlp_nadir_dc20a.yaml data=nadir_dc20a ++results.name=nerf_mlp_nadir_dc20a ++logger.mode="disabled" evaluation=natl60_dc20a
# MLP | SWOTNADIR
python main.py stage=inference pretrained=mlp_swotnadir_dc20a.yaml data=swot_dc20a ++results.name=nerf_mlp_swot_dc20a ++logger.mode="disabled" evaluation=natl60_dc20a
# FFN | NADIR
python main.py stage=inference pretrained=ffn_nadir_dc20a.yaml data=nadir_dc20a ++results.name=nerf_ffn_nadir_dc20a ++logger.mode="disabled" evaluation=natl60_dc20a
# FFN | SWOTNADIR
python main.py stage=inference pretrained=ffn_swotnadir_dc20a.yaml data=swot_dc20a ++results.name=nerf_ffn_swot_dc20a ++logger.mode="disabled" evaluation=natl60_dc20a
# SIREN | NADIR
python main.py stage=inference pretrained=siren_nadir_dc20a.yaml data=nadir_dc20a ++results.name=nerf_siren_nadir_dc20a ++logger.mode="disabled" evaluation=natl60_dc20a
# SIREN | SWOTNADIR
python main.py stage=inference pretrained=siren_swotnadir_dc20a.yaml data=swot_dc20a ++results.name=nerf_siren_swot_dc20a ++logger.mode="disabled" evaluation=natl60_dc20a
```

# Download a Checkpoint

## NATL60 Simulation (Full)

```python
python download_checkpoint.py -p /gpfswork/rech/cli/uvo53rl/checkpoints/nerfs/siren/natl60/full/ -r ige/jax4eo/r6y4hg38 -c ige/jax4eo/experiments-ckpts:v26
```

**TRIAL I**

| Checkpoint | Run | Type | Training | Run | Experiment |
|:----------:|:--:|:-----:|:--------:|:---:|:----------:|
| `ige/jax4eo/experiments-ckpts:v63` | `ige/jax4eo/ikcsms0z` | MLP | Scratch | https://wandb.ai/ige/jax4eo/runs/ikcsms0z?workspace=user-emanjohnson91 | NADIR4 |
| `ige/jax4eo/experiments-ckpts:v74` | `ige/jax4eo/kqdti10n` | FFN | Scratch | https://wandb.ai/ige/jax4eo/runs/tzmywo1b/overview?workspace=user-emanjohnson91 | NADIR4 |
| `ige/jax4eo/experiments-ckpts:v109` | `ige/jax4eo/xc8cl4bj` | SIREN | Scratch | https://wandb.ai/ige/jax4eo/runs/tzmywo1b/overview?workspace=user-emanjohnson91 | NADIR4 |
| `ige/jax4eo/experiments-ckpts:v89` | `ige/jax4eo/cnq231o4` | MLP | Scratch | https://wandb.ai/ige/jax4eo/runs/ikcsms0z?workspace=user-emanjohnson91 | SWOT1NADIR5 |
| `ige/jax4eo/experiments-ckpts:v101`, `94`| `ige/jax4eo/52tqyi8h`, `ige/jax4eo/sdgtvt1i` | FFN | Scratch | https://wandb.ai/ige/jax4eo/runs/tzmywo1b/overview?workspace=user-emanjohnson91 | SWOT1NADIR5 |
| `ige/jax4eo/experiments-ckpts:v102` | `ige/jax4eo/h1gtxukk` | FFN | TRAIN MORE | https://wandb.ai/ige/jax4eo/runs/tzmywo1b/overview?workspace=user-emanjohnson91 | SWOT1NADIR5 |
| `ige/jax4eo/experiments-ckpts:v95` | `ige/jax4eo/uovjl187` | SIREN | Scratch | https://wandb.ai/ige/jax4eo/runs/tzmywo1b/overview?workspace=user-emanjohnson91 | SWOT1NADIR5 |


### NADIR 4

**From Scratch**


```python
# MLP | SCRATCH
python download_checkpoint.py -p /gpfswork/rech/cli/uvo53rl/checkpoints/nerfs/mlp/nadir4/scratch -r ige/jax4eo/xc8cl4bj -c ige/jax4eo/experiments-ckpts:v63
python download_checkpoint.py -p /gpfswork/rech/cli/uvo53rl/checkpoints/nerfs/mlp/nadir4/scratch -r ige/jax4eo/ikcsms0z -c ige/jax4eo/experiments-ckpts:v109
# RANDOM FOURIER FEATURES | SCRATCH
python download_checkpoint.py -p /gpfswork/rech/cli/uvo53rl/checkpoints/nerfs/ffn/nadir4/scratch -r ige/jax4eo/kqdti10n -c ige/jax4eo/experiments-ckpts:v74
python download_checkpoint.py -p /gpfswork/rech/cli/uvo53rl/checkpoints/nerfs/ffn/nadir4/scratch -r ige/jax4eo/6wklm1mi -c ige/jax4eo/experiments-ckpts:v107
python download_checkpoint.py -p /gpfswork/rech/cli/uvo53rl/checkpoints/nerfs/ffn/nadir4/scratch -r ige/jax4eo/vcsjhsxb -c ige/jax4eo/experiments-ckpts:v113
# SIREN | SCRATCH
python download_checkpoint.py -p /gpfswork/rech/cli/uvo53rl/checkpoints/nerfs/siren/nadir4/scratch -r ige/jax4eo/xc8cl4bj -c ige/jax4eo/experiments-ckpts:v65
python download_checkpoint.py -p /gpfswork/rech/cli/uvo53rl/checkpoints/nerfs/siren/nadir4/scratch -r ige/jax4eo/s12gumax -c ige/jax4eo/experiments-ckpts:v112
```


### SWOT


```python
# MLP
python download_checkpoint.py -p /gpfswork/rech/cli/uvo53rl/checkpoints/nerfs/mlp/swot1nadir5/scratch -r ige/jax4eo/cnq231o4 -c ige/jax4eo/experiments-ckpts:v89
# RANDOM FOURIER FEATURES | SCRATCH
python download_checkpoint.py -p /gpfswork/rech/cli/uvo53rl/checkpoints/nerfs/ffn/swot1nadir5/scratch -r ige/jax4eo/52tqyi8h -c ige/jax4eo/experiments-ckpts:v101
# RANDOM FOURIER FEATURES | TRAIN MORE
python download_checkpoint.py -p /gpfswork/rech/cli/uvo53rl/checkpoints/nerfs/ffn/swot1nadir5/scratch -r ige/jax4eo/h1gtxukk -c ige/jax4eo/experiments-ckpts:v102
# RANDOM FOURIER FEATURES | PRETRAIN
python download_checkpoint.py -p /gpfswork/rech/cli/uvo53rl/checkpoints/nerfs/ffn/swot1nadir5/scratch -r ige/jax4eo/gtqurpzl -c ige/jax4eo/experiments-ckpts:v99
python download_checkpoint.py -p /gpfswork/rech/cli/uvo53rl/checkpoints/nerfs/ffn/swot1nadir5/scratch -r ige/jax4eo/vnvwe0h6 -c ige/jax4eo/experiments-ckpts:v115
# SIREN
# python download_checkpoint.py -p /gpfswork/rech/cli/uvo53rl/checkpoints/nerfs/siren/swot1nadir5/scratch -r ige/jax4eo/uovjl187 -c ige/jax4eo/experiments-ckpts:v95
# SIREN | SCRATCH | LARGE BATCH
python download_checkpoint.py -p /gpfswork/rech/cli/uvo53rl/checkpoints/nerfs/siren/swot1nadir5/scratch -r ige/jax4eo/d6xysoi8 -c ige/jax4eo/experiments-ckpts:v96
# SIREN | PRETRAINED | LARGE BATCH
python download_checkpoint.py -p /gpfswork/rech/cli/uvo53rl/checkpoints/nerfs/siren/swot1nadir5/scratch -r ige/jax4eo/cpcut06s -c ige/jax4eo/experiments-ckpts:v98
python download_checkpoint.py -p /gpfswork/rech/cli/uvo53rl/checkpoints/nerfs/siren/swot1nadir5/scratch -r ige/jax4eo/ldrv7y8x -c ige/jax4eo/experiments-ckpts:v132
```

**Train More**

```python
# MLP
python download_checkpoint.py -p /gpfswork/rech/cli/uvo53rl/checkpoints/nerfs/mlp/swot1nadir5/scratch -r ige/jax4eo/cnq231o4 -c ige/jax4eo/experiments-ckpts:v89
# RANDOM FOURIER FEATURES
python download_checkpoint.py -p /gpfswork/rech/cli/uvo53rl/checkpoints/nerfs/ffn/swot1nadir5/pretrained -r ige/jax4eo/52tqyi8h -c ige/jax4eo/experiments-ckpts:v101
# SIREN
python download_checkpoint.py -p /gpfswork/rech/cli/uvo53rl/checkpoints/nerfs/siren/swot1nadir5/pretrained -r ige/jax4eo/g7rfcz0d -c ige/jax4eo/experiments-ckpts:v103
```

**PreTrained**

```python
# MLP
python download_checkpoint.py -p /gpfswork/rech/cli/uvo53rl/checkpoints/nerfs/mlp/swot1nadir5/scratch -r ige/jax4eo/cnq231o4 -c ige/jax4eo/experiments-ckpts:v89
# RANDOM FOURIER FEATURES
python download_checkpoint.py -p /gpfswork/rech/cli/uvo53rl/checkpoints/nerfs/ffn/swot1nadir5/pretrained -r ige/jax4eo/52tqyi8h -c ige/jax4eo/experiments-ckpts:v101
# SIREN
python download_checkpoint.py -p /gpfswork/rech/cli/uvo53rl/checkpoints/nerfs/siren/swot1nadir5/pretrained -r ige/jax4eo/g7rfcz0d -c ige/jax4eo/experiments-ckpts:v103
```
