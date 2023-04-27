
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

| Checkpoint | Run | Type | Training | Run | Experiment |
|:----------:|:--:|:-----:|:--------:|:---:|:----------:|
| `ige/jax4eo/experiments-ckpts:v63` | `ige/jax4eo/ikcsms0z` | MLP | Scratch | https://wandb.ai/ige/jax4eo/runs/ikcsms0z?workspace=user-emanjohnson91 | NADIR4 |
| `ige/jax4eo/experiments-ckpts:v74` | `ige/jax4eo/kqdti10n` | FFN | Scratch | https://wandb.ai/ige/jax4eo/runs/tzmywo1b/overview?workspace=user-emanjohnson91 | NADIR4 |
| `ige/jax4eo/experiments-ckpts:v65` | `ige/jax4eo/xc8cl4bj` | SIREN | Scratch | https://wandb.ai/ige/jax4eo/runs/tzmywo1b/overview?workspace=user-emanjohnson91 | NADIR4 |
| `ige/jax4eo/experiments-ckpts:v89` | `ige/jax4eo/cnq231o4` | MLP | Scratch | https://wandb.ai/ige/jax4eo/runs/ikcsms0z?workspace=user-emanjohnson91 | SWOT1NADIR5 |
| `ige/jax4eo/experiments-ckpts:v94` | `ige/jax4eo/sdgtvt1i` | FFN | Scratch | https://wandb.ai/ige/jax4eo/runs/tzmywo1b/overview?workspace=user-emanjohnson91 | SWOT1NADIR5 |
| `ige/jax4eo/experiments-ckpts:v95` | `ige/jax4eo/uovjl187` | SIREN | Scratch | https://wandb.ai/ige/jax4eo/runs/tzmywo1b/overview?workspace=user-emanjohnson91 | SWOT1NADIR5 |


### NADIR 4

**From Scratch**


```python
# MLP 
python download_checkpoint.py -p /gpfswork/rech/cli/uvo53rl/checkpoints/nerfs/mlp/nadir4/scratch -r ige/jax4eo/ikcsms0z -c ige/jax4eo/experiments-ckpts:v63
# RANDOM FOURIER FEATURES
python download_checkpoint.py -p /gpfswork/rech/cli/uvo53rl/checkpoints/nerfs/ffn/nadir4/scratch -r ige/jax4eo/kqdti10n -c ige/jax4eo/experiments-ckpts:v74
# SIREN
python download_checkpoint.py -p /gpfswork/rech/cli/uvo53rl/checkpoints/nerfs/siren/nadir4/scratch -r ige/jax4eo/xc8cl4bj -c ige/jax4eo/experiments-ckpts:v65
```


### SWOT

**From Scratch**

```python
# MLP 
python download_checkpoint.py -p /gpfswork/rech/cli/uvo53rl/checkpoints/nerfs/mlp/swot1nadir5/scratch -r ige/jax4eo/cnq231o4 -c ige/jax4eo/experiments-ckpts:v89
# RANDOM FOURIER FEATURES
python download_checkpoint.py -p /gpfswork/rech/cli/uvo53rl/checkpoints/nerfs/ffn/swot1nadir5/scratch -r ige/jax4eo/sdgtvt1i -c ige/jax4eo/experiments-ckpts:v94
# SIREN
python download_checkpoint.py -p /gpfswork/rech/cli/uvo53rl/checkpoints/nerfs/siren/swot1nadir5/scratch -r ige/jax4eo/uovjl187 -c ige/jax4eo/experiments-ckpts:v95
```
