
# Download the Data


## NATL60 Simulations

### Full Dataset

```bash

```


### Temporally Resampled (1D)

```bash

```

# Download a Checkpoint

## NATL60 Simulation (Full)

```python
python download_checkpoint.py -p /gpfswork/rech/cli/uvo53rl/checkpoints/nerfs/siren/natl60/full/ -r ige/jax4eo/r6y4hg38 -c ige/jax4eo/experiments-ckpts:v26
```


## NADIR 4


### From Scratch


#### SIREN

```python
python download_checkpoint.py -p /gpfswork/rech/cli/uvo53rl/checkpoints/nerfs/siren/nadir4/scratch -r ige/jax4eo/1nsi8rz7 -c ige/jax4eo/experiments-ckpts:v29
```

#### MLP


#### Random Fourier Features

| Checkpoint | Run | Type | Training | Run | 
| `ige/jax4eo/experiments-ckpts:v50` | `ige/jax4eo/tzmywo1b` | FFN | Scratch | https://wandb.ai/ige/jax4eo/runs/tzmywo1b/overview?workspace=user-emanjohnson91 |

```bash
ige/jax4eo/experiments-ckpts:v50
```

```python
python download_checkpoint.py -p /gpfswork/rech/cli/uvo53rl/checkpoints/nerfs/ffn/nadir4/scratch -r ige/jax4eo/tzmywo1b -c ige/jax4eo/experiments-ckpts:v50
```

### Trained More

```bash
python download_checkpoint.py -p /gpfswork/rech/cli/uvo53rl/checkpoints/nerfs/siren/nadir4/train_more -r ige/jax4eo/51roatuc -c ige/jax4eo/experiments-ckpts:v32
```

### PreTrained

```bash
python download_checkpoint.py -p /gpfswork/rech/cli/uvo53rl/checkpoints/nerfs/siren/nadir4/pretrained -r ige/jax4eo/4yhvingm -c ige/jax4eo/experiments-ckpts:v28
```


## SWOT



### From Scratch

#### SIREN

```bash
python download_checkpoint.py -p /gpfswork/rech/cli/uvo53rl/checkpoints/nerfs/siren/swot1nadir5/scratch -r ige/jax4eo/o93rers5 -c ige/jax4eo/experiments-ckpts:v44
```

#### MLP


#### RFF


```bash
python download_checkpoint.py -p /gpfswork/rech/cli/uvo53rl/checkpoints/nerfs/ffn/swot1nadir5/scratch -r ige/jax4eo/labo174g -c ige/jax4eo/experiments-ckpts:v45
```



### PreTrained


```bash
python download_checkpoint.py -p /gpfswork/rech/cli/uvo53rl/checkpoints/nerfs/siren/swot1nadir5/pretrained -r ige/jax4eo/qgqj6e7a -c ige/jax4eo/experiments-ckpts:v38
```