import autoroot
import hydra
from loguru import logger
import jax
import jax.numpy as jnp
from jejeqx._src.trainers.base import TrainerModule
from jejeqx._src.trainers.callbacks import wandb_model_artifact
from jejeqx._src.losses import psnr
import equinox as eqx
from pathlib import Path
import wandb
from omegaconf import OmegaConf
import train
import inference


@hydra.main(config_path="config", config_name="main", version_base="1.2")
def main(cfg):

    if cfg.stage == "train":
        logger.info(f"Starting training stage...!")
        train.main(cfg)

    elif cfg.stage == "train_more":
        raise NotImplementedError()

    elif cfg.stage == "inference":
        logger.info(f"Starting inference stage...!")
        inference.main(cfg)

    elif cfg.stage == "metrics":
        raise NotImplementedError()

    elif cfg.stage == "viz":
        raise NotImplementedError()

    else:
        raise ValueError(f"Unrecognized stage: {cfg.stage}")


if __name__ == "__main__":
    main()
