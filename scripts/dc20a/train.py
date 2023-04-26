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
import optax
from omegaconf import OmegaConf
import joblib
import utils
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'

class RegressorTrainer(TrainerModule):
    def __init__(self,
                 model,
                 optimizer,
                 pl_logger=None,
                 **kwargs):
        super().__init__(
            model=model, 
            optimizer=optimizer, 
            pl_logger=pl_logger,
            log_dir=pl_logger.experiment.dir,
            **kwargs
        )
        
    @property
    def model(self):
        return self.state.params

    @property
    def model_batch(self):
        return jax.vmap(self.state.params, in_axes=(0,0))

    def create_functions(self):

        @eqx.filter_value_and_grad
        def mse_loss(model, batch):
            x, t, y = batch["spatial"], batch["temporal"], batch["data"]
            # pred = jax.vmap(model, in_axes=(0,0))(x, t)
            pred = jax.vmap(model)(jnp.hstack([x, t]))
            loss = jnp.mean((y - pred)**2)
            return loss
        
        def train_step(state, batch):
            
            loss, grads = mse_loss(state.params, batch)
            state = state.update_state(state, grads)
            psnr_loss = psnr(loss) 
            metrics = {"loss": loss, "psnr": psnr_loss}
            return state, loss, metrics

        def eval_step(model, batch):
            loss, _ = mse_loss(model, batch)
            psnr_loss = psnr(loss) 
            return {"loss": loss, "psnr": psnr_loss}
        
        def test_step(model, batch):
            x, t, y = batch["spatial"], batch["temporal"], batch["data"]
            pred = jax.vmap(model)(jnp.hstack([x, t]))
            loss, _ = mse_loss(model, batch)
            psnr_loss = psnr(loss)
            return pred, {"loss": loss, "psnr": psnr_loss}
        
        def predict_step(model, batch):
            x, t = batch["spatial"], batch["temporal"]
            pred = jax.vmap(model)(jnp.hstack([x, t]))
            return pred

        return train_step, eval_step, test_step, predict_step
    
    def on_training_end(self,):
        
        if self.pl_logger:
            save_dir = Path(self.log_dir).joinpath(self.save_name)
            self.save_model(save_dir)
            wandb_model_artifact(self)
            self.pl_logger.finalize("success")


@hydra.main(config_path='config', config_name='main', version_base='1.2')
def main(cfg):
        
    
    # initialize logger
    wandb_config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
        
    wandb_logger = hydra.utils.instantiate(cfg.logger)
    
    wandb_logger.experiment.config.update(wandb_config)
    
    
    logger.info("Starting!")
    
    logger.info("Initializing spatial transforms...")
    
    spatial_transforms = hydra.utils.instantiate(
        cfg.spatial_transforms, 
        _recursive_=False
    )
    
    logger.info(f"Initializing temporal transforms...")
    temporal_transforms = hydra.utils.instantiate(
        cfg.temporal_transforms, 
        _recursive_=False
    )
    
    logger.info(f"Initializing datamodule...")
    dm = hydra.utils.instantiate(
        cfg.data, 
        spatial_transform=spatial_transforms,
        temporal_transform=temporal_transforms,
    )
    
    dm.setup()
    
    logger.info(f"Number Training: {len(dm.ds_train):_}")
    logger.info(f"Number Validation: {len(dm.ds_valid):_}")
    logger.info(f"Number Testing: {len(dm.ds_test):_}")
    
    init = dm.ds_train[:32]
    x_init, t_init, y_init = init["spatial"], init["temporal"], init["data"]
    
    
    logger.info(cfg.pretrained.config)
    logger.info(cfg.pretrained.checkpoint)
    if cfg.pretrained.config is not None and cfg.pretrained.checkpoint is not None:
        logger.info(f"Loading old model...")
        logger.info(f"{cfg.pretrained.config}")
        old_config = joblib.load(cfg.pretrained.config)
        logger.info(f"Initializing model...")
        model = hydra.utils.instantiate(old_config["model"])
        logger.info(f"Updating config...")
        wandb_logger.experiment.config["model"].update(old_config["model"])
    else:
        logger.info(f"Initializing model...")
        model = hydra.utils.instantiate(cfg.model)
    
    # check output of models
    out = jax.vmap(model)(jnp.hstack([x_init,t_init]))
    assert out.shape == y_init.shape
    
    logger.info(f"Initializing optimizer...")
    optimizer = hydra.utils.instantiate(cfg.optimizer)

    if cfg.lr_scheduler.scheduler is not None:
        logger.info(f"Initializing scheduler...")

        scheduler = hydra.utils.instantiate(
            cfg.lr_scheduler.scheduler, 
            decay_steps=int(cfg.num_epochs * len(dm.ds_train))
        )
        optimizer = optax.chain(optimizer, optax.scale_by_schedule(scheduler))

    
    logger.info(f"Initializing Trainer...")
    trainer = RegressorTrainer(
        model, 
        optimizer, 
        seed=123, 
        debug=False, 
        enable_progress_bar=True,
        pl_logger=wandb_logger
    )
    
    logger.info(f"Metrics BEFORE Training...")
    out, metrics = trainer.test_model(dm.test_dataloader())
    logger.info(f"Loss (MSE): {metrics['loss']:.2e}")
    logger.info(f"Loss (PSNR): {metrics['psnr']:.4f}")
    

    if cfg.pretrained.config is not None and cfg.pretrained.checkpoint is not None:
        logger.info(f"Loading old checkpoint...")
        logger.info(f"{cfg.pretrained.checkpoint}")
        checkpoint_file = cfg.pretrained.checkpoint
        trainer.load_model(checkpoint_file)
        
        out, metrics = trainer.test_model(dm.test_dataloader())
        logger.info(f"Metrics WITH PreTrained Model...")
        logger.info(f"Loss (MSE): {metrics['loss']:.2e}")
        logger.info(f"Loss (PSNR): {metrics['psnr']:.4f}")
    
    
    logger.info(f"Starting Training...")
    train_metrics = trainer.train_model(dm, num_epochs=cfg.num_epochs)
    logger.info(f"Finished Training...!")
    
    logger.info(f"Starting Testing...")
    out, test_metrics = trainer.test_model(dm.test_dataloader())
    logger.info(f"Loss (MSE): {test_metrics['loss']:.2e}")
    logger.info(f"Loss (PSNR): {test_metrics['psnr']:.4f}")
    
    
    logger.info(f"Loading Evaluation datamodule...")
    dm_eval = hydra.utils.instantiate(
            cfg.evaluation, 
            spatial_transform=dm.spatial_transform,
            temporal_transform=dm.temporal_transform,
        )
    dm_eval.setup()
    
    logger.info(f"Number Training: {len(dm_eval.ds_test):_}")
    
    logger.info(f"Testing on Evaluation...")
    out, eval_metrics = trainer.test_model(dm_eval.test_dataloader())
    logger.info(f"Loss (MSE): {eval_metrics['loss']:.2e}")
    logger.info(f"Loss (PSNR): {eval_metrics['psnr']:.4f}")
    
    
    logger.info(f"Create xarray dataset...")
    xrda = dm_eval.load_xrds()
    
    logger.info(f"Create add predictions to model...")
    xrda["ssh_model"] = dm_eval.data_to_df(out).to_xarray().sossheig
    
    logger.info(f"Calculate Physical Quantities...")
    ds_model = utils.calculate_physical_quantities(xrda.ssh_model)
    ds_natl60 = utils.calculate_physical_quantities(xrda.sossheig)
    
    logger.info(f"SpatialTemporal Statistics...")
    
    for ivar in ds_model:
        
        
        rmse_xyt = utils.nrmse_da(ds_model[ivar], ds_natl60[ivar], dim=["lat", "lon", "time"])
        logger.info(f"nrmse [mean]: {rmse_xyt:.2f} [{ivar.upper()}]")
        wandb_logger.log_metrics({f"{ivar}_nrmse_mean": rmse_xyt})
        
        
        rmse_t = utils.nrmse_da(ds_model[ivar], ds_natl60[ivar], dim=["lat", "lon"]).std()
        logger.info(f"nrmse [std]: {rmse_t:.2f} [{ivar.upper()}]")
        wandb_logger.log_metrics({f"{ivar}_nrmse_std": rmse_t})
    
    logger.info(f"Calculate PSD Isotropic Scores...")
    ds_psd_score = utils.calculate_isotropic_psd_score(ds_model, ds_natl60)
    
    logger.info(f"Resolved Isotropic Scales:")
    for ivar in ds_psd_score:
        resolved_spatial_scale = ds_psd_score[ivar].attrs["resolved_scale_space"] / 1e3 
        logger.info(f"Wavelength [km]: {resolved_spatial_scale:.2f} [{ivar.upper()}]")
        logger.info(f"Wavelength [degree]: {resolved_spatial_scale/111:.2f} [{ivar.upper()}]")
        wandb_logger.log_metrics({f"{ivar}_iso_psd_spatial_km": resolved_spatial_scale})
        wandb_logger.log_metrics({f"{ivar}_iso_psd_spatial_degree": resolved_spatial_scale/111})
    
    logger.info(f"PSD SpaceTime Scores")
    ds_psd_score = utils.calculate_spacetime_psd_score(ds_model, ds_natl60)
    
    logger.info(f"Resolved SpaceTime Scales:")
    for ivar in ds_psd_score:
        resolved_spatial_scale = ds_psd_score[ivar].attrs["resolved_scale_space"] / 1e3
        
        wandb_logger.log_metrics({f"{ivar}_psd_spatial_km": resolved_spatial_scale})
        logger.info(f"Wavelength [km]: {resolved_spatial_scale:.2f} [{ivar.upper()}]")
        
        wandb_logger.log_metrics({f"{ivar}_psd_spatial_degree": resolved_spatial_scale/111})
        logger.info(f"Wavelength [degree]: {resolved_spatial_scale/111:.2f} [{ivar.upper()}]")
        
        resolved_temporal_scale = ds_psd_score[ivar].attrs["resolved_scale_time"]
        wandb_logger.log_metrics({f"{ivar}_psd_time_days": resolved_temporal_scale})
        logger.info(f"Period [days]: {resolved_temporal_scale:.2f}  [{ivar.upper()}]")
    
    
    

if __name__ == "__main__":
    main()