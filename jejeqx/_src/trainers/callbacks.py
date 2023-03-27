import glob
import os
import wandb


def wandb_model_artifact(trainer):
    
    experiment = trainer.pl_logger.experiment
    
    ckpts = wandb.Artifact("experiments-ckpts", type="checkpoints")
    
    for path in glob.glob(
        os.path.join(trainer.log_dir, "**/*.ckpt"), recursive=True
    ):
        ckpts.add_file(path)
        
    experiment.log_artifact(ckpts)
    
    
    