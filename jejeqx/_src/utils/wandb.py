import wandb
from pathlib import Path


def download_artifact():
    pass


def load_wandb_run_config(entity: str, project: str, id: str):

    api = wandb.Api()
    reference = str(f"{entity}/{project}/{id}")
    prev_run = api.run(reference)
    # prev_run = api.run("ige/inr4ssh/pbi50xfu")
    prev_cfg = prev_run.config

    return prev_cfg


def download_wandb_artifact(
    entity: str,
    project: str,
    reference: str,
    mode: str = "offline",
    root=None,
):
    """Loads a checkpoint given a reference
    Args:
        reference (str): _description_
        mode (str, optional): _description_. Defaults to "disabled".
    Returns:
        _type_: _description_
    """
    api = wandb.Api()
    reference = f"{entity}/{project}/{reference}"
    artifact = api.artifact(reference)
    artifact_dir = artifact.download()

    return artifact_dir


def get_checkpoint_filename(directory: str, name: str) -> str:
    files = list(Path(directory).glob(f"*{name}*"))
    
    assert len(files) == 1
    
    return files[0]

def download_wandb_artifact_model(entity, project, reference, model_name):
    
    artifact_dir = download_wandb_artifact(
        entity=entity, project=project, reference=reference
    )
    
    filename = get_checkpoint_filename(artifact_dir, model_name)
    
    return filename
