#!/user/bin/python
import wandb
import pickle
import argparse
from pathlib import Path
from loguru import logger
import joblib

def main(args):
    
    logger.info(f"Starting wandb api...")
    api = wandb.Api()
    logger.info(f"Loading reference...")
    prev_run = api.run(args.reference)
    prev_cfg = prev_run.config
    
    path = Path(args.path).joinpath("config.pkl")
    
    logger.info(f"Saving file:\n{path}")
    
    joblib.dump(prev_cfg, f"{path}") 
    logger.info(f"Done!")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", default="./")
    parser.add_argument("-r", "--reference")
    args = parser.parse_args()
    main(args)