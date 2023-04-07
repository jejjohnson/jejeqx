import time
from typing import Dict, Any, Optional, Iterator, List, Tuple
from jaxtyping import Array, PyTree
import jax
import numpy as np
import optax
import equinox as eqx
from tqdm import tqdm
from jejeqx._src.trainers.trainstate import TrainState
from collections import defaultdict
from pathlib import Path




# WANDBLOGGER = pl


class TrainerModule:
    def __init__(
        self,
        model: eqx.Module,
        optimizer: optax.GradientTransformation,
        seed: int = 42,
        pl_logger: Optional = None,
        enable_progress_bar: bool = True,
        debug: bool = False,
        check_val_every_n_epoch: int = 1,
        log_dir: str = "",
        save_name: str = "checkpoint_model.ckpt",
        **kwargs,
    ):
        """
        A basic Trainer module summarizing most common training functionalities
        like logging, model initialization, training loop, etc.

        Atributes:
          model (eqx.Module): The class of the model that should be trained.
          optimizer: A dictionary of all hyperparameters of the optimizer.
            Used during initialization of the optimizer.
          exmp_input: Input to the model for initialization and tabulate.
          seed: Seed to initialize PRNG.
          logger_params: A dictionary containing the specification of the logger.
          enable_progress_bar: If False, no progress bar is shown.
          debug: If True, no jitting is applied. Can be helpful for debugging.
          check_val_every_n_epoch: The frequency with which the model is evaluated
            on the validation set.
        """
        super().__init__()
        self.optimizer = optimizer
        self.enable_progress_bar = enable_progress_bar
        self.debug = debug
        self.seed = seed
        self.check_val_every_n_epochs = check_val_every_n_epoch
        self.log_dir = log_dir
        # init trainer parts
        self.pl_logger = pl_logger
        self.create_jitted_functions()
        self.state = TrainState(params=model, tx=optimizer)
        self.save_name = save_name
        
    @property
    def model(self):
        return self.state.params

    @property
    def model_batch(self):
        return jax.vmap(self.state.params)
    

    def create_jitted_functions(self):
        train_step, eval_step, predict_step = self.create_functions()
        if self.debug:
            self.train_step = train_step
            self.eval_step = eval_step
            self.predict_step = predict_step

        else:
            self.train_step = eqx.filter_jit(train_step)
            self.eval_step = eqx.filter_jit(eval_step)
            self.predict_step = eqx.filter_jit(predict_step)

    def create_functions(self):
        def train_step(state: TrainState, batch: Any):
            metrics = {}
            return state, metrics

        def eval_step(state: TrainState, batch: Any):
            metrics = {}
            return metrics

        raise NotImplementedError

    def tracker(self, iterator: Iterator, **kwargs):
        if self.enable_progress_bar:
            return tqdm(iterator, **kwargs)
        else:
            return iterator

    def train_epoch(self, train_dataloader: Iterator, state) -> Tuple[PyTree, Dict]:
        """
        Trains a model for one epoch.

        Args:
          train_loader: Data loader of the training set.

        Returns:
          A dictionary of the average training metrics over all batches
          for logging.
        """
        # Train model for one epoch, and log avg loss and accuracy
        metrics = defaultdict(float)
        num_train_steps = len(train_dataloader)
        start_time = time.time()
        for batch in self.tracker(train_dataloader, desc="Training", leave=False):
            # self.state, loss, step_metrics = self.train_step(self.state, batch)
            state, loss, step_metrics = self.train_step(state, batch)
            for key in step_metrics:
                metrics["train/" + key] += step_metrics[key] / num_train_steps
        metrics = {key: metrics[key].item() for key in metrics}
        metrics["epoch_time"] = time.time() - start_time
        return state, metrics

    def eval_model(self, dataloader: Iterator, model, log_prefix=""):
        metrics = defaultdict(float)
        num_elements = 0
        for batch in dataloader:
            step_metrics = self.eval_step(model, batch)
            
            if isinstance(batch, (list, tuple)):
                batch_size = batch[0].shape[0]
            elif isinstance(batch, dict):
                batch_size = list(batch.values())[0].shape[0]
            else:
                batch_size = batch.shape[0]
            
            for key in step_metrics:
                metrics[key] += step_metrics[key] * batch_size

            num_elements += batch_size
        metrics = {
            (log_prefix + key): (metrics[key] / num_elements).item() for key in metrics
        }
        return metrics
    
    def predict_model(self, dataloader, log_prefix=""):
        metrics = defaultdict(float)
        
        model = self.state.params
        num_elements = 0
        out = list()
        for batch in dataloader:
            pred, step_metrics = self.predict_step(model, batch)
            
            if isinstance(batch, (list, tuple)):
                batch_size = batch[0].shape[0]
            elif isinstance(batch, dict):
                batch_size = list(batch.values())[0].shape[0]
            else:
                batch_size = batch.shape[0]
            
            for key in step_metrics:
                metrics[key] += step_metrics[key] * batch_size
                
            num_elements += batch_size
            out.append(pred)
            
        metrics = {
            (log_prefix + key): (metrics[key] / num_elements).item() for key in metrics
        }
        out = np.vstack(out)
        return out, metrics
        

    def train_model(self, dm, num_epochs: int = 500):
        self.on_training_start()
        state = self.state
        best_eval_metrics = None
        with tqdm(range(1, num_epochs + 1), desc="Epochs") as pbar_epoch:
            for epoch_idx in pbar_epoch:
                self.on_training_epoch_start(epoch_idx)

                state, train_metrics = self.train_epoch(dm.train_dataloader(), state)

                pbar_epoch.set_description(
                    f"Epochs: {epoch_idx} | Loss: {train_metrics['train/loss']:.3e}"
                )
                if self.pl_logger:
                    self.pl_logger.log_metrics(train_metrics, step=epoch_idx)
                self.on_training_epoch_end(epoch_idx)

                if epoch_idx % self.check_val_every_n_epochs == 0:
                    eval_metrics = self.eval_model(
                        dm.val_dataloader(), state.params, log_prefix="val/"
                    )
                    self.on_validation_epoch_end(
                        epoch_idx, eval_metrics, dm.val_dataloader()
                    )
                    if self.pl_logger:
                        self.pl_logger.log_metrics(eval_metrics, step=epoch_idx)
                        
                    if self.is_new_model_better(eval_metrics, best_eval_metrics):
                        best_eval_metrics = eval_metrics
                        best_eval_metrics.update(train_metrics)
                        self.save_model()

        self.state = state
        self.on_training_end()
        return train_metrics
    
    
    def is_new_model_better(self,
                            new_metrics : Dict[str, Any],
                            old_metrics : Dict[str, Any]) -> bool:
        """
        Compares two sets of evaluation metrics to decide whether the
        new model is better than the previous ones or not.

        Args:
          new_metrics: A dictionary of the evaluation metrics of the new model.
          old_metrics: A dictionary of the evaluation metrics of the previously
            best model, i.e. the one to compare to.

        Returns:
          True if the new model is better than the old one, and False otherwise.
        """
        if old_metrics is None:
            return True
        for key, is_larger in [('val/val_metric', False), ('val/acc', True), ('val/loss', False)]:
            if key in new_metrics:
                if is_larger:
                    return new_metrics[key] > old_metrics[key]
                else:
                    return new_metrics[key] < old_metrics[key]
        assert False, f'No known metrics to log on: {new_metrics}'
        
    def save_metrics(self,
                     filename : str,
                     metrics : Dict[str, Any]):
        """
        Saves a dictionary of metrics to file. Can be used as a textual
        representation of the validation performance for checking in the terminal.

        Args:
          filename: Name of the metrics file without folders and postfix.
          metrics: A dictionary of metrics to save in the file.
        """
        with open(os.path.join(self.log_dir, f'metrics/{filename}.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

    def on_training_start(self):
        pass

    def on_training_epoch_start(self, epoch_idx: int):
        pass

    def on_training_epoch_end(self, epoch_idx: int):
        pass

    def on_validation_epoch_end(self, epoch_idx, eval_metrics, dataloader):
        pass

    def on_training_end(self):
        if self.pl_logger:
            self.pl_logger.finalize("success")

    def save_state(self, name: Optional[str] = None):
        if name is None:
            name = "checkpoint_state.ckpts"
        from pathlib import Path

        path = Path(self.log_dir).joinpath(name)
        eqx.tree_serialise_leaves(str(path), self.state)

    def load_state(self, name: str):
        state = eqx.tree_deserialise_leaves(f"{name}", self.state)
        self.state = state

    def save_model(self, name: Optional[str]=None):
        
        if name is None:
            name = self.save_name
        from pathlib import Path

        path = Path(self.log_dir).joinpath(name)
        eqx.tree_serialise_leaves(str(path), self.state.params)

    def load_model(self, name: str):
        params = eqx.tree_deserialise_leaves(f"{name}", self.state.params)
        self.state = eqx.tree_at(lambda x: x.params, self.state, params)

