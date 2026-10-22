"""Callbacks related to printing & keeping track of stuff, basically."""
from typing import override

import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_info

from lgm.types import Number
from lgm.visualization import ImagePlotter


class FitInfo(Callback):
    """Prints some data & step information at the start of training.
     
    This prints data statistics like shape, dtype, mean, std, median, min/max. Also plots example
    training data points to Tensorboard. Note that these include any augmentations or other
    transforms on the training set."""
    @override
    def __init__(self,
                 image_size: tuple[Number, Number] | Number = 1,
                 colormap: str = "Greys",
                 n_rows: int = 8,
                 n_cols: int = 16):
        """Configure the callback.

        Parameters:
            image_size: Figure will alot this size (inches) per image.
            colormap: Which colormap to use; only relevant for grayscale images.
            n_rows: Plot (up to) this many rows of data examples from the first batch.
            n_cols: Plot (up to) this many columns of data examples from the firsst batch.
        """
        super().__init__()
        self.image_size = image_size
        self.colormap = colormap
        # TODO could allow these as arguments
        self.n_rows = n_rows
        self.n_cols = n_cols

    @override
    def on_fit_start(self,
                     trainer: Trainer,
                     pl_module: LightningModule):
        inputs, targets = next(iter(trainer.datamodule.train_dataloader()))
        rank_zero_info("\n===================================================================")
        rank_zero_info("FitInfo callback output here:\n")
        rank_zero_info(f"Shape/dtype of input batch [N, C, ...]: {inputs.shape}, {inputs.dtype}")
        rank_zero_info(f"Min/max pixel values in input batch: {inputs.min().item():.3g}, "
                       f"{inputs.max().item():.3g}")
        rank_zero_info(f"Median/mean/standard deviation of values in input batch: "
                       f"{torch.median(inputs)}, {inputs.mean()}, {torch.std(inputs)}")
        rank_zero_info(f"Shape/dtype of target batch: {targets.shape}, {targets.dtype}")

        if isinstance(trainer.logger, TensorBoardLogger):
            plotter = ImagePlotter(image_size=self.image_size,
                                   transform=trainer.datamodule.normalizer.invert,
                                   colormap=self.colormap,
                                   tight_layout=True,
                                   writer=trainer.logger.experiment)
            plotter.image_grid(inputs[:self.n_cols * self.n_rows],
                               title="Training dataset examples",
                               n_rows=self.n_rows,
                               n_cols=self.n_cols,
                               epoch_ind=0,
                               suppress_plots=True)
        
        n_train_steps = trainer.estimated_stepping_batches
        rank_zero_info(f"\nTotal training steps: {n_train_steps:,d}")
        rank_zero_info("===================================================================\n")


class HyperParamLogger(Callback):
    """Logs metrics of finished model for HParams view in Tensorboard.
    
    This hasn't been tested all that much and could likely be streamlined.
    """
    @override
    def on_fit_start(self,
                     trainer: Trainer,
                     pl_module: LightningModule):
        """Called when fit starts. Checks if `trainer` has a valid logger.
        
        Raises:
            ValueError: If `trainer.logger` is not a `TensorBoadLogger`.
        """
        if not isinstance(trainer.logger, TensorBoardLogger):
            raise ValueError(f"HyperParamLogger only works for TensorBoardLogger, but got "
                             f"{trainer.logger}")

    @override
    def on_fit_end(self,
                   trainer: Trainer,
                   pl_module: LightningModule):
        trainer.logger.log_hyperparams(params=pl_module.hparams,
                                       metrics={f"final_{name}": float(value)
                                                for name, value in trainer.callback_metrics.items()
                                                if value.ndim == 0})
