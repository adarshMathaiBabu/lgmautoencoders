"""Callbacks related to visually judging generative model training progress.

See parent module docs (`lgm.callbacks`) for usage recommendations.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar, override

import torch
from lightning import Callback, LightningModule, Trainer
from matplotlib import pyplot as plt

from lgm.models.autoencoder import AETrainingModule
from lgm.ops import interleave
from lgm.types import Number
from lgm.visualization import ImagePlotter


class ImageGrid(Callback):
    """Base class for callbacks regularly saving plots of image grids."""
    name: ClassVar[str] = ""

    @override
    def __init__(self,
                 config: GridConfig):
        """Configure the callback.

        Note that the plotter object is not configured here, as this needs access to the `Trainer`
        object, which is not available here. The object is configured `on_fit_start`.

        Parameters:
            config: `GridConfig` instance.
        """
        super().__init__()
        self.every_n_epochs = config.every_n_epochs
        self.n_rows = config.n_rows
        self.image_size = config.image_size
        self.colormap = config.colormap
        self.plotter = None

    def save_figure(self,
                    figure: plt.Figure,
                    trainer: Trainer,
                    epoch: int):
        """Save the given figure to the path.
        
        Parameters:
            figure: The figure to save.
            trainer: Lightning `Trainer` object to get the log directory from.
            epoch: Index to include in the file name.
        """
        save_dir = os.path.join(trainer.log_dir, self.name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"epoch_{epoch:04d}.png")
        figure.savefig(save_path)
        plt.close(figure)

    @override
    def on_fit_start(self,
                     trainer: Trainer,
                     pl_module: LightningModule):
        self.plotter = ImagePlotter(image_size=self.image_size,
                                    transform=trainer.datamodule.normalizer.invert,
                                    colormap=self.colormap,
                                    tight_layout=True,
                                    writer=trainer.logger.experiment)


class ReconstructionLogging(ImageGrid):
    """Regularly creates plots showing reconstructed image examples.
    
    Intended for autoencoders only!  
    Figures are stored in Tensorboard as well as saved directly to disk (log directory) for
    convenience.
    """
    name: ClassVar[str] = "reconstructions"

    @override
    def __init__(self,
                 config: GridConfig,
                 conditional: bool = False):
        """Configure the callback.

        Parameters:
            config: `GridConfig` instance. Note that `n_rows` must be even!
            conditional: If `True`, assumes conditional model, meaning that labels will be handed
                to the model call!

        Raises:
            ValueError: If `grid_config.n_rows` is not even.
        """
        if config.n_rows % 2:
            raise ValueError(f"n_rows must be even, but got {config.n_rows}.")
        super().__init__(config)
        self.conditional = conditional
        self.examples = None

    @override
    def on_train_epoch_start(self,
                             trainer: Trainer,
                             pl_module: AETrainingModule):
        """Called at the start of each epoch. Compute reconstructions and store them.
        
        We use `on_train_epoch_start` because this results in reconstructions plotted before any
        training has happened (before the very first epoch). This can be informative to see any
        architectural biases in the model: Are the reconstructions just blank images? Smooth? Noisy?

        We grab the first batch of the validation dataloader to have some inputs to reconstruct. The
        same batch is used throughout.
        """
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs != 0 or not trainer.is_global_zero:
            return

        if trainer.is_global_zero and self.examples is None:
            batch, labels  = next(iter(trainer.datamodule.val_dataloader()))
            self.examples = batch[:self.n_rows**2 // 2].to(pl_module.device)
            self.labels = labels[:self.n_rows**2 // 2].to(pl_module.device)

        # must make sure to go into eval mode and then back to training if we were in there
        was_training = pl_module.training
        pl_module.eval()
        with torch.inference_mode():
            if self.conditional:
                reconstructions = pl_module(self.examples, self.labels, map_to_expected=True)
            else:
                reconstructions = pl_module(self.examples, map_to_expected=True)
        if was_training:
            pl_module.train()

        alternating = interleave(self.examples, reconstructions)
        subtitles = ["Original", "Reconstruction"] * (self.n_rows // 2)
        subtitles = subtitles + [""] * (self.n_rows**2 - self.n_rows)
        figure = self.plotter.image_grid(alternating,
                                         title="Reconstruction Examples",
                                         subtitles=subtitles,
                                         n_rows=self.n_rows,
                                         epoch_ind=epoch,
                                         suppress_plots=True)
        self.save_figure(figure, trainer, epoch)


@dataclass(frozen=True)
class GridConfig:
    """Bundles parameters for `ImageGrid` callbacks.
    
    Mainly to avoid argument duplication in `__init__`.
    
    Parameters:
        every_n_epochs: Plots are created every this many epochs.
        n_rows: Will create a square grid with this many rows and columns.
        image_size: Figure will alot this size (inches) per image.
        colormap: Which colormap to use; only relevant for greyscale images.
    """
    every_n_epochs: int
    image_size: tuple[Number, Number] | Number = 1.5
    n_rows: int = 10
    colormap: str = "Greys"
