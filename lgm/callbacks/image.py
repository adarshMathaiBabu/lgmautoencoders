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
                 config: GridConfig):
        """Configure the callback.

        Parameters:
            config: `GridConfig` instance. Note that `n_rows` must be even!

        Raises:
            ValueError: If `grid_config.n_rows` is not even.
        """
        if config.n_rows % 2:
            raise ValueError(f"n_rows must be even, but got {config.n_rows}.")
        super().__init__(config)
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
            batch, _  = next(iter(trainer.datamodule.val_dataloader()))
            self.examples = batch[:self.n_rows**2 // 2].to(pl_module.device)

        # must make sure to go into eval mode and then back to training if we were in there
        was_training = pl_module.training
        pl_module.eval()
        with torch.inference_mode():
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


class GenerationLogging(ImageGrid):
    """Regularly creates plots showing generated image examples.
    
    The `LightningModule` MUST have a `generate()` function that takes an argument for number of
    samples in first position, and no further mandatory arguments. We currently do not support
    passing additional arguments for optional settings, so make sure the model has reasonable
    default settings.  
    Figures are stored in Tensorboard as well as saved directly to disk (log directory) for
    convenience. 
    """
    name: ClassVar[str] = "generations"

    @override
    def __init__(self,
                 config: GridConfig,
                 seed: int | None = None):
        """Configure the callback.

        Parameters:
            config: `GridConfig` instance.
            seed: If given, always use this seed for generations. This allows to see how generations
                for the same seed proceed over time.
        """
        super().__init__(config)
        self.seed = seed

    @override
    def on_train_epoch_start(self,
                             trainer: Trainer,
                             pl_module: LightningModule):
        """Called at the start of each epoch. Create generations and store them.
        
        We use `on_train_epoch_start` because this results in generations plotted before any
        training has happened. This can be informative to see any architectural biases in the model:
        Are the generations just blank images? Smooth? Noisy?
        """
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs != 0 or not trainer.is_global_zero:
            return

        # must make sure to go into eval mode and then back to training if we were in there
        # also stores seed to restore later
        was_training = pl_module.training
        pl_module.eval()
        n_samples = self.n_rows**2
        if self.seed is not None:
            rng_state = torch.get_rng_state()
            cuda_state = torch.cuda.get_rng_state()
            torch.manual_seed(self.seed)

        with torch.inference_mode():
            generated = pl_module.generate(n_samples=n_samples)
        if self.seed is not None:
            torch.set_rng_state(rng_state)
            torch.cuda.set_rng_state(cuda_state)
        if was_training:
            pl_module.train()

        figure = self.plotter.image_grid(generated,
                                         title="Generation Examples",
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
