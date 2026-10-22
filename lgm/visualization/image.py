"""Image-related visualization utilities."""
from collections.abc import Iterable
from pathlib import Path
from PIL import Image
from typing import Literal

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from lgm.types import ImageBatchFloat, ImageTransform, Number


class ImagePlotter:
    def __init__(self,
                 image_size: tuple[Number, Number] | Number = 1.5,
                 transform: ImageTransform | None = None,
                 colormap: str = "Greys",
                 tight_layout: bool = True,
                 writer: SummaryWriter | None = None):
        """Configure the plotter.

        Note that many of these arguments only affect the `image_grid()` method. We bundle
        `make_animation` mainly since it also needs the `transform` to invert normalized data.
        
        Parameters:
            image_size: The size of one image as `(width, height)`. The total size of the figure
                will be `(width * n_cols, height * n_rows)` given `n_cols, n_rows` for a specific
                plot. If passed a single number, will use this as width and height.  
                Only used for `image_grid()`; `make_animation` uses the full number of pixels in the
                images. 
            transform: If you have supplied additional transforms that change the scale of the data,
                you likely want to undo that (scale back to `[0, 1]`) before plotting. This function
                should do that. Often the `invert` method of a `data_module.normalizer`.
            colormap: Which colormap to use to display images in `image_grid()`. Only used for
                single-channel images.
            tight_layout: If True, `image_grid()`use `plt.tight_layout()`.
            writer: Optional Tensorboard writer. If given, `image_grid()` writes to Tensorboard
                using the title as tag.  
                NOTE if this is not `None`, `image_grid()` must *always* receive `epoch_ind`!
        """
        self.image_size = image_size
        self.transform = transform
        self.colormap = colormap
        self.tight_layout = tight_layout
        self.writer = writer

    def image_grid(self,
                   images: ImageBatchFloat,
                   title: str,
                   n_rows: int,
                   n_cols: int | None = None,
                   subtitles: Iterable[str] | None = None,
                   epoch_ind: int | None = None,
                   suppress_plots: bool = False) -> plt.Figure:
        """Display a rectangular grid of images.

        Optionally also writes to Tensorboard if a writer is supplied.

        Parameters:
            images: Batch of images we want to plot. We assume that this has `n_rows * n_cols` many
                entries. Fewer are okay, but will result in parts of the grid being empty. More will
                crash.
            title: Will be used as figure title as well as for naming Tensorboard summaries.
            n_rows: Will plot this many rows, and `n_rows**2` many examples in total if `n_cols` is
                not given.
            n_cols: Will plot this many columns of examples. Defaults to `n_rows`.
            subtitles: If given, should be an iterable of strings of the same length as `images`.
                Each string will be used as title for the respective image's subplot.
            epoch_ind: MUST be given if the instance has a `writer`. Used to supply the time step to
                save the figure to.
            suppress_plots: If `True`, the plot is not shown and also NOT closed, unless `writer` is
                also given. Presumably, you want to do something else with the plot, instead. So
                make sure you are closing the plot later!

        Returns:
            The figure object.

        Raises:
            ValueError: If there are more images than grid cells, or if `subtitles` was given, but
                not enough were provided for the given number of images.
        """
        if n_cols is None:
            n_cols = n_rows

        if images.shape[0] > n_rows * n_cols:
            raise ValueError(f"Got {images.shape[0]} images, but only {n_rows * n_cols} cells in "
                             "the grid.")
        if subtitles is not None and len(subtitles) < images.shape[0]:
            raise ValueError(f"Got {images.shape[0]} images, but only {len(subtitles)} subtitles.")

        images = self.to_display(images)
        figure_size = (self.image_size * n_cols, self.image_size * n_rows)

        figure = plt.figure(figsize=figure_size)
        for ind, image in enumerate(images):
            plt.subplot(n_rows, n_cols, ind + 1)
            plt.imshow(image, vmin=0, vmax=1, cmap=self.colormap)
            plt.axis("off")
            if subtitles is not None:
                plt.title(subtitles[ind], fontsize=8)
        plt.suptitle(title)
        if self.tight_layout:
            plt.tight_layout()

        if self.writer is not None and epoch_ind is not None:
            self.writer.add_figure(tag=title,
                                   figure=figure,
                                   global_step=epoch_ind,
                                   close=suppress_plots)
        if not suppress_plots:
            plt.show()
            plt.close()
        return figure

    def make_animation(self,
                       save_path: str | Path,
                       image_series: ImageBatchFloat,
                       frame_rate: int = 10,
                       format: Literal["gif", "webp"] = "webp",
                       invert_colors: bool = False):
        """Create a GIF/WEBP from a batch of images.

        For color images WEBP is preferred, as the GIF format is very limited with respect to how
        many colors it can display.
        
        Parameters:
            save_path: Path where the result should be stored, WITHOUT extension. Actual saved file
                will have `.gif` or `.webp` extension, depending on `format` argument.
            image_series: Batched image tensor. Each row will become one frame.
            frame_rate: Number of images per second.
            format: One of `"gif"` or `"webp"`. the format to save the animation as.
            invert_colors: If `True`, invert all brightness values, i.e. use 1-image. Can be useful
                for MNIST or FashionMNIST, where representing 1s as black and 0s as white leads to
                more natural-looking images.

        Raises:
            ValueError: If given `format` is invalid.
        """
        if format not in ["gif", "webp"]:
            raise ValueError(f"Invalid format {format}. Allowed are 'gif', 'web'")

        image_series = self.to_display(image_series)

        frame_duration = 1000 // frame_rate
        save_path = Path(save_path).with_suffix(f".{format}")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if image_series.shape[-1] == 1:  # grayscale image
            image_series = image_series.repeat(3, axis=-1)
        if invert_colors:
            image_series = 1 - image_series
        image_series = (image_series * 255).astype(np.uint8)

        pil_images = [Image.fromarray(image, mode="RGB") for image in image_series]
        pil_images[0].save(save_path,
                           save_all=True,
                           append_images=pil_images[1:],
                           duration=frame_duration,
                           loop=0)
        print(f"{format} saved to {save_path}!")

    def to_display(self,
                   images: ImageBatchFloat) -> np.ndarray:
        """Makes images 'ready to be displayed'.
        
        Specifically:
        - Convert to [0, 1] scaling.
        - Clip to [0, 1], since many generative models do not respect borders.
        - Convert to numpy array.
        - Transpose to `channels_last`
        """
        with torch.inference_mode():
            if self.transform is not None:
                images = self.transform(images)
            return np.clip(images.cpu().numpy(), 0, 1).transpose(0, 2, 3, 1)
