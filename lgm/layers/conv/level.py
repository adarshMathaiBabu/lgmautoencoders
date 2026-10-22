"""CNN "level" implementation. See `CNNLevel` class docs for more information."""
from __future__ import annotations

from typing import override

from torch import nn

from lgm.types import DataBatchFloat
from .blocks import BlockConfig, ConvBlockPrenorm
from .core import INTERPOLATION_MAP, UpsampleAA


class CNNLevel(nn.Module):
    """One level of a CNN consists of a number of convolutional blocks as well as up-/down-sampling.
    
    All blocks at a level output the same image size and number of filters.
    """
    @override
    def __init__(self,
                 n_blocks: int,
                 previous_filters: int,
                 current_filters: int,
                 stride: int,
                 direction: str,
                 block_config: BlockConfig,
                 use_attention: bool = False):
        """Configure the level and create component blocks.
        
        Parameters:
            n_blocks: How many blocks the level should consist of.
            previous_filters: Number of channels of the input to this level.
            current_filters: Number of channels on this level.
            stride: Up- or down-sampling factor applied by the first block. NOTE: In case of
                downsampling, the stride is applied at the start of the first block. In case of
                upsampling, bilinear upsampling is added at the *end* of the level.
            direction: One of `"up"` or `"down"`. `"down"` would usually be for encoder-type models,
                `"up"` for decoder/generator-like.
            block_config: `BlockConfig` object to config the blocks.
            use_attention: Whether to insert attention blocks after the residuals at this level.
        """
        super().__init__()
        self.blocks = nn.ModuleList()
        for layer_ind in range(n_blocks):
            self.blocks.append(ConvBlockPrenorm(
                previous_filters=previous_filters if layer_ind == 0 else current_filters,
                current_filters=current_filters,
                config=block_config,
                stride=stride if (layer_ind == 0 and direction == "down") else 1,
                use_attention=use_attention))
        if direction == "up" and stride > 1:
            self.upsample = UpsampleAA(scale_factor=stride,
                                       mode=INTERPOLATION_MAP[block_config.data_format])
        else:
            self.upsample = None

    @override
    def forward(self,
                inputs: DataBatchFloat) -> DataBatchFloat:
        """Compute level outputs.
        
        Parameters:
            inputs: Input features for this level.

        Returns:
            The final output features.
        """
        outputs = inputs
        for block in self.blocks:
            outputs = block(outputs)

        if self.upsample is not None:
            outputs = self.upsample(outputs)
        return outputs
