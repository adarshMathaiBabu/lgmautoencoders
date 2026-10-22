"""Layers that don't fit anywhere else."""
from typing import override

import torch
from torch import nn

from lgm.ops import channels_first_to_last, channels_last_to_first
from lgm.types import DataBatchFloat, ScalarFloat, TabularBatchFloat, VectorBatchFloat


class OneHot(nn.Module):
    """Just a module wrapper around the one-hot function."""
    @override
    def __init__(self,
                 n_classes: int):
        """Configure the module.
        
        Parameters:
            n_classes: How many dimensions the one-hot vectors should have.
        """
        super().__init__()
        self.n_classes = n_classes

    @override
    def forward(self,
                inputs: VectorBatchFloat) -> TabularBatchFloat:
        return nn.functional.one_hot(inputs, self.n_classes).to(torch.float32)
    

class AttentionBlock(nn.Module):
    """Minimal multi-head self-attention block to be used in U-Nets for Diffusion etc.
            
    NOT a Transformer layer. Only performs (group) normalization, key/query/value projection,
    multi-head attention, output projection and residual connection. There is no feed-forward layer,
    and no positional encodings are used.
    """
    residual_factor: ScalarFloat

    @override
    def __init__(self,
                 n_channels: int,
                 num_heads: int,
                 num_groups: int,
                 rescale_residual: bool = False):
        """Configure the module and create component layers.

        Parameters:
            n_channels: Number of channels in the input.
            num_heads: How many heads to use for self-attention. Each head will have dimensionality
                `n_channels // num_heads`.
            num_groups: Number of groups for `GroupNorm`. If `None`, will use `LayerNorm`.
                TODO allow `LayerNorm` in any case if desired.
            rescale_residual: If `True`, residual connections is multiplied by `1/sqrt(2)` (after
                addition). 
        """
        super().__init__()
        if num_groups is None:
            self.norm = nn.LayerNorm(n_channels)
        else:
            self.norm = nn.GroupNorm(num_groups, n_channels)
        
        # *3 outputs because key, query, value are needed
        self.qkv = nn.Linear(n_channels, 3*n_channels)
        self.mha = nn.MultiheadAttention(n_channels, num_heads, batch_first=True)
        self.out_projection = nn.Linear(n_channels, n_channels)
        
        self.num_heads = num_heads
        if rescale_residual:
            self.register_buffer("residual_factor", 1/torch.sqrt(torch.tensor(2.)))
        else:
            self.register_buffer("residual_factor", torch.tensor(1.))

    @override
    def forward(self,
                inputs: DataBatchFloat) -> DataBatchFloat:
        outputs = inputs
        batch_size, n_channels = inputs.shape[:2]
        spatial_dims = inputs.shape[2:]

        # this sucks but not sure how else to do it -- groupnorm requires channels_first so we have
        # to do it here
        if not isinstance(self.norm, nn.LayerNorm):
            outputs = self.norm(outputs)
        # flatten spatial dimensions and then transpose to channels-last sequence
        outputs = channels_first_to_last(outputs.view(batch_size, n_channels, -1))
        # this sucks, part 2
        if isinstance(self.norm, nn.LayerNorm):
            outputs = self.norm(outputs)

        queries, keys, values = self.qkv(outputs).chunk(3, dim=-1)
        attention, _ = self.mha(queries, keys, values)
        outputs = self.out_projection(attention)

        # back to image shape
        outputs = channels_last_to_first(outputs).view(batch_size, n_channels, *spatial_dims)
        return self.residual_factor * (inputs + outputs)
