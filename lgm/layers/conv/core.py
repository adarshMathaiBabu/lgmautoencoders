"""Core computations such as convolutions or pooling."""
from __future__ import annotations

from typing import override

import torch
from torch import nn

from lgm.ops import spatial_broadcast_index
from lgm.types import DataBatchFloat, DataFormat


INTERPOLATION_MAP = {"1d": "linear",
                     "2d": "bilinear",
                     "3d": "trilinear"}

DEFAULT_NORM = "batch"


class DWSConv(nn.Module):
    """Depthwise-separable convolution.

    This implements a spatial convolution with group size set to number of input channels, i.e. each
    channel is processed separately by one filter. Then, a 1x1 convolution is applied to mix
    channels. NOTE that there is no normalization/activation after the spatial conv; that is, this
    is a "linear separation" of the full convolution.
    """
    @override
    def __init__(self,
                 data_format: DataFormat,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: str | int = 0,
                 bias: bool = True):
        """Create submodules for the component convolutions.

        Parameters:
            data_format: One of `'1d'`, `'2d'`, or `'3d'`.
            Other arguments: See e.g. `nn.Conv2d`.
        """
        super().__init__()
        self.depthwise_conv = conv_nd(data_format)(in_channels=in_channels,
                                                   out_channels=in_channels,
                                                   kernel_size=kernel_size,
                                                   stride=stride,
                                                   padding=padding,
                                                   groups=in_channels)
        self.channel_conv = conv_nd(data_format)(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=1,
                                                 bias=bias)

    @override
    def forward(self,
                inputs: torch.Tensor) -> torch.Tensor:
        return self.channel_conv(self.depthwise_conv(inputs))


class DWSConv1d(DWSConv):
    @override
    def __init__(self,
                 **kwargs):
        """Create `DWSConv` with data_format forced to `1d`.
        
        Parameters:
            kwargs: Keyword arguments for `DWSConv`.
        """
        super().__init__(data_format="1d", **kwargs)


class DWSConv2d(DWSConv):
    @override
    def __init__(self,
                 **kwargs):
        """Create `DWSConv` with data_format forced to `2d`.
        
        Parameters:
            kwargs: Keyword arguments for `DWSConv`.
        """
        super().__init__(data_format="2d", **kwargs)


class DWSConv3d(DWSConv):
    @override
    def __init__(self,
                 **kwargs):
        """Create `DWSConv` with data_format forced to `3d`.
        
        Parameters:
            kwargs: Keyword arguments for `DWSConv`.
        """
        super().__init__(data_format="3d", **kwargs)


class SqueezeExcite(nn.Module):
    """Squeeze-and-excite module.
    
    Used as a component in the CNN block functions.
    """
    @override
    def __init__(self,
                 input_dim: int,
                 squeeze_dim: int,
                 activation: type[nn.Module],
                 bias: float):
        """Configure the module and create components.

        Parameters:
            input_dim: Number of channels in the input.
            squeeze_dim: Dimension of the bottleneck.
            activation: Hidden activation to use.
            bias: Bias of the last layer is initialized to this value. Since there is a sigmoid
                function after, starting with a positive bias will result in sigmoid outputs being
                closer to 1 at the start.
        """
        super().__init__()
        self.excitor = nn.Sequential(nn.Linear(input_dim, squeeze_dim),
                                     activation(),
                                     nn.Linear(squeeze_dim, input_dim),
                                     nn.Sigmoid())
        nn.init.constant_(self.excitor[-2].bias, bias)

    @override
    def forward(self,
                inputs: DataBatchFloat) -> DataBatchFloat:
        spatial_dims = tuple(range(2, inputs.dim()))
        squeezed = inputs.mean(spatial_dims)
        excitation = self.excitor(squeezed)[spatial_broadcast_index(inputs)]
        return inputs * excitation
    

class UpsampleAA(nn.Upsample):
    """Standard upsampling layer, but forcing `antialias=True`.
    
    I don't think this does anything for upsampling... but whatever.
    """
    @override
    def forward(self,
                inputs: DataBatchFloat) -> DataBatchFloat:
        return torch.nn.functional.interpolate(
            inputs,
            self.size,
            self.scale_factor,
            self.mode,
            self.align_corners,
            recompute_scale_factor=self.recompute_scale_factor,
            antialias=self.mode in ["bilinear", "bicubic"])


class DownsampleAA(UpsampleAA):
    """Bilinear downsampling layer with anti-aliasing.
    
    We abuse the fact that upsampling layers just use `interpolate`, which can up- OR downsample. It
    also doesn't check whether the `scale_factor` is `> 1`. So we can use an upsampling layer for
    downsampling. Yay!
    # TODO possibly protect against floating point inaccuracy when dividing here.
    """
    @override
    def __init__(self,
                 scale_factor: float,
                 **kwargs):
        super().__init__(scale_factor=1/scale_factor, **kwargs)


def conv_nd(data_format: DataFormat) -> type[nn.Module]:
    """Simple wrapper to return a `Conv` module for the desired dimensionality.
    
    Parameters:
        data_format: One of `'1d'`, `'2d'`, `'3d'`.

    Returns:
        `Convolution` module type.

    Raises:
        ValueError: For unsupported `data_format`.
    """
    if data_format == "1d":
        return nn.Conv1d
    elif data_format == "2d":
        return nn.Conv2d
    elif data_format == "3d":
        return nn.Conv3d
    else:
        raise ValueError(f"Unsupported data format {data_format}. Supported are '1d', '2d', '3d'.")


def dwsconv_nd(data_format: str) -> type[nn.Module]:
    """Like `conv_nd` for depthwise conv.

    Seem superfluous with the `DWSConv` class, but if we want to have named classes per
    dimensionality AND have this kind of wrapper, then we need it.

    Parameters:
        data_format: One of `'1d'`, `'2d'`, or `'3d'`.

    Returns:
        `DWSConv` module type.

    Raises:
        ValueError: For unsupported `data_format`.
    """
    if data_format == "1d":
        return DWSConv1d
    elif data_format == "2d":
        return DWSConv2d
    elif data_format == "3d":
        return DWSConv3d
    else:
        raise ValueError(f"Unsupported data format {data_format}. Supported are '1d', '2d', '3d'.")


def dropout_nd(data_format: DataFormat) -> type[nn.Module]:
    """Simple wrapper to return a Dropout module for the desired dimensionality.
    
    Parameters:
        data_format: One of `'1d'`, `'2d'`, `'3d'`.

    Returns:
        `Dropout` module type.

    Raises:
        ValueError: For unsupported `data_format`.
    """
    if data_format == "1d":
        return nn.Dropout1d
    elif data_format == "2d":
        return nn.Dropout2d
    elif data_format == "3d":
        return nn.Dropout3d
    else:
        raise ValueError(f"Unsupported data format {data_format}. Supported are '1d', '2d', '3d'.")


def adaptive_avg_pool_nd(data_format: DataFormat) -> type[nn.Module]:
    """Simple wrapper to return an `AdaptiveAvgPool` module for the desired dimensionality.
    
    Parameters:
        data_format: One of `'1d'`, `'2d'`, `'3d'`.

    Returns:
        `AdaptiveAvgPool` module type.

    Raises:
        ValueError: For unsupported `data_format`.
    """
    if data_format == "1d":
        return nn.AdaptiveAvgPool1d
    elif data_format == "2d":
        return nn.AdaptiveAvgPool2d
    elif data_format == "3d":
        return nn.AdaptiveAvgPool3d
    else:
        raise ValueError(f"Unsupported data format {data_format}. Supported are '1d', '2d', '3d'.")
