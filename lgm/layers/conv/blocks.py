"""CNN "block" implementation. See implemented class docs for more information."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, override

import torch
from torch import nn

from lgm.layers.generic import AttentionBlock
from lgm.layers.norms import n_groups_convert, norm_selector
from lgm.types import DataBatchFloat, DataFormat, ScalarFloat
from .core import (DEFAULT_NORM, INTERPOLATION_MAP, DownsampleAA, SqueezeExcite, conv_nd,
                   dropout_nd, dwsconv_nd)


class ConvBlockPrenorm(nn.Module):
    """Multi-purpose CNN block.
    
    Implements an arbitrary number of Normalization-Activation-Convolution operations. Optional
    features include a residual connection, squeeze-excite, and bottlenecks using 1x1 convolutions.
    For example, the blocks proposed in the original ResNet paper could be implemented by either
    - `n_layers=2` and `bottleneck_factor=1` (default block)
    - `n_layers=3` and `bottleneck_factor=4` (bottleneck block)
    """
    residual_factor: ScalarFloat

    @override
    def __init__(self,
                 previous_filters: int,
                 current_filters: int,
                 config: BlockConfig,
                 stride: int = 1,
                 use_attention: bool = False):
        """Configure the block and create sublayers.

        Parameters:
            previous_filters: Number of channels in input.
            current_filters: Output channels.
            config: `BlockConfig` object to config the block.
            stride: Stride applied by the first convolution.
            use_attention: If `True`, the residual block will be followed by an attention block.

        Raises:
            ValueError: For invalid `data_format`.
        """
        super().__init__()
        self.config = config

        data_format = config.data_format
        if data_format not in ["1d", "2d", "3d"]:
            raise ValueError(f"Invalid data format {data_format}. Allowed are '1d', '2d', '3d'.")
        conv_fn = dwsconv_nd(data_format) if config.depthwise else conv_nd(data_format)
        default_conv = conv_nd(data_format)
        norm_fn = norm_selector(config.norm, data_format)

        use_bias = config.norm == "dummy"
        input_dim = previous_filters

        def first_layer(filters_: int,
                        kernel_size_: int) -> nn.Sequential:
            # helper to reduce code duplication
            conv_fn_ = default_conv if kernel_size_ == 1 else conv_fn
            norm_groups_ = n_groups_convert(config.norm_groups, input_dim)
            layer = nn.Sequential(norm_fn(norm_groups_, input_dim),
                                  config.activation(),
                                  conv_fn_(in_channels=input_dim,
                                           out_channels=filters_,
                                           kernel_size=kernel_size_,
                                           bias=use_bias, padding="same"))
            if stride > 1:
                layer.append(DownsampleAA(scale_factor=stride,
                                          mode=INTERPOLATION_MAP[data_format]))
            return layer
        
        if config.n_layers < 1:
            raise ValueError("Must have at least one layer!")
        elif config.n_layers == 1:
            if config.bottleneck_factor > 1:
                print("WARNING! You have created a single-layer block with bottleneck. The "
                      "bottleneck will be ignored.")
            self.layers = first_layer(current_filters, config.kernel_size)
        else:
            bottleneck_filters = current_filters // config.bottleneck_factor
            use_bottleneck = config.bottleneck_factor > 1
            bottleneck_kernel_size = 1 if use_bottleneck else config.kernel_size
            norm_groups = n_groups_convert(config.norm_groups, bottleneck_filters)
            if use_bottleneck and config.n_layers == 2:
                print("WARNING! You have created a bottleneck block with only two layers. "
                      "This block consists purely of 1x1 convolutions!")

            self.layers = first_layer(bottleneck_filters, bottleneck_kernel_size)
            for _ in range(config.n_layers - 2):
                self.layers.append(norm_fn(norm_groups, bottleneck_filters))
                self.layers.append(config.activation())
                self.layers.append(conv_fn(in_channels=bottleneck_filters,
                                           out_channels=bottleneck_filters,
                                           kernel_size=config.kernel_size,
                                           bias=use_bias, padding="same"))

            self.layers.append(norm_fn(norm_groups, bottleneck_filters))
            self.layers.append(config.activation())
            last_conv = default_conv if bottleneck_kernel_size == 1 else conv_fn
            self.layers.append(last_conv(in_channels=bottleneck_filters,
                                         out_channels=current_filters,
                                         kernel_size=bottleneck_kernel_size,
                                         bias=use_bias, padding="same"))

        if config.use_residual:
            if config.force_1x1 or previous_filters != current_filters or stride > 1:
                shortcut_layer = default_conv(in_channels=previous_filters,
                                              out_channels=current_filters,
                                              kernel_size=1,
                                              bias=use_bias)
                if stride > 1:
                    self.shortcut = nn.Sequential(shortcut_layer,
                                                  DownsampleAA(scale_factor=stride,
                                                               mode=INTERPOLATION_MAP[data_format]))
                else:
                    self.shortcut = shortcut_layer
            else:
                self.shortcut = nn.Identity()
        self.use_residual = config.use_residual

        if config.use_se:
            self.squeeze_excite = SqueezeExcite(current_filters,
                                                current_filters // config.se_reduction_factor,
                                                config.activation, config.se_bias)
        else:
            self.squeeze_excite = nn.Identity()

        if use_attention:
            # TODO vary number of heads
            norm_groups = n_groups_convert(config.norm_groups, bottleneck_filters)
            self.attention = AttentionBlock(current_filters,
                                            num_heads=8,
                                            num_groups=norm_groups,
                                            rescale_residual=config.rescale_residual)
        else:
            self.attention = nn.Identity()

        if config.dropout > 0.:
            self.dropout = dropout_nd(data_format)(p=config.dropout)
        else:
            self.dropout = nn.Identity()

        if config.rescale_residual:
            self.register_buffer("residual_factor", 1/torch.sqrt(torch.tensor(2.)))
        else:
            self.register_buffer("residual_factor", torch.tensor(1.))

        if config.zero_init and config.use_residual:
            final_layer = self.layers[-1]
            if isinstance(self.layers[-1], dwsconv_nd(data_format)):
                final_layer = final_layer.channel_conv
            nn.init.zeros_(final_layer.weight)
            if use_bias:
                nn.init.zeros_(final_layer.bias)

    @override
    def forward(self,
                inputs: DataBatchFloat) -> DataBatchFloat:
        """Apply the block.
        
        This computes features, adds them to the input if residual connection is desired, and
        optionally applies attention as well as residual rescaling.

        Parameters:
            inputs: Input features to the block.

        Returns:
            Result of the block.
        """
        # TODO cleaner to separate attention away from residual block
        features = self.residual_se(inputs)
        if self.use_residual:
            return self.attention(self.residual_factor * (features + self.shortcut(inputs)))
        else:
            return self.attention(features)

    def residual_se(self,
                    inputs: DataBatchFloat) -> DataBatchFloat:
        """Computes features for `forward()`.
        
        Parameters:
            See: `forward()` method.

        Returns:
            Transformed features.
        """
        features = inputs
        for layer in self.layers:
            features = layer(features)
        features = self.squeeze_excite(features)
        features = self.dropout(features)
        return features


@dataclass(frozen=True)
class BlockConfig:
    """Helper object to bundle parameters for `ConvBlockPrenorm`.
    
    These are parameters that, in general, all blocks in a model will share, and that are (rarely)
    relevant) to the classes building the blocks. In the past, I passed those arguments through via
    **kwargs. However, `LightningCLI` does not like `**kwargs` at all, as these make it impossible
    to validate the config. This would require duplicating all the arguments for the higher classes.
    To prevent this, use this little dataclass.

    NOTE in general, I would prefer to have a "base" config, and then several more specialized
    configs that cover common use cases. However, this does not mesh well with `LightningCLI`, as
    when you annotate a variable as a `dataclass`, inheritance is pretty much impossible. For this
    reason, we have here a default block that will serve you well for the vast majority of use
    cases: Standard residual block with two layers, kernel size of 3, Mish activation, and
    squeeze-and-excite. Some things you might want to change:
    - For "bottleneck" residual blocks, use `n_layers=3` and `bottleneck_factor=4`.
    - For other domains like audio, you will likely want a larger `kernel_size` for 1D convolutions,
      e.g.` kernel_size=7`.
    - `depthwise=True` can lead to a much more efficient architecture.

    Parameters:
        data_format: One of `"1d"`, `"2d"` or `"3d"`. Determines the convolution used. `2d` is for
            images, `1d` for time series, e.g. audio, `3d `for volumes or videos (we don't really do
            that here).
        n_layers: Number of Norm-Act-Conv layers.
        kernel_size: Size of the convolutional kernels (that are not `1x1` by default).
        activation: Activation function. Must be a type, i.e. callable as a constructor, *not* an
            instantiated module.
        bottleneck_factor: If `> 1`, first and last convolutions will be `1x1` convolutions. The
            first one will map to channel number `current_filters // bottleneck_factor`. All
            "middle" convolutions will operate at this channel number. The final convolution will
            map back to `current_filters`.
        depthwise: If True, use depthwise convolution. Note that `1x1` convs will use regular
            convolution.
        norm: Desired normalization. See `layers.norms.norm_selector` for available options.
        norm_groups: If `norm` is `GroupNorm`, this needs to be passed. If this is a positive
            number, use this many groups for `GroupNorm`. If this is a negative number, we will use
            this as *group size* instead.
        use_residual: If `True`, input is added to output of the block.
        force_1x1: If `True`, input is always transformed by a `1x1` convolution before being added
            in the residual connection. Otherwise, a `1x1` convolution is only used if necessary
            (`stride > 1`, or channel numbers differ). Ignored if `use_residual` is `False`.
        rescale_residual: If `True`, all residual connections are multiplied by `1/sqrt(2)`. This
            has been reported to improve learning by preventing activations from growing out of
            bounds through repeatedly adding residuals to the inputs. However, it can also lead to
            the original input "decaying" as it goes deeper through the network. Your mileage may
            vary.  
            **Do not mix** with `zero_init`. This leads to a pathological initialization where
            outputs are decayed by rescaling, and nothing is added due to the zero-init. For
            safety's sake, we crash if both arguments are True.
        zero_init: If `True`, the final convolutional layer in the block is initialized with 0
            weights and biases. This causes the model to be as close as possible to an identity
            function at the start of training if the block is residual. This argument is ignored if
            `use_residual` is `False`.  
            See note for `rescale_residual`!
        use_se: If `True`, apply a squeeze-excite operation at the end of the block (in the residual
            part, if that is used).
        se_reduction_factor: The hidden layer in the SE "mini network" will use
            `current_filters // reduction units`.
        se_bias: Pass a number `> 0` to bias the final SE layer (sigmoid output) towards 1, to let
            through more activation early during training.
        dropout: Chance to randomly drop channels during training. Dropout is applied at the very
            end of each residual block, before the addition. NOT recommended if you are using
            `BatchNorm`, but should be ok to combine with other normalizations (e.g. `GroupNorm`).
    """
    data_format: DataFormat = "2d"
    n_layers: int = 2
    kernel_size: int = 3
    activation: type[nn.Module] = nn.Mish
    bottleneck_factor: int = 1
    depthwise: bool = False

    norm: Literal["batch", "group", "layer", "instance", "dummy"] = DEFAULT_NORM
    norm_groups: int | None = None

    use_residual: bool = True
    force_1x1: bool = False
    rescale_residual: bool = False
    zero_init: bool = False

    use_se: bool = True
    se_reduction_factor: int | None = 8
    se_bias: float = 2.

    dropout: float = 0.

    @override
    def __post_init__(self):
        """Validate arguments.
        
        Raises:
            ValueError: If `use_residual`, `rescale_residual` and `zero_init` are all `True`.
        """
        if self.use_residual and self.rescale_residual and self.zero_init:
            raise ValueError("Residual block detected with both rescaling and zero_init. These "
                             "arguments do not interact well and are not supposed to be used "
                             "together. Please choose one out of rescale_residual or zero_init.")
