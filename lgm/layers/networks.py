"""Network architectures bundling CNN stacks.

General usage and hierarchy is:
- `DownCNN` is a pure CNN stack with downsampling.
- `UpCNN` is a pure CNN stack with upsampling.
- `EncoderNet` is a `DownCNN` with a "head" at the end.
- `DecoderNet` is an `UpCNN` with a "root" at the start.
- `EncoderDecoder` is a stack of `EncoderNet` -> `DecoderNet`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, override

import numpy as np
import torch
from torch import nn

from lgm.types import DataBatchFloat, ScalarFloat
from .conv import (INTERPOLATION_MAP, BlockConfig, BodyConfig, CNNBody, UpsampleAA,
                   adaptive_avg_pool_nd, conv_nd)


class DownCNN(nn.Module):
    """General "encoder-type" CNN with downsampling."""
    @override
    def __init__(self,
                 input_channels: int,
                 body_config: BodyConfig,
                 block_config: BlockConfig):
        """Configure and build the CNN.
        
        Parameters:
            See: `EncoderConfig`.
        """
        super().__init__()
        conv_fn = conv_nd(block_config.data_format)

        self.root = conv_fn(in_channels=input_channels,
                            out_channels=body_config.level_filters[0],
                            kernel_size=block_config.kernel_size,
                            padding="same")
        self.body = CNNBody(direction="down",
                            body_config=body_config,
                            block_config=block_config)

    @override
    def forward(self,
                inputs: DataBatchFloat) -> DataBatchFloat:
        return self.body(self.root(inputs))


class UpCNN(nn.Module):
    """General "decoder-type" CNN with upsampling."""
    @override
    def __init__(self,
                 output_channels: int,
                 body_config: BodyConfig,
                 block_config: BlockConfig):
        """Configure and build the CNN.

        NOTE that the `level_*` arguments in `body_config` must already be in the correct order. You
        most likely want to reverse them from whatever you use for the corresponding `DownCNN`. See
        `EncoderDecoder` for an example.
        
        Parameters:
            take_skips: See `layers.conv.CNNBody`.
            Remaining: See `DecoderConfig`.
        """
        super().__init__()
        conv_fn = conv_nd(block_config.data_format)

        self.body = CNNBody(direction="up",
                            body_config=body_config,
                            block_config=block_config)
        self.head = conv_fn(in_channels=body_config.level_filters[-1],
                            out_channels=output_channels,
                            kernel_size=block_config.kernel_size,
                            padding="same")

    @override
    def forward(self,
                inputs: DataBatchFloat) -> DataBatchFloat:
        return self.head(self.body(inputs))

class EncoderNet(nn.Module):
    """Encoder network with optional conditioning.
            
    This combines a `DownCNN` with a "head" network that is applied afterward and can be used, for
    example, to map to a fixed-size vector.
    """
    @override
    def __init__(self,
                 config: EncoderConfig):
        """Configure and build the network.

        Parameters:
            config: `EncoderConfig` for the network.

        Raises:
            ValueError: For invalid `head_style`.
        """
        super().__init__()
        body_config = config.body_config
        block_config = config.block_config

        self.cnn = DownCNN(input_channels=config.input_shape[0],
                           body_config=config.body_config,
                           block_config=block_config)

        final_filters = body_config.level_filters[-1]
        head_layers = []
        if config.head_style == "fc":
            # we could use LazyLinear and be completely independent of input shape,
            # but this protects against misconfiguration.
            total_stride = int(np.prod(body_config.level_strides))
            final_spatial_shape = np.array(config.input_shape[1:]) // total_stride
            flattened_units = final_filters * int(np.prod(final_spatial_shape))
            head_layers.extend([nn.Flatten(),
                                nn.Linear(in_features=flattened_units,
                                          out_features=config.out_dim)])
        elif config.head_style == "pool-fc":
            head_layers.append(adaptive_avg_pool_nd(block_config.data_format)(1))
            if config.global_dropout > 0.:
                head_layers.append(nn.Dropout(config.global_dropout))
            head_layers.extend([nn.Flatten(),
                                nn.Linear(in_features=final_filters,
                                          out_features=config.out_dim)])
        elif config.head_style == "conv-flat":
            conv_fn = conv_nd(block_config.data_format)
            head_layers.extend([conv_fn(in_channels=final_filters,
                                        out_channels=config.out_dim,
                                        kernel_size=1),
                                nn.Flatten()])
        elif config.head_style == "conv":
            conv_fn = conv_nd(block_config.data_format)
            head_layers.extend([conv_fn(in_channels=final_filters,
                                        out_channels=config.out_dim,
                                        kernel_size=1)])
        else:
            raise ValueError(f"Invalid head_style {config.head_style}. Allowed are 'fc', 'pool-fc', "
                             "'conv-flat', 'conv'.")
        self.head_style = config.head_style
        self.head = nn.Sequential(*head_layers)

    @override
    def forward(self,
                inputs: DataBatchFloat) -> DataBatchFloat:
       hidden_out = self.cnn(inputs)
       return self.head(hidden_out)
       

class DecoderNet(nn.Module):
    """Decoder network with optional conditioning.
            
    This combines an `UpCNN` with a "root" network that is applied beforehand and can be used, for
    example, to map a fixed-size vector to the correct input shape for the CNN.
    """
    @override
    def __init__(self,
                 config: DecoderConfig):
        """Configure and build the network.

        Parameters:
            config: `DecoderConfig` for the network.

        Raises:
            ValueError: For invalid `root_style`.
        """
        super().__init__()
        body_config = config.body_config
        block_config = config.block_config

        self.cnn = UpCNN(output_channels=config.input_shape[0],
                         body_config=body_config,
                         block_config=block_config)

        down_channels = body_config.level_filters[0]
        total_stride = int(np.prod(body_config.level_strides))
        # ugh... unflatten really needs python ints, numpy ints are not acceptable
        down_spatial_shape = tuple(int(val) for val in np.array(config.input_shape[1:]) // total_stride)
        if config.root_style == "fc":
            self.root = nn.Sequential(
                nn.Linear(in_features=config.latent_dim,
                          out_features=down_channels * int(np.prod(down_spatial_shape))),
                nn.Unflatten(1, (down_channels, *down_spatial_shape)))
            self.latent_units = config.latent_dim
        elif config.root_style == "conv-flat":
            conv_fn = conv_nd(block_config.data_format)
            self.root = nn.Sequential(nn.Unflatten(1, (config.latent_dim, *down_spatial_shape)),
                                      conv_fn(in_channels=config.latent_dim,
                                              out_channels=down_channels,
                                              kernel_size=1))
            self.latent_units = config.latent_dim * int(np.prod(down_spatial_shape))
        elif config.root_style == "conv":
            conv_fn = conv_nd(block_config.data_format)
            self.root = nn.Sequential(conv_fn(in_channels=config.latent_dim,
                                              out_channels=down_channels,
                                              kernel_size=1))
        else:
            raise ValueError(f"Invalid root_style {config.root_style}. Allowed are 'fc', 'conv'.")

    @override
    def forward(self,
                inputs: DataBatchFloat) -> DataBatchFloat:
        return self.cnn(self.root(inputs))


class EncoderDecoder(nn.Module):
    """Combines an encoder and decoder with mirrored architectures and a common head/root style."""
    @override
    def __init__(self,
                 config: EncoderDecoderConfig,
                 output_multiplier: int = 1):
        """Configure and build the network.

        Parameters:
            config: `EncoderDecoderConfig` to configure the object.
            output_multiplier: Number of decoder output channels is multiplied by this value. This
                is necessary for autoencoders using likelihoods that require multiple parameters,
                for example a Beta distribution.
        """
        super().__init__()
        body_config = config.body_config

        enc_config = EncoderConfig(input_shape=config.input_shape,
                                   out_dim=config.latent_dim,
                                   head_style=config.style,
                                   body_config=body_config,
                                   block_config=config.block_config)
        self.encoder = EncoderNet(config=enc_config)

        body_config = replace(body_config,
                              blocks_per_level=list(reversed(body_config.blocks_per_level)),
                              level_filters=list(reversed(body_config.level_filters)),
                              level_strides=list(reversed(body_config.level_strides)),
                              level_attentions=list(reversed(body_config.level_attentions)))

        dec_config = DecoderConfig(input_shape=(config.input_shape[0] * output_multiplier,) + config.input_shape[1:],
                                   latent_dim=config.latent_dim,
                                   root_style=config.style,
                                   body_config=body_config,
                                   block_config=config.block_config)
        self.decoder = DecoderNet(config=dec_config)

        # required by some subclasses
        self.input_shape = config.input_shape

    @override
    def forward(self,
                inputs: DataBatchFloat) -> DataBatchFloat:
        return self.decoder(self.encoder(inputs))


@dataclass(frozen=True)
class EncoderDecoderConfig:
    """Config wrapping arguments for `EncoderDecoder` networks.
    
    The main reason we have this class is that we will have a bunch of models inheriting from
    `EncoderDecoder`, and having to duplicate all the arguments in the various `__init__` methods
    would be cumbersome. By bundling them within a dataclass, we only have to pass around a single
    object.  
    You will notice that these are not *all* arguments that `EncoderDecoder` can take. The main
    criterion here was to include arguments that are commonly just "passed through" to the `super`
    class. Arguments that tend to be set depending on context, such as `latent_multiplier` or
    `output_multiplier`, are not included, as they would commonly be overriden, and this may result
    in less clarity.

    Parameters:
        style: Common head/root (encoder/decoder) style. Strictly speaking, it's not necessary to
            use the same for both, but not doing so would be kinda weird.
        Remaining: See `EncodeConfig` and `DecoderConfig` docs.
    """
    input_shape: tuple[int, ...]
    latent_dim: int
    style: Literal["fc", "conv-flat", "conv"]
    body_config: BodyConfig
    block_config: BlockConfig


@dataclass(frozen=True)
class EncoderConfig:
    """Config wrapping arguments for `EncoderNet` models.
    
    See `EncoderDecoderConfig` for more notes. This only has arguments relevant for the encoder.
    Mainly intended for "standalone" encoder models, such as classifiers or GAN discriminators. Use
    `EncoderDecoderConfig` for combined models like autoencoders.

    Parameters:
        input_shape: Shape of a single input, without batch axis, given as `(channels, ...)` where
            `...` denotes spatial dimensions, e.g. `(channels, height, width)` for images.
        out_dim: Desired dimension of output.  
            NOTE that for `head_style="fc"` or `"pool-fc"`, this is the total size of the output
            vector. For `head_style="conv"`, the output feature map of the component `DownCNN` is
            mapped to this number of *channels*, after which the result is flattened. Since your
            `DownCNN` probably is not mapping to `1x1` pixel images, the resulting vector size will
            thus be *a multiple of* `out_dim`!
        head_style: What kind of head to apply. Can be one of `"fc"`, `"pool-fc"` or `"conv"`.  
            `"fc"` flattens the CNN output and then applies a fully-connected layer.  
            `"pool-fc"` first applies global average pooling, then acts like `"fc"`.  
            `"conv-flat"` applies a `1x1` convolution, then flattens.  
            `"conv"` applies a `1x1` conv without flattening, i.e. outputs stay in "image shape".  
            The idea is that `"fc"` results in "intermingled" dimensions, each of which can
            represent the entire image, whereas with `"conv"` the resulting dimensions maintain some
            degree of locality. Flattening after conv (as in `"conv-flat"`) can be useful to
            guarantee that latent representations are vectors.
        body_config: Config for the body.
        block_config: Config for the blocks in the network.
        global_dropout: Apply dropout in head after global pooling. Only has an effect if
            `head_style` is `"pool-fc"`.
    """
    input_shape: tuple[int, ...]
    out_dim: int
    head_style: Literal["fc", "pool-fc", "conv-flat", "conv"]
    body_config: BodyConfig
    block_config: BlockConfig
    global_dropout: float = 0.


@dataclass(frozen=True)
class DecoderConfig:
    """Config wrapping arguments for `DecoderNet` models.
    
    See `EncoderDecoderConfig` for more notes. This only has arguments relevant for the decoder.
    Mainly intended for "standalone" decoder models, such as GAN generators. Use
    `EncoderDecoderConfig` for combined models like autoencoders.

    Parameters:
        root_style: One of `'fc'`, '`conv-flat', or `'conv'`; determines how the root operates. Both
            options correspond to inverse versions of the `EncoderNet` `'head_style'`.  
            `'fc'` expects a vector input and first maps to the required number of units for
            `UpCNN`, then reshapes to the structured shape.  
            `'conv-flat'` expects flattened versions of structured inputs (e.g. images), reshapes
            them and then maps to the correct number of channels via `1x1` convolution.  
            `'conv'` expects unflattened feature maps.  
            Note that there is no `'pool-fc'` version here, as pooling cannot sensibly be
            inverted.
        Remaining: See `EncoderConfig`. Note that we use `input_shape` to derive the *output*
            channels simply because this is usually about reverting a latent representation back to
            data space. Also, `latent_dim` corresponds to `output_dim` for the `EncoderConfig`.
    """
    input_shape: tuple[int, ...]
    latent_dim: int
    root_style: Literal["fc", "conv-flat", "conv"]
    body_config: BodyConfig
    block_config: BlockConfig
