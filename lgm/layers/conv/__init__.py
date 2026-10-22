"""Building blocks required to build convolutional neural networks.

We strive for a balance between offering many options, but also relying on strong baseline ideas to
not overcomplicate things. The essential structure of our CNN is as follows:

- At the lowest level, we have "blocks" representing the smallest unit of computation in a CNN. The
prime example would be a residual block. This consists of a couple of convolutional layers with
normalization and activation, the output of which is added to the input to the block. In fact, the
only block we currently implement is the "pre-activation" residual block. However, other blocks, for
example Inception or Dense blocks, could also be used without changing anything about the other
parts of the hierarchy.
- A "level" is a sequence of blocks, all of which output the same dimensionality. This refers to
both number of channels and spatial size (e.g. number of pixels). However, a level may (and often
does) contain other operations, such as up-/downsampling, that modify input size. The "same size"
contract only holds for the blocks within a level. You can think of one level as "all the
computation that happens on one resolution", and it usually starts or ends with a change in size, so
that each level works at a different resolution than the one before/after.
- A "body" is a sequence of levels. It essentially represents an entire CNN computation from start
to finish.

A full architecture may combine multiple bodies. For example, a U-Net can be seen as a
"downsampling" body followed by an "upsampling" body. See the `lgm.layers.networks` module for full
architectures making use of these structures.
"""
from .blocks import BlockConfig, ConvBlockPrenorm
from .body import BodyConfig, CNNBody
from .core import (INTERPOLATION_MAP, DownsampleAA, SqueezeExcite, UpsampleAA, adaptive_avg_pool_nd,
                   conv_nd, dropout_nd)
from .level import CNNLevel
