"""This module contains various reusable layers and networks to build architectures with.

REFERENCES
ResNet: https://arxiv.org/abs/1512.03385
Preactivation/normalization blocks: https://arxiv.org/abs/1603.05027
Squeeze-and-excite: https://arxiv.org/abs/1709.01507
Batch normalization: https://arxiv.org/abs/1502.03167
Group normalization: https://arxiv.org/abs/1803.08494
"Special" CNNBody architectures (input/output skips/residuals): https://arxiv.org/abs/1912.04958
Antialiased downsampling: https://arxiv.org/pdf/1904.11486

Some activation functions:
ReLU: https://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf
PReLU (and He/Kaiming initialization): https://openaccess.thecvf.com/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
GELU: https://arxiv.org/abs/1606.08415
Swish/SiLU: https://arxiv.org/abs/1710.05941v1
Mish (my favorite :)): https://arxiv.org/abs/1908.08681
"""
from .conv import (BlockConfig, BodyConfig, CNNBody, CNNLevel, ConvBlockPrenorm, DownsampleAA,
                   SqueezeExcite, UpsampleAA, adaptive_avg_pool_nd, dropout_nd)
from .generic import AttentionBlock, OneHot
from .networks import (DecoderConfig, DecoderNet, DownCNN, EncoderConfig, EncoderDecoder,
                       EncoderDecoderConfig, EncoderNet, UpCNN)
from .norms import norm_selector
