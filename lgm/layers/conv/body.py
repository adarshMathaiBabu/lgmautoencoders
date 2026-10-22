"""CNN "body" implementation. See `CNNBody` and `BodyConfig` class docs for more information."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import override

from torch import nn

from lgm.types import DataBatchFloat
from lgm.layers.norms import n_groups_convert, norm_selector
from .blocks import BlockConfig
from .level import CNNLevel


class CNNBody(nn.Module):
    """A sequence of CNN "levels", which in turn consist of sequences of "blocks".
    
    Each level can have a different number of filters, and optional down-/up-sampling.
    """
    @override
    def __init__(self,
                 direction: str,
                 body_config: BodyConfig,
                 block_config: BlockConfig):
        """Configure the body and create levels.
    
        Parameters:
            direction: One of `"up"` or `"down"`. `"down"` would usually be for encoder-type models,
                `"up"` for decoder/generator-like.
            body_config: `BodyConfig` object.
            block_config: `BlockConfig` to config the blocks.
        """
        super().__init__()

        self.levels = nn.ModuleList()
        previous_filters = body_config.level_filters[0]
        for level_ind in range(len(body_config.level_filters)):
            current_filters = body_config.level_filters[level_ind]
            self.levels.append(CNNLevel(body_config.blocks_per_level[level_ind],
                                        previous_filters,
                                        current_filters,
                                        stride=body_config.level_strides[level_ind],
                                        direction=direction,
                                        block_config=block_config,
                                        use_attention=body_config.level_attentions[level_ind]))
            previous_filters = current_filters

        self.final_normact = body_config.final_normact
        if self.final_normact:
            norm_groups = n_groups_convert(block_config.norm_groups, body_config.level_filters[-1])
            self.final_norm = norm_selector(block_config.norm, block_config.data_format)(
                norm_groups,
                body_config.level_filters[-1])
            self.final_activation = block_config.activation()
            # TODO use dropout here?

    @override
    def forward(self,
                inputs: DataBatchFloat) -> DataBatchFloat:
        """Apply the body.
        
        Parameters:
            inputs: Input features.

        Returns:
            The final output features.
        """
        outputs = inputs
        for level in self.levels:
            outputs = level(outputs)

        if self.final_normact:
            outputs = self.final_activation(self.final_norm(outputs))
        return outputs


@dataclass(frozen=True)
class BodyConfig:
    """Helper object to bundle parameters for `CNNBody`.
    
    In practice, `CNNBody` is often called as a chain like `EncoderNet -> DownCNN -> CNNBody`.
    Carrying arguments all the way through is cumbersome. Thus, we bundle them in a config, as
    `**kwargs` are not an option (see `BlockConfig` docs).

    NOTE that this does not include all arguments to `CNNBody`, as some are generally set depending
    on context by the class(es) above. Also NOTE that certain classes may override some attributes 
    to make sure they work in context.

    Parameters:
        blocks_per_level: How many blocks each level consists of. Can be `int`, then all levels will
            use the same number of blocks. Or an iterable like `filters` and `strides.`
        level_filters: Iterable (e.g. list) with number of filters for each level.
        level_strides: Stride (up- or downsampling factor) at each level. If not given, will use
            stride of 1 for the first level, and 2 for all others. This is a sensible default for
            encoder-style models working on images, causing the first level to work at the original
            input resolution, and each successive level at a reduced one. Take care to explictly
            invert this for decoder-style models!  
            For models on e.g. raw audio, you will likely want to use larger strides, such as 4.
        level_attentions: Optional iterable of bools. Each level where this is `True` will have
            multi-head self-attention blocks interspersed after each convolutional  block. Commonly
            used in diffusion models at lower resolutions.  
            NOTE be careful with this -- activating at high resolutions will result in massive
            memory and runtime increase. Usually recommended at `16x16` at most, especially for the
            kind of models/data we use in this course.
        final_normact: If `True`, apply a final sequence of normalization & activation after the
            last level. This can improve performance somewhat. But note that this can sometimes be
            superfluous in models where we apply multiple `CNNBodys` in sequence, e.g. U-Nets. Here,
            the first `Body` would end with a final `norm-act`, and the second body would also start
            with `norm-act`. In such cases, you could set this to `False` for the first `Body`.
    """
    blocks_per_level: int | Iterable[int]
    level_filters: Iterable[int]
    level_strides: Iterable[int] | None = None
    level_attentions: Iterable[bool] | None = None
    final_normact: bool = True

    @override
    def __post_init__(self):
        """Validate arguments and bring to common form.
        
        Raises:
            ValueError: If any of the per-level lists have different lengths.
        """
        if self.level_strides is None:
            object.__setattr__(self, "level_strides",
                               [1] + [2] * (len(self.level_filters) - 1))
        if self.level_attentions is None:
            object.__setattr__(self, "level_attentions",
                               [False] * len(self.level_filters))
        if isinstance(self.blocks_per_level, int):
            object.__setattr__(self, "blocks_per_level",
                               [self.blocks_per_level] * len(self.level_filters))

        lengths = {"blocks": len(self.blocks_per_level),
                   "filters": len(self.level_filters),
                   "strides": len(self.level_strides),
                   "attentions": len(self.level_attentions)}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Expected all configuration lists to have the same length, got "
                             f"{lengths}.")
