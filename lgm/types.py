"""This module uses jaxtyping to add various more specific tensor types.

Annotating tensors as just torch.Tensor isn't very informative. Is this a scalar? A batch? If so, a
batch of what? Images? Time series? Or maybe a vector? These types are supposed to clear up such
questions.

This is all a bit chaotic, but covers many different types that can help get across the intention
behind an argument even without using static type checkers, although jaxtyping presumably supports
the latter. "Intention" is also why some types are seemingly redundant/duplicated.
"""
from collections.abc import Callable
from typing import Literal, TypeAlias, TypeVar

from jaxtyping import Bool, Float, Int
from torch import nn, Tensor

Model = TypeVar("Model", bound=nn.Module)


DataFormat = Literal["1d", "2d", "3d"]
Number: TypeAlias = int | float

ScalarBool: TypeAlias = Bool[Tensor, ""]


### FLOAT DATA INPUTS
ImageFloat: TypeAlias = Float[Tensor, "channel height width"]
ImageBatchFloat: TypeAlias = Float[Tensor, "batch channel height width"]
BroadcastImageBatchFloat: TypeAlias = Float[Tensor, "batch channel 1 1"]

SequenceFloat: TypeAlias = Float[Tensor, "channel time"]
SequenceBatchFloat: TypeAlias = Float[Tensor, "batch channel time"]
BroadcastSequenceBatchFloat: TypeAlias = Float[Tensor, "batch channel 1"]

TabularFloat: TypeAlias = Float[Tensor, "channel"]
TabularBatchFloat: TypeAlias = Float[Tensor, "batch channel"]

DataFloat = TypeVar("DataFloat",
                    ImageFloat,
                    SequenceFloat,
                    TabularFloat)
DataBatchFloat = TypeVar("DataBatchFloat",
                         ImageBatchFloat,
                         SequenceBatchFloat,
                         TabularBatchFloat)
BroadcastDataBatchFloat = TypeVar("BroadcastDataBatchFloat",
                                  BroadcastImageBatchFloat,
                                  BroadcastSequenceBatchFloat)


### OTHER FLOATS
ScalarFloat: TypeAlias = Float[Tensor, ""]
VectorBatchFloat: TypeAlias = Float[Tensor, "batch"]
VectorTimeFloat: TypeAlias = Float[Tensor, "time"]
VectorChannelFloat: TypeAlias = Float[Tensor, "channel"]
VectorFloat: TypeAlias = Float[Tensor, "d"]
BatchOrScalarFloat = TypeVar("BatchOrScalarFloat",
                             ScalarFloat,
                             VectorBatchFloat)
TimeOrScalarFloat = TypeVar("TimeOrScalarFloat",
                            ScalarFloat,
                            VectorTimeFloat)

DistanceMatrixFloat: TypeAlias = Float[Tensor, "batch1 batch2"]
SquareMatrixFloat: TypeAlias = Float[Tensor, "d d"]


### INDEX TYPES
ImageIndex: TypeAlias = Int[Tensor, "height width"]
ImageBatchIndex: TypeAlias = Int[Tensor, "batch height width"]

SequenceIndex: TypeAlias = Int[Tensor, "time"]
SequenceBatchIndex: TypeAlias = Int[Tensor, "batch time"]

BatchIndex: TypeAlias = Int[Tensor, "batch"]
DataBatchIndex = TypeVar("DataBatchIndex",
                         ImageBatchIndex,
                         SequenceBatchIndex,
                         BatchIndex)
DataIndex = TypeVar("DataIndex",
                    ImageIndex,
                    SequenceIndex)

ScalarIndex: TypeAlias = Int[Tensor, ""]

# bit confusing that these are repeated. but they have a different "intention".
# time for SequenceIndex is e.g. time in an audio wave.
# time for VectorTimeIndex is e.g. "time" in a diffusion model.
VectorBatchIndex: TypeAlias = Int[Tensor, "batch"]
VectorTimeIndex: TypeAlias = Int[Tensor, "time"]


### CHANNELS LAST TYPES
ImageFloatChannelsLast: TypeAlias = Float[Tensor, "height width channel"]
ImageBatchFloatChannelsLast: TypeAlias = Float[Tensor, "batch height width channel"]
BroadcastImageBatchFloatChannelsLast: TypeAlias = Float[Tensor, "batch 1 1 channel"]

SequenceFloatChannelsLast: TypeAlias = Float[Tensor, "time channel"]
SequenceBatchFloatChannelsLast: TypeAlias = Float[Tensor, "batch time channel"]
BroadcastSequenceBatchFloatChannelsLast: TypeAlias = Float[Tensor, "batch 1 channel"]

DataFloatChannelsLast = TypeVar("DataFloatChannelsLast",
                                ImageFloatChannelsLast,
                                SequenceFloatChannelsLast,
                                TabularFloat)
DataBatchFloatChannelsLast = TypeVar("DataBatchFloatChannelsLast",
                                     ImageBatchFloatChannelsLast,
                                     SequenceBatchFloatChannelsLast,
                                     TabularBatchFloat)
BroadcastDataBatchFloatChannelsLast = TypeVar("BroadcastDataBatchFloatChannelsLast",
                                              BroadcastImageBatchFloatChannelsLast,
                                              BroadcastSequenceBatchFloatChannelsLast)


## LABELS
# either a single index, or a vector (soft labels)
# NOT ROBUST to e.g. multi-label setup
Label = TypeVar("Label", ScalarIndex, VectorChannelFloat)
LabelBatch = TypeVar("LabelBatch", VectorBatchIndex, TabularBatchFloat)

DataTransform: TypeAlias = Callable[[DataBatchFloat], DataBatchFloat]
ImageTransform: TypeAlias = Callable[[ImageBatchFloat], ImageBatchFloat]
