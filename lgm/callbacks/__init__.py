"""Custom callbacks to augment the training process.

As a rough guide:
- FitInfo, HyperParamLogger, can always be added.
- BestCheckpoint can usually be added, but for some model types, there isn't really a good metric to
  use. Examples for this are RBMs or GANs. Probably better to leave this out for those models.
- Autoencoders of any type should add `ReconstructionLogging`. Note that VAEs are both generative
  and autoencoders, so the last point also applies.
"""
from .image import GenerationLogging, ReconstructionLogging
from .logging import FitInfo, HyperParamLogger
from .saving import BestCheckpoint
