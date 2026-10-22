"""This module provides functionalities for standard autoencoders.

These are not generative models, but serve as a good warmup. We will also be able to reuse parts of
the code for some actual generative models, such as variational autoencoders and their variants.

Pretty much all models we implement follow this structure: `model` implements the actual network,
defining how inputs are transformed into outputs, how to generate samples etc. `training` implements
a `LightningModule` for training the model. There may be additional submodules, and `model` may be
split into separate parts as well, depending on what we are doing.

For autoencoders, `likelihoods` is of particular importance -- this implements several choices for
data likelihoods, which then correspond to specific loss functions for maximum likelihood training.
This will become crucial for variational autoencoders later, so we want to already get used to it
now!
"""
from .likelihoods import Likelihood
from .model import Autoencoder
from .training import AETrainingModule
