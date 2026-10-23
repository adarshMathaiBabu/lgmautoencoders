"""This module implements a `LightningModule` for training a basic autoencoder."""
from collections.abc import Callable
from typing import Literal, override

import torch

from lgm.models import TrainingModuleBase
from lgm.optimizers import OptimizationConfig
from lgm.types import DataBatchFloat, LabelBatch, ScalarFloat
from lgm.utils import rescale_likelihood
from .model import Autoencoder


class AETrainingModule(TrainingModuleBase[Autoencoder]):
    """Lightning module for a basic Autoencoder.
    
    This is our first fully functional lightning module, and all other models we implement in this
    class will follow the same structure. As such, make sure you can understand what is going on
    here! It will get more complicated in the future.
    
    The main thing is to implement `core_step()`. This computes the loss that is then used for
    training. Depending on the model, there may be some smaller things to change with respect to the
    base class, e.g. autoencoders cannot `generate()` anything. Everything else is usually in
    service of `core_step()`.

    Autoencoders are trained to simply minimize the reconstruction loss between inputs and outputs.
    We also report a more general negative log-likelihood, scaled to bits/dim if desired. This makes
    comparison between models using different likelihoods much easier. For example, say you have one
    model with a `Gaussian` likelihood and one with `Laplace`: The reconstruction losses are squared
    and absolute error, respectively. These are not comparable directly. Likelihoods, however, bring
    both values on the same playing field. But be careful about comparing these likelihoods to other
    models we discuss later (e.g. VAEs, Flows); autoencoders are not really generative models by
    themselves, and as such these likelihoods are not really "complete".  
    It is recommended to add a `ReconstructionLogging` callback for subjective evaluation of output
    quality.
    """
    @override
    def __init__(self,
                 model: Autoencoder,
                 optimization_config: OptimizationConfig,
                 scale_multiplier: float = 256.,
                 reconstruction_loss_fn: Callable[[DataBatchFloat,
                                                   DataBatchFloat],
                                                   ScalarFloat] | None = None):
        """Configure the module.

        Parameters:
            model: `Autoencoder` instance.
            optimization_config: See `TrainingModuleBase`.
            scale_multiplier: See `lgm.utils.rescale_likelihood`. Will not affect training loss,
                only the reported NLL.
            reconstruction_loss_fn: If desired, you can overwrite the likelihood-based loss by
                passing a loss function yourself.
        """
        super().__init__(model=model,
                         optimization_config=optimization_config)
        if reconstruction_loss_fn is None:
            reconstruction_loss_fn = self.model.likelihood.loss
        self.reconstruction_loss_fn = reconstruction_loss_fn

        self.scale_multiplier = scale_multiplier

    @override
    def forward(self,
                inputs: DataBatchFloat,
                map_to_expected: bool = False) -> DataBatchFloat:
        """Forward method with additional expected value toggle. See `Autoencoder.forward()`."""
        return self.model(inputs, map_to_expected=map_to_expected)

    @override
    def core_step(self,
                  tag: Literal["train", "val", "test"],
                  batch: tuple[DataBatchFloat, LabelBatch],
                  batch_index: int) -> ScalarFloat:
        """Compute reconstructions and corresponding loss.

        Parameters:
            See: `TrainingModuleBase`.

        Returns:
            Scalar reconstruction loss for the batch.
        """
        input_batch, _ = batch
        # NOTE since we want to train the model to return correct *parameters*, we do not
        # map_to_expected here
        output_batch = self(input_batch, map_to_expected=False)
        reconstruction_loss = self.reconstruction_loss_fn(output_batch, input_batch)

        # the below just for show :) the loss above is sufficient for training!
        # this only becomes relevant once we want to link reconstruction losses to data likelihoods!
        with torch.inference_mode():
            nll = self.model.likelihood.nll(output_batch, input_batch)
            nll = rescale_likelihood(nll,
                                     input_batch,
                                     self.scale_multiplier,
                                     self.trainer.datamodule.normalizer.log_inverse_jacobian)

        self.log_metrics(tag,
                         {"reconstruction_loss": reconstruction_loss,
                          "negative_log_likelihood": nll})
        return reconstruction_loss
    
    @override
    def predict_step(self,
                     batch: tuple[DataBatchFloat, LabelBatch],
                     batch_index: int) -> DataBatchFloat:
        # NOTE since we (probably) want to return proper reconstructions, we map_to_expected here!
        # since we never really use predict_step, this is actually questionable.
        return self.model.reconstruct(batch[0])

    @override
    def generate(self,
                 n_samples: int,
                 **kwargs):
        raise NotImplementedError("generate() not meaningful for autoencoder.")
