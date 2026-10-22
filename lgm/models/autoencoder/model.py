"""This module implements the "core model" for an autoencoder."""
from typing import override

from lgm.layers import EncoderDecoder, EncoderDecoderConfig
from lgm.types import DataBatchFloat, TabularBatchFloat
from .likelihoods import Likelihood


class Autoencoder(EncoderDecoder):
    """It's an autoencoder!
    
    In practice, this is a thin wrapper around `EncoderDecoder`. Truth be told, this class itself
    is no more "auto" than the base class -- that comes through the training, which mandates the
    targets being equal to the inputs.  
    Either way, this class mainly adds a `Likelihood` object to the model. Network outputs are then
    interpreted as distribution parameters, which have to be mapped to the distribution's expected
    value to get proper reconstructions.
    """
    @override
    def __init__(self,
                 likelihood: Likelihood,
                 enc_dec_config: EncoderDecoderConfig):
        """Configure the module.

        Parameters:
            likelihood: The likelihood we assume for the data, which leads to a loss function
                equivalent to maximum likelihood.
            enc_dec_config: `EncoderDecoderConfig` for the network.
        """
        super().__init__(config=enc_dec_config,
                         output_multiplier=likelihood.n_parameters)
        self.likelihood = likelihood

    @override
    def forward(self,
                inputs: DataBatchFloat,
                map_to_expected: bool = False) -> DataBatchFloat:
        """Apply the autoencoder.
        
        Parameters:
            inputs: Batch of inputs.
            map_to_expected: If `True`, model outputs (distribution parameters) are mapped to
                expected values. Only with this flag will you get "true" reconstructions. Otherwise,
                the model returns distribution parameters.

        Returns:
            Reconstruction parameters or expected values, depending on `map_to_expected`.
        """
        reconstructions = super().forward(inputs)
        if map_to_expected:
            reconstructions = self.expected_value(reconstructions)
        return reconstructions

    def reconstruct(self,
                    inputs: DataBatchFloat) -> DataBatchFloat:
        """Shortcut for `forward(inputs, map_to_expected=True)`."""
        return self(inputs, map_to_expected=True)

    def reconstruct_from_code(self,
                              latents: TabularBatchFloat):
        """Shortcut for `expected_value(self.decoder(latents))`."""
        return self.expected_value(self.decoder(latents))

    def expected_value(self, 
                       outputs: DataBatchFloat) -> DataBatchFloat:
        """Convert network outputs (distribution parameters) to data (expected values).

        Parameters:
            outputs: The parameters to convert.

        Returns:
            Expected values for model likelihood.
        """
        return self.likelihood.expected_value(outputs)
