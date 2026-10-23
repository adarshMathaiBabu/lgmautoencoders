"""This module implements likelihoods for the data space.

When you try to model data with a likelihood-based probabilistic model such as a VAE, you need to
settle on a distribution for the data space. As we discussed in the exercise, this then naturally
leads to a reconstruction loss function `-log p(x|z)` according to the maximum likelihood principle.
In this module, we provide various likelihoods with corresponding loss functions and related
utilities.

All of these should "work" for image data in the range [0, 1].
For other modalities or re-scaled data, there may be restrictions.
For example, the Bernoulli or Beta likelihoods expect data in the appropriate range.
Gaussian or Laplace likelihoods are flexible in that regard.

We could implement many, many more likelihoods -- either totally different ones, or more variations
on Gaussian, Laplace etc. This can be seen as more of a selection, showcasing some of the things
that are possible/valid.
"""
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar, override

import torch
from torch import nn

from lgm.ops import mean_except, sum_except
from lgm.types import DataBatchFloat, ScalarFloat


class Likelihood(nn.Module, ABC):
    """Base class for likelihoods.
    
    This covers both choosing the "correct" loss function for maximum likelihood training, as well
    as converting distribution parameters to expected values for proper predictions
    (reconstructions). We also support full negative log-likelihood including normalization
    constants, which often drop out of the loss.
    """
    n_parameters: ClassVar[int]

    def loss(self,
             outputs: DataBatchFloat,
             targets: DataBatchFloat) -> ScalarFloat:
        """Compute loss corresponding to negative log-likelihood.

        NOTE that these do NOT need to be full negative log-likelihoods -- normalizing constants may
        be discarded where it is convenient to do so. If you want proper likelihoods, see the
        `nll()` method. Still, we use the nll as the default loss, since it works just as well; it
        just has some more constants floating around.
        
        Parameters: 
            outputs: Model outputs (distribution parameters!).
            targets: Target observations (NOT parameters!).

        Returns:
            Scalar loss value, minimizing which corresponds to maximizing likelihood.
        """
        return self.nll(outputs, targets)

    @abstractmethod
    def nll(self,
            outputs: DataBatchFloat,
            targets: DataBatchFloat) -> ScalarFloat:
        """Compute full negative log-likelihood, including normalizing constants.
        
        This can be useful for comparing different likelihood functions. Without normalizing
        constants, the losses for different likelihoods are not really comparable, as the missing
        parts may be wildly different. By including all components of the likelihood, different
        classes become more easier to compare.
        """

    @abstractmethod
    def expected_value(self,
                       parameters: DataBatchFloat) -> DataBatchFloat:
        """Map distribution parameters to expected values.
        
        Usually, we don't want to sample from the output distribution; rather, we want deterministic
        reconstructions. The expected value is the best deterministic choice under some metrics.

        Parameters:
            parameters: Distribution parameters as returned by a decoder network.

        Returns:
            Expected values for the distributions as given by `parameters`.
        """


class MeanSquaredError(Likelihood):
    """Dummy "likelihood" that just computes squared error.
    
    This is NOT valid for proper probabilistic modeling, but can serve as a simple loss to train
    basic autoencoders.
    """
    n_parameters: ClassVar[int] = 1

    @override
    def nll(self,
            outputs: DataBatchFloat,
            targets: DataBatchFloat) -> ScalarFloat:
        return ((outputs - targets)**2).mean()

    @override
    def expected_value(self,
                       parameters: DataBatchFloat) -> DataBatchFloat:
        return parameters


class GaussianBase(Likelihood, ABC):
    """Base class for Gaussian likelihoods where the model only predicts `mu`.
    
    Sigma can then either be fixed, or be chosen in some optimal manner.

    Core assumptions:
    - Data has unlimited range, though it is extremely unlikely far away from the mean.
    - Distribution is unimodal.
    - Dimensions are independent (usually given latents).
    """
    @override
    def loss(self,
             outputs: DataBatchFloat,
             targets: DataBatchFloat) -> ScalarFloat:
        squared_difference = (outputs - targets)**2
        sigma = self.get_sigma(outputs, targets)
        scaled_difference = squared_difference / (2*sigma**2)
        sigma_term = torch.log(sigma)
        return sum_except(scaled_difference + sigma_term).mean()

    @override
    def nll(self,
            outputs: DataBatchFloat,
            targets: DataBatchFloat) -> ScalarFloat:
        gaussian = torch.distributions.Normal(loc=outputs,
                                              scale=self.get_sigma(outputs, targets))
        return -sum_except(gaussian.log_prob(targets)).mean()

    @override
    def expected_value(self,
                       parameters: DataBatchFloat) -> DataBatchFloat:
        return parameters

    @abstractmethod
    def get_sigma(self,
                  outputs: DataBatchFloat,
                  targets: DataBatchFloat) -> DataBatchFloat:
        """Get value(s) for the per-dimension standard deviation `sigma`.
        
        Parameters:
            See: As for `loss()` or `nll()` methods.

        Returns:
            `sigma` in the same shape as `outputs`/`targets`, broadcast if necessary.
        """


class GaussianFixedSigma(GaussianBase):
    """Assumes data comes from an independent Gaussian with a single fixed `sigma`.
    
    `sigma` then drops out of the loss and expected value completely, and can thus technically be
    "unknown". However, you MUST pass some value for sigma if you want to be able to compute the
    full log-likelihood including normalizing constants! Also, we use one loss across all Gaussian
    likelihoods for consistency, and that one includes the sigma-dependent multiplier on the squared
    error, as well as the additive `log(sigma)`.
    """
    n_parameters: ClassVar[int] = 1
    sigma: ScalarFloat

    @override
    def __init__(self,
                 sigma: float | None = None):
        """Configure the likelihood.
        
        Parameters:
            sigma: Fixed standard deviation for this distribution, applied to all data dimensions.
                Note that while not really intended, you could also pass an array here that can be
                broadcast to the data batches, and thus choose different sigmas for each dimension
                in the data. However, this might lead to incorrect handling of dtypes. The intended
                use is to pass a single float.  
                If not given, defaults to `sqrt(0.5)`. This leads to `2*sigma**2 = 1`, and thus
                would result in a "clean" squared error with no scaling. However, the `log(sigma)`
                term would still be there. As `sqrt(0.5)` is smaller than 1, `log(sqrt(0.5))` is
                below 0. As such, while this default setting does not *scale* the squared error, it
                does add an (negative) *offset*.

        Raises:
            ValueError: If `sigma` is 0 or negative.
        """
        super().__init__()
        if sigma is None:
            sigma = math.sqrt(0.5)
        if sigma <= 0.:
            raise ValueError("sigma must be positive.")
        self.register_buffer("sigma", torch.tensor(sigma))

    @override
    def get_sigma(self,
                  outputs: DataBatchFloat,
                  targets: DataBatchFloat) -> DataBatchFloat:
        return torch.broadcast_to(self.sigma, outputs.shape)


class GaussianOptimalImageSigma(GaussianBase):
    """Assumes data comes from an independent Gaussian with optimal `sigma` chosen per data sample.
    
    A first evolution over `GaussianFixedSigma` might be to have a second output besides the mean,
    so the model also estimates `sigma`. However, this doesn't always work well, and it turns out
    you can actually derive the best `sigma` output given the mean output, *if* the targets are
    known. Since `sigma` is only required for training and drops out of the expected value, this is
    sufficient. But note that this means you cannot get an uncertainty estimate from your model, as
    there is no `sigma` output!

    This version computes the optimal `sigma` for each data sample, i.e. one per image, for example,
    but the parameter is shared over the data dimensions, e.g. pixels. It turns out the best `sigma`
    for each data sample is simply the average squared difference for that sample, over all data
    dimensions.
    """
    n_parameters: ClassVar[int] = 1

    @override
    def __init__(self,
                 lower_lim: float = 0.):
        """Configure the likelihood.
        
        Parameters:
            lower_lim: If the optimal `sigma`  is below this, it will be clipped upwards to this
                value. Otherwise, it may approach 0 and lead to numerical trouble.
        """
        super().__init__()
        self.lower_lim = lower_lim

    @override
    def loss(self,
             outputs: DataBatchFloat,
             targets: DataBatchFloat) -> ScalarFloat:
        if self.lower_lim == 0.:
            # formula can simplify
            feature_dim = torch.prod(torch.tensor(outputs.shape[1:]))
            sum_of_squares = sum_except((outputs - targets)**2, keepdim=True)
            return 0.5*feature_dim * torch.log(sum_of_squares).mean()
        else:
            return super().loss(outputs, targets)

    @override
    def get_sigma(self,
                  outputs: DataBatchFloat,
                  targets: DataBatchFloat) -> DataBatchFloat:
        squared_difference = (outputs - targets)**2
        sigmas = torch.sqrt(mean_except(squared_difference, keepdim=True))
        return torch.broadcast_to(torch.clamp(sigmas, self.lower_lim),
                                  outputs.shape)


class GaussianOptimalPixelSigma(GaussianBase):
    """Independent Gaussian with optimal `sigma` chosen for each pixel in each sample.
    
    This tends to be rather unstable... It will put extreme focus on pixels that are easy to
    predict, such as background pixels in MNIST.

    See `GaussianOptimalImageSigma` for more general notes!
    """
    n_parameters: ClassVar[int] = 1

    @override
    def __init__(self,
                 lower_lim: float = 0.):
        """Configure the likelihood.
        
        Parameters:
            lower_lim: If the optimal `sigma` is below this, it will be clipped upwards to this
                value. Otherwise, it may approach 0 and lead to numerical trouble.
        """
        super().__init__()
        self.lower_lim = lower_lim

    @override
    def loss(self,
             outputs: DataBatchFloat,
             targets: DataBatchFloat) -> ScalarFloat:
        if self.lower_lim == 0.:
            absolute_difference = (outputs - targets).abs()
            return sum_except(torch.log(absolute_difference)).mean()
        else:
            return super().loss(outputs, targets)

    @override
    def get_sigma(self,
                  outputs: DataBatchFloat,
                  targets: DataBatchFloat) -> DataBatchFloat:
        sigmas = (outputs - targets).abs()
        return torch.clamp(sigmas, self.lower_lim)


class GaussianLearnedSigma(GaussianFixedSigma):
    """Assumes data comes from an independent Gaussian with a single trainable `sigma`.
    
    This can find the "best" value for sigma, i.e. the one resulting in the lowest loss. Note that
    there are two conflicting terms in the loss: The squared difference is divided by sigma, so
    choosing a larger value to "accommodate" larger differences will result in a lower loss.
    However, the log(sigma) term punishes large standard deviations. In practice, there will be a
    compromise somewhere in the middle.

    Note that this doesn't really add anything for regular autoencoders, as it mainly scales and
    shifts the squared difference. But for VAEs, it significantly changes the relation between
    reconstruction loss and KL-divergence.
    """
    n_parameters: ClassVar[int] = 1

    @override
    def __init__(self,
                 sigma: float | None = None):
        """Configure the likelihood.
        
        Parameters:
            sigma: Initial value for the standard deviation for this distribution, applied to all
                data dimensions.
                See `GaussianFixedSigma` for more notes.

        Raises:
            ValueError: If `sigma` is 0 or negative.
        """
        super().__init__()
        if sigma is None:
            sigma = math.sqrt(0.5)
        if sigma <= 0.:
            raise ValueError("sigma must be positive.")
        self.sigma = nn.Parameter(torch.tensor(sigma))


class LaplaceBase(Likelihood, ABC):
    """Base class for Laplace likelihoods where the model only predicts `mu`.
    
    The scale `b` can then either be fixed, or be chosen in some optimal manner. All `Laplace`
    likelihoods are essentially copy & paste from the `Gaussian`, but replacing squared error by
    absolute error.

    Core assumptions:
    - Exactly as `GaussianFixedSigma`; the distribution just has a somewhat different shape, in
    particular with stronger "tails". This means that values further away from the mean are more 
    likely, so they are punished less harshly.
    """
    @override
    def loss(self,
             outputs: DataBatchFloat,
             targets: DataBatchFloat) -> ScalarFloat:
        abs_difference = (outputs - targets).abs()
        scale = self.get_scale(outputs, targets)
        scaled_difference = abs_difference / scale
        scale_term = torch.log(scale)
        return sum_except(scaled_difference + scale_term).mean()

    @override
    def nll(self,
            outputs: DataBatchFloat,
            targets: DataBatchFloat) -> ScalarFloat:
        laplace = torch.distributions.Laplace(loc=outputs,
                                              scale=self.get_scale(outputs, targets))
        return -sum_except(laplace.log_prob(targets)).mean()

    @override
    def expected_value(self,
                       parameters: DataBatchFloat) -> DataBatchFloat:
        return parameters

    @abstractmethod
    def get_scale(self,
                  outputs: DataBatchFloat,
                  targets: DataBatchFloat) -> DataBatchFloat:
        """Get value(s) for the per-dimension scale `b.`
        
        Parameters:
            See: As for `loss()` or `nll()` methods.

        Returns:
            `b` in the same shape as `outputs`/`targets,` broadcast if necessary.
        """
        raise NotImplementedError("get_scale() not implemented.")


class LaplaceFixedScale(LaplaceBase):
    """Assumes data comes from an independent Laplace distribution with a single fixed `scale`."""
    n_parameters: ClassVar[int] = 1
    scale: ScalarFloat
    
    @override
    def __init__(self,
                 scale: float = 1.):
        """Configure the likelihood.
        
        Parameters:
            scale: Fixed scale for this distribution, applied to all data dimensions. Same notes
                apply as for `GaussianFixedSigma`.  
                If not given, defaults to 1. This results in a "clean" absolute error with no
                scaling. Since `log(1)` is 0, it is in fact a pure absolute error loss.

        Raises:
            ValueError: If `scale` is 0 or negative.
        """
        super().__init__()
        if scale <= 0.:
            raise ValueError("scale must be positive.")
        self.register_buffer("scale", torch.tensor(scale))

    @override
    def get_scale(self,
                  outputs: DataBatchFloat,
                  targets: DataBatchFloat) -> DataBatchFloat:
        return torch.broadcast_to(self.scale, outputs.shape)


class LaplaceOptimalImageScale(LaplaceBase):
    n_parameters: ClassVar[int] = 1

    @override
    def __init__(self,
                 lower_lim: float = 0.):
        """Configure the likelihood.
        
        Parameters:
            lower_lim: If the optimal `scale` is below this, it will be clipped upwards to this
                value. Otherwise, it may approach 0 and lead to numerical trouble.
        """
        super().__init__()
        self.lower_lim = lower_lim

    @override
    def get_scale(self,
                  outputs: DataBatchFloat,
                  targets: DataBatchFloat) -> DataBatchFloat:
        abs_difference = (outputs - targets).abs()
        scales = mean_except(abs_difference, keepdim=True)
        return torch.broadcast_to(torch.clamp(scales, self.lower_lim),
                                  outputs.shape)


class Bernoulli(Likelihood):
    """Assumes data comes from an independent Bernoulli distribution.

    NOT a valid likelihood for continuous data, but it will still work as a reconstruction loss.  
    NOTE that outputs should be logits! No sigmoid in your model!
    
    Core assumptions:
    - Data is binary. This assumption does not hold for pretty much any dataset we care about.
    Still, an autoencoder trained with the resulting loss (binary cross-entropy extended to the real
    `[0, 1]` range), will work just fine. Just don't interpret the values you get too much.
    - Dimensions are independent (usually given latents).
    """
    n_parameters: ClassVar[int] = 1

    @override
    def nll(self,
            outputs: DataBatchFloat,
            targets: DataBatchFloat) -> ScalarFloat:
        cross_entropy = nn.functional.binary_cross_entropy_with_logits(outputs, targets,
                                                                       reduction='none')
        return sum_except(cross_entropy).mean()
    
    @override
    def expected_value(self,
                       parameters: DataBatchFloat) -> DataBatchFloat:
        return torch.sigmoid(parameters)


class ContinuousBernoulli(Likelihood):
    """Assumes data comes from an independent Continuous Bernoulli distribution.

    This takes the Bernoulli formula end extends it to the range `[0, 1]` by normalizing it
    properly. Adapted from the TF code: https://github.com/cunningham-lab/cb_and_cc  
    NOTE that outputs should be logits! No sigmoid in your model!
    
    Core assumptions:

    - Data is in range `[0, 1]`, although values near the borders become numerically troublesome and
    hard to model.
    - Distribution is unimodal.
    - Dimensions are independent (usually given latents).
    """
    n_parameters: ClassVar[int] = 1

    @override
    def __init__(self,
                 limit_eps: float = 1e-5,
                 center_eps: float = 0.01):
        """Configure stability parameters for the loss and expected value.
        
        Parameters:
            limit_eps: The model output logits have to be converted to parameters in `(0, 1)`. Since
                the normalizing constant and expected value are unstable near the limits, we
                restrict them to the range `[eps, 1 - eps]`. The smallest tolerable value at
                `float32` precision seems to be `1e-7`.
            center_eps: Computations are similarly unstable near 0.5, as some intermediate results
                diverge at that point. Thus, formulas are replaced by simplified approximations in
                the range `[0.5 - eps, 0.5 + eps]`.
        """
        super().__init__()
        self.limit_eps = limit_eps
        self.upper_center = 0.5 + center_eps
        self.lower_center = 0.5 - center_eps

    @override
    def nll(self,
            outputs: DataBatchFloat,
            targets: DataBatchFloat) -> ScalarFloat:
        cross_entropy = nn.functional.binary_cross_entropy_with_logits(outputs, targets,
                                                                       reduction='none')
        normalizer = self.log_normalizer(outputs)
        return sum_except(cross_entropy - normalizer).mean()
    
    @override
    def expected_value(self,
                       parameters: DataBatchFloat) -> DataBatchFloat:
        probs = self.safe_sigmoid(parameters)
        numerical_condition = torch.logical_or(torch.less(probs, self.lower_center),
                                               torch.greater(probs, self.upper_center))
        cut_probs = torch.where(numerical_condition,
                                probs,
                                self.lower_center * torch.ones_like(probs))
        expected = (cut_probs/(2*cut_probs - 1) 
                    + 1/(2 * torch.atanh(1 - 2*cut_probs)))
        return torch.where(numerical_condition,
                           expected,
                           0.5 * torch.ones_like(expected))

    def log_normalizer(self,
                       logits: DataBatchFloat) -> DataBatchFloat:
        """Normalizer for the Continuous Bernoulli distribution.
        
        This is numerically problematic around 0.5, so we use a Taylor expansion around that value.
        Note that the value for `cut_probs` where the condition is `False` is actually irrelevant
        since the result at those points is not used. We just don't want to use 0.5, since that
        would result in `nan`.

        Parameters:
            logits: Pre-sigmoid parameters of the Continuous Bernoulli distribution for each
                dimension.

        Returns:
            Log of the normalizing constant for the distribution at each dimension.
        """
        probs = self.safe_sigmoid(logits)
        numerical_condition = torch.logical_or(torch.less(probs, self.lower_center),
                                               torch.greater(probs, self.upper_center))
        cut_probs = torch.where(numerical_condition,
                                probs,
                                self.lower_center * torch.ones_like(probs))
        log_normalizer = (torch.log(torch.abs(2 * torch.atanh(1 - 2 * cut_probs)))
                          - torch.log(torch.abs(1 - 2 * cut_probs)))
        taylor_expansion = (math.log(2.)
                            + 4/3 * torch.pow(probs - 0.5, 2) 
                            + 104/45 * torch.pow(probs - 0.5, 4))
        return torch.where(numerical_condition,
                           log_normalizer,
                           taylor_expansion)

    def safe_sigmoid(self,
                     logits: DataBatchFloat) -> DataBatchFloat:
        """Sigmoid returning values in a limited range, exluding borders at 0 and 1.
        
        The output value range is `[limit_eps, 1-limit_eps]`. This is necessary as some of the
        Continuous Bernoulli components can be unstable near the limits.

        Parameters:
            logits: Pre-sigmoid parameter of the Continuous Bernoulli distribution for each
                    dimension.

        Returns:
            Sigmoid restricted to the range `[eps, 1 - eps]`.
        """
        return self.limit_eps + (1 - 2*self.limit_eps) * torch.sigmoid(logits)


class Beta(Likelihood):
    """Assumes data comes from an independent Beta distribution.

    NOTE that there must be two outputs for each target, as the Beta distribution has two
    parameters. E.g. for a three-channel color image input, model outputs should have six channels.
    Also, model outputs should have unrestricted range, as they will be mapped to positive
    distribution parameters automatically.

    Furthermore, the Beta distribution can be very flexible, including being bimodal if both
    parameters are < 1. This makes the expected value a questionable choice in those cases, and
    could lead to artifacts in the outputs such as gray "smears".
    
    Core assumptions:

    - Data is in range `(0, 1)`, i.e. the open interval. Targets are clipped automatically when
      computing the loss.
    - Dimensions are independent (usually given latents).
    """
    n_parameters: ClassVar[int] = 2

    @override
    def __init__(self,
                 positive_function: Callable[[torch.Tensor], torch.Tensor] = torch.nn.functional.softplus,
                 limit_eps: float = 1e-3):
        """Configure the parameterization and stability parameters.
        
        Parameters:
            positive_function: Any function mapping arbitrary real numbers to strictly positive
                numbers.
            limit_eps: The Beta distribution is not defined for 0 or 1, only in the open interval
                between them. Thus, we clip target values to `[eps, 1 - eps]`. NOTE that this is NOT
                the same as for Continuous Bernoulli, where the *model outputs* are restricted;
                however due to the value range of the Beta distribution, outputs are also in the
                open interval in this case.
        """
        super().__init__()
        self.positive_function = positive_function
        self.limit_eps = limit_eps

    @override
    def nll(self,
            outputs: DataBatchFloat,
            targets: DataBatchFloat) -> ScalarFloat:
        targets = torch.clamp(targets, self.limit_eps, 1 - self.limit_eps)
        alphas, betas = self.convert_parameters(outputs)
        nll = -torch.distributions.Beta(alphas, betas).log_prob(targets)
        return sum_except(nll).mean()
    
    @override
    def expected_value(self,
                       parameters: DataBatchFloat) -> DataBatchFloat:
        alphas, betas = self.convert_parameters(parameters)
        return alphas / (alphas + betas)

    def convert_parameters(self,
                           outputs: DataBatchFloat) -> tuple[DataBatchFloat, DataBatchFloat]:
        """Turn unrestricted network outputs into proper positive Beta distribution parameters.
        
        We do this by splitting along axis 1 (valid for `Linear` layers or channels-first
        convolutions) and applying a function that should have strictly positive outputs. Typical
        candidates are `softplus` (default) or `exp` (potentially unstable).

        Parameters:
            outputs: Model outputs with unrestricted range.

        Returns:
            Result of applying `self.positive_function` to the model outputs and splitting into two
            equal-sized chunks.
        """
        outputs = self.positive_function(outputs)
        alphas, betas = torch.chunk(outputs, chunks=2, dim=1)
        return alphas, betas
