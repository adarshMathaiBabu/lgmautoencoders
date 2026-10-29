"""Various transforms that are not available in torchvision."""
import math
from abc import ABC, abstractmethod
from typing import Any, override

import torch
from torchvision.transforms.v2 import Transform
from torchvision.transforms.v2.functional import center_crop

from lgm.types import DataBatchFloat, DataFloat, ImageFloat


# TODO most of the typing is wrong, as batches are often also ok


### ===========================================================================
### Shape-preserving Transforms that only change the values
### ===========================================================================
class Binarize(Transform):
    """Turn inputs into binary numbers according to a threshold."""
    @override
    def __init__(self,
                 threshold: float = 0.5):
        """Configure the transform.

        Parameters:
            threshold: Everything above this value will become 1; everything else 0.
        """
        super().__init__()
        self.threshold = threshold

    @override
    def transform(self,
                  inputs: DataFloat,
                  _params: Any) -> DataFloat:
        return torch.where(inputs > self.threshold, 1., 0.)


class InvertibleTransform(Transform, ABC):
    """Base class for transforms that can be inverted.
    
    Mainly used for normalization. If a model was trained on normalized data, it will also generate
    normalized data. But in order to plot the results properly, we have to scale back to the
    original range. We also implement the log jacobian of the (inverse) transformation for use in
    likelihood-based models.
    """
    @abstractmethod
    def invert(self,
               inputs: DataFloat) -> DataFloat:
        """Invert normalization on normalized data."""

    @abstractmethod
    def log_inverse_jacobian(self,
                             inputs: DataBatchFloat) -> DataBatchFloat:
        """Get the (sort of) log Jacobian of the inverse transform on normalized data.
        
        This will generally return a tensor with the same shape as `inputs`, where each element is
        the log of the partial derivative of the transform wrt to that respective dimension. This
        only really makes sense for element-wise transforms.  
        The usual application is to sum up these values over the data dimensions to get the log of
        the Jacobian determinant. Truth be told, a "log Jacobian" isn't really a thing.

        TODO clear this up.
        """


class Identity(InvertibleTransform):
    """Dummy transform that does nothing.
    
    Mainly used so that the data module always has a `normalizer.invert` defined, preventing a bunch
    of if/else handling in other classes that need to use that. If we are not normalizing, this just
    does a dummy "inversion", and the "customer" classes can always rely on `normalizer` being
    there.
    """
    @override
    def transform(self,
                    inputs: DataFloat,
                    _params: Any) -> DataFloat:
        return inputs

    @override
    def invert(self,
                inputs: DataFloat) -> DataFloat:
        return inputs

    @override
    def log_inverse_jacobian(self,
                             inputs: DataBatchFloat) -> DataBatchFloat:
        return torch.zeros_like(inputs)


class Normalize(InvertibleTransform):
    """Normalize inputs by a given mean and standard deviation.

    There is already such a transform in `torchvision`, but that seems to be a bit restrictive with
    regards to what shapes it accepts for the parameters. This one should be more flexible. That is,
    you could even use one value per pixel in an image, such that each *position* gets normalized
    independently. We also support "de-normalizing" by applying the inverse transformation.

    TODO typing for mean/stddev
    """
    @override
    def __init__(self,
                 mean: torch.Tensor,
                 stddev: torch.Tensor):
        """Configure the transform.
        
        Parameters:
            mean: Value to subtract from all data points.
            stddev: Value to divide by after mean has been subtracted.
        """
        super().__init__()
        self.mean = mean
        self.stddev = stddev

    @override
    def transform(self,
                  inputs: DataFloat,
                  _params: Any) -> DataFloat:
        return (inputs - self.mean) / self.stddev

    @override
    def invert(self,
               inputs: DataFloat) -> DataFloat:
        return self.stddev*inputs + self.mean

    @override
    def log_inverse_jacobian(self,
                             inputs: DataBatchFloat) -> DataBatchFloat:
        return torch.broadcast_to(torch.log(self.stddev), inputs.shape)


class Logit(InvertibleTransform):
    @override
    def __init__(self,
                 alpha: float = 0.001):
        """Transform inputs in [0, 1] to logit space.
        
        Parameters:
            alpha: Should be a small number to pull inputs away from 0 or 1, since the logit
                function is infinite for those values.
        """
        super().__init__()
        self.alpha = alpha

    @override
    def transform(self,
                  inputs: ImageFloat,
                  _params: Any) -> ImageFloat:
        squeezed = self.alpha + (1 - 2*self.alpha) * inputs
        return torch.logit(squeezed)

    @override
    def invert(self,
               inputs: ImageFloat) -> ImageFloat:
        squeezed = torch.nn.functional.sigmoid(inputs)
        return (squeezed - self.alpha) / (1 - 2*self.alpha)

    @override
    def log_inverse_jacobian(self,
                             inputs: DataBatchFloat) -> DataBatchFloat:
        return (torch.nn.functional.logsigmoid(inputs)
                + torch.nn.functional.logsigmoid(-inputs)
                - math.log(1 - 2*self.alpha))

    
class UniformDequantize(Transform):
    """Dequantize 8bit images by "filling in" the space between the integer values.
    
    We assume the images were already normalized to range `[0, 1]` beforehand, simply because it
    matches the way the preprocessing is implemented in out repository. This WILL NOT BE CORRECT if
    they are not in that range.
    
    There isn't one correct way to do this. We randomly add up to 1 pixel value, and then
    re-normalize such that the maximum is 1. This is the accepted version in the literature. Another
    version would be to add or subtract up to 0.5 pixel values (normalized to the correct range) and
    then clip to `[0, 1]`. However, this has the issue of too many values at the edges due to the
    clipping.
    """
    @override
    def transform(self,
                  inputs: DataFloat,
                  _params: Any) -> DataFloat:
        return (inputs*255 + torch.rand_like(inputs)) / 256


### ===========================================================================
### Transforms that aim at changing the values, but also change shape as a side effect
### ===========================================================================
class ToGrayscale(Transform):
    """Convert color image(s) to grayscale.

    We count axes from the back so it should work for both batches or unbatched single images. The
    grayscale conversion is "naive" in that we simply use the mean of the RGB channels. There are
    other, more sophisticated methods that properly respect the differences in luminance between the
    different colors.
    """
    @override
    def transform(self,
                  inputs: ImageFloat,
                  _params: Any) -> ImageFloat:
        return inputs.mean(dim=-3, keepdim=True)


### ===========================================================================
### Transforms that aim at changing shape, but otherwise leave values untouched
### ===========================================================================
class Transpose(Transform):
    """Flip height/width dimensions for images."""
    @override
    def transform(self,
                  inputs: ImageFloat,
                  _params: Any) -> ImageFloat:
        return torch.permute(inputs, (-3, -1, -2))


class AdaptiveCenterCrop(Transform):
    """Center crop an image to whatever the smallest side length is."""
    @override
    def transform(self,
                  inputs: ImageFloat,
                  _params: Any) -> ImageFloat:
        min_size = min(inputs.shape[-2:])
        return center_crop(inputs, (min_size,))
 