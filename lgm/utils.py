"""Small utility functions that don't fit anywhere else."""
import math
from typing import Literal

import torch
from torch import nn

from lgm.ops import sum_except
from lgm.types import (DataBatchFloat, DataFloat, DataTransform, LabelBatch, ScalarFloat,
                       TabularBatchFloat)


def count_parameters(model: nn.Module) -> int:
    """Get number of (trainable) parameters in a model.
    
    Parameters:
        model: Model to count parameters for.

    Returns:
        Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def interpolate(start: DataFloat,
                end: DataFloat,
                n_interpolations: int,
                style: Literal["linear", "slerp"] = "slerp") -> DataBatchFloat:
    """Compute a series of interpolations between two points.
    
    Parameters:
        start: Starting point for interpolation. Currently does not support batches, i.e. start and
            end have to be single data points *without* batch axis.
        end: End point for interpolation.
        n_interpolations: Will generate this many interpolations, including start and end.
        style: One of "slerp" or "linear". Interpolation style.  
            "linear" simply interpolates in a straight line.  
            "slerp" goes in a circular arc, which can be more appropriate for the unintuitive
            geometry of high-dimensional spaces. This is particularly true when interpolating in
            latent spaces that should be distributed according to normal or uniform distributions,
            for example.

    Returns:
        Batch of data where the first element is input start, last element is input end, and all
        elements inbetween are successively interpolated from start to end.

    Raises:
        ValueError: If invalid `style` is given.
    """
    coefficients = torch.linspace(0, 1, n_interpolations)
    interpolations = []
    for coeff in coefficients:
        if style == "linear":
            interpolations.append(coeff*end + (1 - coeff)*start)
        elif style == "slerp":
            angle = torch.acos(torch.dot(start.view(-1), end.view(-1))
                               / (torch.linalg.vector_norm(start) * torch.linalg.vector_norm(end)))
            interpolations.append(torch.sin((1 - coeff) * angle) * start / torch.sin(angle) 
                                  + torch.sin(coeff * angle) *end / torch.sin(angle))
        else:
            raise ValueError(f"Invalid style {style}. Allowed are 'slerp', 'linear'.")
    return torch.stack(interpolations)


def rescale_likelihood(negative_log_p: ScalarFloat,
                       input_batch: DataBatchFloat,
                       scale_multiplier: float = 1.,
                       log_jacobian_fn: DataTransform | None = None) -> ScalarFloat:
    """Convert negative log-likelihood loss to more interpretable/common form.

    Usually, we compute log probabilities as sums over dimensions, in whatever space the data is in.
    However, literature often reports results in "bits per dim", which is base-2 logarithm
    *averaged* over dimensions.  
    Also, models may be trained on different scalings/normalizations of the same data, which
    significantly affects continuous likelihoods, without meaningfully changing model quality (as
    probabiltiies defined over a larger range are smaller).  
    This function solves these issues by implicitly rescaling data to a desired range, as well as
    converting to base-2 logarithm and averaging over dimensions.
    
    Parameters:
        negative_log_p: Scalar batch average loss, as returned by a `Likelihood.nll()` method.
        input_batch: Batch of inputs that loss was computed on. This is needed if there was a
            `normalizer` transform applied, so the scaling can be inverted, as well as to get the
            number of data dimensions.
        scale_multiplier: Will implicitly scale data from `[0, 1]` to range `[0, scale_multiplier]`.  
            Scaling to a larger range leads to smaller probabilities, as the density has to cover
            more space. For example, pass 256 to scale `[0, 1]` image data to range `[0, 256]`,
            which matches what is usually reported in the literature for image data.  
            If your data is not in `[0, 1]`, you may pass a different value here. For example, if
            your data was centered to `[-1, 1]`, you could pass 128. The fact that the data was also
            shifted is irrelevant. But crucially, any `normalizer` transform in the data can already
            be accounted for separately via `log_jacobian`, so do NOT pass different values here to
            compensate for having added a normalizer -- do it via `log_jacobian`.

            "NOW HOLD UP", you might say, "why 256, and not 255?" After all, the `uint8` max value
            is 255, and that's what we divide by if we scale to `[0, 1]`. However, continuous
            likelihood models like Flows or (most) VAEs are usually trained on *dequantized* data,
            and the standard uniform dequantization adds random uniform values up to 1, then divides
            by 256! So yeah, if you added the `UniformDequantize` transform (as you should!), 256
            would be the correct value here.

    Returns:
        NLL shifted back to original data range (if it was normalized) and then to whatever
        `scale_multiplier` was set, and further converted to bits per dim.
    """
    n_dims = torch.prod(torch.tensor(input_batch.shape[1:]))
    nll = negative_log_p
    if log_jacobian_fn is not None:
        normalizer_log_jacobian = log_jacobian_fn(input_batch)
        nll = nll + sum_except(normalizer_log_jacobian).mean()
    nll = nll / (n_dims * math.log(2)) + math.log2(scale_multiplier)
    return nll


def accuracy(outputs: TabularBatchFloat,
             labels: LabelBatch) -> ScalarFloat:
    """Accuracy for multiclass (softmax) classification.

    Only intended for typical multiclass settings, i.e. softmax classifiers. **Does not work for
    multilabel classification**, where multiple labels can be true for one input (i.e. sigmoid
    outputs).
    
    Parameters:
        outputs: Must be 2D, `(batch x n_classes)`.
        labels: Can be given in sparse (index) format, or as one-hot. That is, can be either a
            vector with one index entry per batch element, or a matrix with the same shape as
            `outputs`.

    Returns:
        Scalar accuracy.
    """
    return top_k_accuracy(outputs, labels, k=1)


def top_k_accuracy(outputs: TabularBatchFloat,
                   labels: LabelBatch,
                   k: int) -> ScalarFloat:
    """Top-k accuracy. Counts a hit if one of the k largest outputs matches the label.
    
    Parameters:
        outputs: See `accuracy()`.
        labels: See `accuracy()`.
        k: It's the k in top-k!

    Returns:
        Scalar top-k accuracy.
    """
    top_k_predictions = torch.topk(outputs, k, dim=1, sorted=False).indices
    if labels.ndim == 2:
        labels = labels.argmax(dim=1)
    labels = labels[:, None]
    matches = torch.any(labels == top_k_predictions, dim=1)
    return matches.float().mean()
