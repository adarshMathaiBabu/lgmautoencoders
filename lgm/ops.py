"""Small tensor operations that are needed regularly."""
import torch

from lgm.types import (BroadcastDataBatchFloat, DataBatchFloat, DataBatchFloatChannelsLast,
                       DistanceMatrixFloat, TabularBatchFloat, VectorBatchFloat)


def sum_except(inputs: DataBatchFloat,
               keepdim: bool = False) -> VectorBatchFloat | BroadcastDataBatchFloat:
    """Sum over all dimensions of x except the first one.
    
    Parameters:
        inputs: Tensor to sum over.
        keepdim: If True, summed dimensions are preserved as size-1 axes. Otherwise, output is a
            vector.
    """
    return inputs.sum(dim=tuple(range(1, inputs.ndim)), keepdim=keepdim)


def mean_except(inputs: DataBatchFloat,
                keepdim: bool = False) -> VectorBatchFloat | BroadcastDataBatchFloat:
    """It's sum_except, but with the mean. Wow.
    
    Parameters:
        inputs: Tensor to average over.
        keepdim: If True, averaged dimensions are preserved as size-1 axes. Otherwise, output is a
                 vector.
    """
    return inputs.mean(dim=tuple(range(1, inputs.ndim)), keepdim=keepdim)


def batched_multiply(tensor: DataBatchFloat,
                     vector: VectorBatchFloat) -> DataBatchFloat:
    """Multiply a `(B x ...)` tensor with a `(B,)` vector.

    The vector is expanded to the necessary number of dimensions, and then multiplied element-wise
    via broadcasting. A typical use case is having a batch of inputs or activations, and wanting to
    multiply each element (e.g. image) in the batch via a corresponding scalar, which differs per
    batch element.
    
    Parameters:
        tensor: The `(B x ...)` shaped tensor.
        vector: The `B`-element vector (go figure).
    """
    broadcast = (...,) + (torch.newaxis,) * (tensor.ndim - 1)
    return tensor * vector[broadcast]


def spatial_broadcast_index(inputs: DataBatchFloat) -> tuple:
    """Create indexing tuple for broadcasting one tensor to another's shape.
    
    The intended usage is when you have a batch of data, say `(B x C x H x W)` images. You also have
    a `(B x C)` tensor that you would like to broadcast over the spatial dimensions. This can be
    achieved via indexing with `[:, :, None, None]`. However, the number of `None` required depends
    on the dimensionality of the spatial inputs (images, waveforms, etc.). This function returns the
    index given an arbitrarily-shaped input.

    NOTE that it seems like there is no way to type-hint the tuple contents properly -- the first
    element is always ellipses `(...)`, followed by a variable number of `None` (or `torch.newaxis`)
    entries.

    Parameters:
        inputs: The data tensor you want to broadcast to.
    """
    return (...,) + (torch.newaxis,) * (inputs.dim() - 2)


def interleave(inputs: DataBatchFloat,
               other: DataBatchFloat) -> DataBatchFloat:
    """Interleave rows of two tensors of the same shape.
    
    The result will feature rows from inputs and other in an alternating fashion.

    Parameters:
        inputs: First tensor.
        other: Second tensor. Must have same shape as `inputs`.

    Returns:
        Tensor where first row is first row from `inputs`, second row is first row from `other`,
        third row is second row from `inputs`, etc.
    """
    stacked = torch.stack((inputs, other), dim=1)
    return stacked.view(2*inputs.shape[0], *inputs.shape[1:])
    

def channels_first_to_last(inputs: DataBatchFloat) -> DataBatchFloatChannelsLast:
    """Move channels of a batch (of e.g. images) to the last axis.
    
    Parameters:
        inputs: Tensor of shape `(B x C x ...)`.

    Returns:
        Tensor of shape `(B x ... x C)`.
    """
    n_dims = inputs.dim()
    return inputs.permute(0, *range(2, n_dims), 1)


def channels_last_to_first(inputs: DataBatchFloatChannelsLast) -> DataBatchFloat:
    """Move channels of a channels-last batch (of e.g. images) to right after the batch axis.
    
    Parameters:
            inputs: Tensor of shape `(B x ... X C)`.
    
        Returns:
            Tensor of shape `(B x C x ...)`.
    """
    last_dim_index = inputs.dim() - 1
    return inputs.permute(0, last_dim_index, *range(1, last_dim_index))


def squared_distances(row_input: TabularBatchFloat,
                      column_input: TabularBatchFloat) -> DistanceMatrixFloat:
    """Compute squared distances between two batches of vectors.
    
    We make use of `(a - b)**2 == a**2 - 2ab + b**2`, which is faster and less memory-intesive than
    naive broadcasting (or, god forbid two nested loops).

    Parameters:
        row_input: Matrix of shape `(M x C)`.
        column_input: Matrix of shape `(N x C)`.

    Returns:
        Elementwise distance matrix of shape `(M x N)`.
    """
    return ((row_input**2).sum(dim=1)[:, torch.newaxis] 
            - 2 * (row_input @ column_input.T) 
            + (column_input**2).sum(dim=1)[torch.newaxis, :])
