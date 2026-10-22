"""Layers & utilities concerned with normalization."""
from typing import Literal, override

from torch import nn

from lgm.types import DataBatchFloat, DataFormat


class DummyNorm(nn.Module):
    """A no-op that just passes the input through unchanged.

    Used out of sheer laziness so we if we want to build a model without normalization, the code
    doesn't need if/else cases every time. Just use `DummyNorm` instead of e.g. `Batchnorm2d` or
    whatever. This guarantees that our models always have normalization layers that can be addressed
    by other code.
    """
    @override
    def __init__(self,
                 *args,
                 **kwargs):
        """Dummy init that ignores arguments. Put in whatever you want!"""
        super().__init__()

    @override
    def forward(self,
                inputs: DataBatchFloat) -> DataBatchFloat:
        return inputs


class BNWrap1d(nn.BatchNorm1d):
    """Dumb wrapper around BatchNorm.
    
    Sometimes we want to use a different normalization instead, like `GroupNormalization`.
    Unfortunately, that one requires an additional argument (number of groups). To avoid having
    different interfaces depending on which normalization we use, we just always pass the `n_groups`
    argument... but then ignore it for `BatchNorm.`

    This is bad! Don't use this as an example of good/smart code! I'm not a software engineer!
    """
    @override
    def __init__(self,
                 n_groups: int,
                 *args,
                 **kwargs):
        """Batchnorm init.

        Parameters:
            n_groups: The parameter we will ignore.
            args: Additional positional arguments to `BatchNorm`
            kwargs: Additional keyword arguments to `BatchNorm`.
        """
        super().__init__(*args, **kwargs)


class BNWrap2d(nn.BatchNorm2d):
    """See `BNWrap1d`, but for 2D `BatchNorm`."""
    @override
    def __init__(self,
                 n_groups: int,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)


class BNWrap3d(nn.BatchNorm3d):
    """See `BNWrap2d`, but for 3D `BatchNorm`."""
    @override
    def __init__(self,
                 n_groups: int,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)


class INWrap1d(nn.InstanceNorm1d):
    """See `BNWrap1d`, but for 1D `InstanceNorm`."""
    @override
    def __init__(self,
                 n_groups: int,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)


class INWrap2d(nn.InstanceNorm2d):
    """See `BNWrap1d`, but for 2D `InstanceNorm`."""
    @override
    def __init__(self,
                 n_groups: int,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)


class INWrap3d(nn.InstanceNorm3d):
    """See `BNWrap1d`, but for 3D `InstanceNorm`."""
    @override
    def __init__(self,
                 n_groups: int,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)


class LNWrap(nn.LayerNorm):
    """See `BNWrap1d`, but for `LayerNorm`."""
    @override
    def __init__(self,
                 n_groups: int,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)


def batchnorm_nd(data_format: DataFormat) -> type[nn.Module]:
    """Simple wrapper to return a `BNWrap` module for the desired dimensionality.
    
    Parameters:
        data_format: One of `'1d'`, `'2d'`, `'3d'`.

    Returns:
        `BNWrap` module type.

    Raises:
        ValueError: For unsupported `data_format`.
    """
    if data_format == "1d":
        return BNWrap1d
    elif data_format == "2d":
        return BNWrap2d
    elif data_format == "3d":
        return BNWrap3d
    else:
        raise ValueError(f"Unsupported data format {data_format}. Supported are '1d', '2d', '3d'.")


def instancenorm_nd(data_format: DataFormat) -> type[nn.Module]:
    """Simple wrapper to return a `INWrap` module for the desired dimensionality.
    
    Parameters:
        data_format: One of `'1d'`, `'2d'`, `'3d'`.

    Returns:
        `INWrap` module type.

    Raises:
        ValueError: For unsupported `data_format`.
    """
    if data_format == "1d":
        return INWrap1d
    elif data_format == "2d":
        return INWrap2d
    elif data_format == "3d":
        return INWrap3d
    else:
        raise ValueError(f"Unsupported data format {data_format}. Supported are '1d', '2d', '3d'.")


def norm_selector(norm: Literal["batch", "group", "layer", "instance", "dummy"],
                  data_format: DataFormat | None = None) -> type[nn.Module]:
    """Select appropriate constructor for a normalization module.
    
    Parameters:
        norm: String identifier for desired normalization. Allowed are `"batch"`, `"group"`,
            `"layer"`, `"instance"`, `"dummy"`.
        data_format: Necessary for batch or instance norm to select the correct module. Ignored for
            other norms.

    Raises:
        ValueError: For unsupported `norm`.
    """
    if norm in ["batch", "instance"] and data_format is None:
        raise ValueError(f"data_format is None, but {norm} norm requested, which requires "
                         "data_format to be set.")
    
    if norm == "batch":
        return batchnorm_nd(data_format)
    elif norm == "group":
        return nn.GroupNorm
    elif norm == "layer":
        return LNWrap
    elif norm == "instance":
        return instancenorm_nd(data_format)
    elif norm == "dummy":
        return DummyNorm
    else:
        raise ValueError(f"Invalid norm {norm}. Allowed are 'batch', 'group', 'layer', 'instance', "
                         "'dummy'.")


def n_groups_convert(n_groups: int | None,
                     n_filters: int) -> int | None:
    """Helper to figure out how many groups we want for `GroupNorm`.
    
    If `n_groups` is a positive number, it will simply be returned. If it is negative, this is
    interpreted as a desired *group size*, so we use `n_filters` to figure out how many groups this
    will come out to be. If `n_groups` is `None`, we just return `None`.

    Parameters:
        n_groups: Perhaps badly named, either the number of groups (if positive) or group size (if
                  negative), `or None`.
        n_filters: Expected input size in terms of number of filters. Needed to compute number of
                   groups from desired group size.

    Returns:
        Number of groups for `GroupNorm`.
    """
    if n_groups is None:
        return None
    elif n_groups < 0:
        group_size = -n_groups
        n_groups = n_filters // group_size
    return n_groups


def is_norm(module: nn.Module) -> bool:
    """Helper to identify whether a module is a normalization layer.
    
    Should also return `True` for the `Wrap` modules we have defined, since they inherit from the
    base `torch` modules. NOTE that this only returns `True` for layers we have explicitly included.
    If you come across and want to use some other normalization module, it will need to be included
    here.

    Parameters:
        module: The `torch` module to check.

    Returns:
        `True` if the module is a normalization, `False` otherwise.
    """
    return (isinstance(module,
                       (nn.GroupNorm,
                        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                        nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                        nn.LayerNorm,
                        DummyNorm)))
