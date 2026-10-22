"""Various properties of datasets, or ways to compute them.."""
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from lgm.ops import channels_first_to_last, spatial_broadcast_index
from lgm.types import DataBatchFloat, LabelBatch


N_CLASSES = defaultdict(
    int, 
    {"mnist": 10, "cifar10": 10, "fashion": 10, "svhn": 10, "emnist": 47, "cifar100": 100})

CLASS_NAMES = defaultdict(
    list,
    {"mnist": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
     "cifar10": ["airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
     "fashion" : ["t-shirt", "trousers", "pullover", "dress", "coat", "sandal", "shirt", "sneaker",
                  "bag", "boot"],
     "svhn": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
     "emnist": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "c", "D", "E", "F", "G",
                "H", "i", "j", "k", "l", "m", "N", "o", "p", "Q", "R", "s", "T", "u", "v", "w", "x",
                "y", "z", "a", "b", "d", "e", "f", "g", "h", "n", "q", "r", "t"],
     "cifar100": ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
                  'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can',
                  'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud',
                  'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin',
                  'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo',
                  'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
                  'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange',
                  'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain',
                  'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road',
                  'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail',
                  'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
                  'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
                  'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
})


def dataset_mean_std(dataloader: DataLoader[tuple[DataBatchFloat, LabelBatch]],
                     mode: Literal["global", "channel", "dim"] = "global",
                     save_path: str | Path | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes mean and standard deviation for a dataset.

    This uses Welford's online algorithm, or rather the "parallel" version to process batches at
    once, rather than having to iterate over the elements of each batch. See here:
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    
    Parameters:
        dataloader: Loader for the dataset to compute statistics for.
        mode: One of `'global'`, `'channel'`, or `'dim'`.  
            `'global'` computes a single mean and standard deviation value, respectively, over all
            dimensions.  
            `'channel'` computes one value for each channel of the data, so for example for RGB
            color images you would get three means and standard deviations. NOTE that spatial
            dimensions are preserved, so e.g. for CIFAR10 the statistics would be shape
            `(3 x 1 x 1)`.  
            `'dim'` computes one value for each dimension of the data, e.g. each pixel position in
            an image dataset. This requires all data points to have the same size.
        rootdir: Results will be saved to this file. If `None`, results will not be saved.

    Returns:
        Mean and standard deviation values. TODO typing

    Raises:
        ValueError: If not all batches in the dataloader have the same shape (except batch axis), or
            if an invalid `mode` is specified.
    """
    example_data = next(iter(dataloader))[0]
    if mode == "global":
        count = 0.
        mean = 0.
        m2 = 0.
    elif mode == "channel":
        n_channels = example_data.shape[1]
        count = 0.
        mean = torch.zeros(n_channels)
        m2 = torch.zeros(n_channels)
    elif mode == "dim":
        sample_shape = example_data.shape[1:]
        count = 0.
        mean = torch.zeros(sample_shape)
        m2 = torch.zeros(sample_shape)
    else:
        raise ValueError(f"Invalid mode {mode}. Allowed are 'global', 'channel', 'dim'.")
        
    for inputs, _ in dataloader:
        if mode == "global":
            inputs = inputs.view(-1)
        elif mode == "channel":
            n_channels = inputs.shape[1]
            inputs = channels_first_to_last(inputs).reshape(-1, n_channels)
        elif mode == "dim":
            if inputs.shape[1:] != mean.shape:
                raise ValueError(f"Per-dim statistics require all data points to have the same "
                                 f"shape. Got shape {inputs.shape[1:]}, but first batch had shape "
                                 f"{mean.shape}")

        batch_count = inputs.shape[0]
        batch_mean = inputs.mean(dim=0)
        batch_m2 = ((inputs - batch_mean)**2).sum(dim=0)

        delta = batch_mean - mean
        new_count = count + batch_count

        mean = mean + delta * batch_count / new_count
        m2 = (m2
              + batch_m2
              + delta**2 * count * batch_count / new_count)
        count = new_count

    variance = m2 / count
    stddev = torch.sqrt(variance)
    if mode == "channel":
        # slight abuse of this function since statistics don't have batch axis, but it works
        broadcast_index = spatial_broadcast_index(example_data)
        mean = mean[broadcast_index]
        stddev = stddev[broadcast_index]

    if save_path:
        np.savez(save_path, mean=mean.numpy(), stddev=stddev.numpy())
    return mean, stddev
