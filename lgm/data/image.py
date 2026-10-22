"""This module mainly implements a data module bundling datasets and loaders for image datasets."""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, override

import numpy as np
import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities import rank_zero_info
from torch.utils.data import DataLoader, default_collate
from torchvision import datasets
from torchvision.transforms.v2 import (CenterCrop, Compose, CutMix, Lambda, MixUp, Pad,
                                       RandomChoice, RandomHorizontalFlip, Resize, ToDtype, ToImage,
                                       Transform)

from lgm.types import ImageBatchFloat, ImageFloat, LabelBatch, ScalarIndex
from .transforms import AdaptiveCenterCrop, Identity, Logit, Normalize, Transpose
from .utils import CLASS_NAMES, N_CLASSES, dataset_mean_std


# NOTE "center" does not require statistics, but we abuse normalize with mean and stddev 0.5
# so it still needs the corresponding code to run
STATISTICS_NORMALIZERS = ["mean_std_global", "mean_std_channel", "mean_std_dim", "center"]


@dataclass(frozen=True)
class DatasetStatistics:
    """Minimalist class storing mean and standard deviation for a dataset."""
    mean: torch.Tensor  # TODO typing
    stddev: torch.Tensor


class ImageDataModule(LightningDataModule):
    """Lightning data module for image datasets.
    
    We provide preprocessing for a number of datasets, mostly through torchvision. More datasets
    could be made available on request.
    """
    @override
    def __init__(self,
                 dataset: str,
                 batch_size: int,
                 num_workers: int = 0,
                 persistent_workers: bool = True,
                 root: str = "data",
                 resize: int | None = None,
                 normalize: Literal["center", "mean_std_global",
                                    "mean_std_channel", "mean_std_dim"] | float | None = None,
                 augment_transforms: list[Transform] | None = None,
                 additional_transforms: list[Transform] | None = None,
                 no_flip: bool = False,
                 cutmixup: tuple[float | None, float | None] |  None = None):
        """Configure the data module.

        Parameters:
            dataset: Name of the dataset. Currently allowed are `"mnist"`, `"emnist"`, `"fashion"`,
                `"svhn"`, `"cifar10"`, `"cifar100"`, `"stl10"`, `"flickr"`, `"celeba"` and
                `"food101"`.
                - mnist: You know; MNIST. Handwritten digits 0-9. Gets padded to 32x32.
                - emnist: Like MNIST, but also includes letters.
                - fashion: FashionMNIST. Like MNIST, but data is fashion items like shirts, shoes,
                  bags...
                - svhn: Street View House Numbers. Not much more difficult than MNIST, but in color!
                - cifar10: Should know this one; 32x32 color images of various animals and vehicles.
                - cifar100: CIFAR but with 100 classes.
                - stl10: 96x96 color images. I don't know how many classes or what kind of objects
                  we have. There are very few labeled images for 10 classes, but many more varied
                  unlabeled images. If 96 pixels is too much, you could add a `Resize` transform to
                  go to 48 or 64. Still more than CIFAR!
                - flickr: Color images of human faces, 128x128 assuming you downloaded the
                  `"thumbnails"` version. NOTE this dataset does come with torchvision.
                  You have to download it here: https://github.com/NVlabs/ffhq-dataset  
                  Be sure to download the thumbnails version!!  
                  There is no explicit train/test separation. The website says there should be
                  70,000 images, but there seem to be fewer, around 65,000. After downloading and
                  unpacking, I propose the following:
                    > Take the last five folders, i.e. named `'65000'` to `'69000'`.
                    > Put them in a separate directory `flickr_test`.
                    > Rename the other folder (originally `thumbnails128x128`) to `flickr_train`.  
                  This gives us around 5000 test images and 60000 training images.  
                  Also NOTE, the dataset will return labels as they are inferred by the folder
                  names. But these are meaningless! This dataset does not have labels.
                - celeba: Faces, but FAMOUS. We crop to square images, as the original ones are not.
                  Since the images are 218x178, this leaves us with 178x178 square images.  
                  NOTE!! Due to the preprocessing of the original dataset, many images contain
                  "smearing" artifacts, usually at the top edge of the image. We try to alleviate
                  this somewhat by cropping to a smaller center frame instead. Unfortunately, this
                  still leaves these artifacts in some cases. Expect your generative model to learn
                  to produce these artifacts, as well.
                - food101: Color images of food items. 101 classes with 1000 images each. Images are
                  large, being 512 pixels on the long side. Aspect ratios are inconsistent, so we
                  center crop each image to a square. Since this results in different-sized images,
                  you **must** resize to a common size to use this dataset.
            batch_size: Guess what!
            num_workers: Used by `DataLoader`.
            persistent_workers: If `True`, use `persistent_workers` for dataloaders. This can
                sometimes cause issues, so we include the option to turn it off.
            root: Base path where datasets should be stored/looked for.
            resize: If given, resize images to this size. Note that you can pass this via
                `additional_transforms`; however, that would be applied after converting to float
                `[0, 1]` images, and it's recommended to handle such transforms *before* float
                conversion.
            normalize: If given, one of `"center"`, `"mean_std_{mode}"`, where `{mode}` can be one
                of the modes seen in the `lgm.data.utils.dataset_mean_std` docs (`"global"`,
                `"channel"`, or `"dim"`), or a float in the open interval (0, 1).  
                `"center"` will result in data scaled to range `[-1, 1]`.  
                `"mean_std_{mode}"` will result in data scaled to mean 0, standard deviation 1.
                See the `dataset_mean_std` docs for the difference between the three modes.  
                These statistics will need to be computed first if they can't be found on disk,
                which can take some time. Future runs will load from disk if possible. NOTE that
                these values are computed on the original data, not taking into account other
                transforms like resizing or padding. Thus, the statistics may be very slightly
                inaccurate for whatever transformed version of the data you are using.  
                A float value `alpha` will first scale data to `[alpha, 1-alpha]`, then apply the
                logit transform. Note that this is a non-linear transformation. It has the effect of
                "stretching out" the value range near the borders 0/1. This area can be problematic
                for likelihood-based models, which would have to learn a sudden drop of likelihood
                to 0 beyond those borders. By stretching out the borders, the drop becomes less
                sudden, which should be easier to model. `alpha` should generally be chosen slightly
                above 0, e.g. 0.05 or below. We got good results using 0.001. This transfrorm should
                only be necessary for specific kinds of models which we will warn you about in
                class, but feel free to experiment with it.
            augment_transforms: Transforms you want to be applied only to the training data, i.e.
                data augmentation. NOTE that most augmentations negatively affect data quality, and
                would make your generative model learn to generate such data (e.g. rotated or
                transposed images). As such, this is only really useful for training classifiers for
                evaluation.
            additional_transforms: Any transforms you want on BOTH training and testing data besides
                standard `ToTensor()` and possibly padding/flipping, or transforms added through
                `resize` or `normalize` arguments.  
                NOTE: These are applied *after* `augment_transforms`.
            no_flip: An exception to the above comment on `augment_transforms` is the random
                horizontal flip transform. This does *not* compromise image quality, and is applied
                *by default* to all training data, unless it doesn't make sense for the dataset,
                e.g. for digits. If you *don't* want data to be flipped randomly for whatever
                reason, you can turn it off by setting this argument to `True`.
            cutmixup: If given, apply cutmix and/or mixup. DO NOT USE for generative models. Only
                used for training classifiers. It's another augmentation, but goes into the
                dataloader, so unfortunately it's a separate argument.  
                Needs to be a tuple of two floats, where the first is the `alpha` for mixup, and
                the second the `alpha` for cutmix. If either entry is `None`, that augmentation
                won't be applied. For each batch, there is a uniform probability that mixup will be
                applied, or cutmix, or neither.
        """
        super().__init__()
        self.save_hyperparameters()
        self.dataset = dataset
        self.root = root

        self.resize = resize
        self.do_normalize = normalize
        self.normalizer = None
        self.augment_transforms = augment_transforms
        self.additional_transforms = additional_transforms
        self.do_flip = not no_flip

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers and num_workers > 0
        self.cutmixup = cutmixup

        self.n_classes = N_CLASSES[dataset]
        self.class_names = CLASS_NAMES[dataset]

        self.prepared = False

    def _make_datasets(self,
                       train_transform: Transform | None = None,
                       test_transform: Transform | None = None,
                       download: bool = False) -> tuple[datasets.VisionDataset,
                                                        datasets.VisionDataset]:
        """Create and return train and test datasets, downloading data if requested.

        NOTE that, since many of the datasets we provide do not come with separate valid/test
        splits, we go the lazy route and just provide train/test splits.  
        NOTE also that not all datasets support downloading. Those datasets will have to be
        downloaded manually beforehand.

        Parameters:
            train_transform: `Transform` to be applied to training data.
            test_transform: `Transform` to be applied to test data.
            download: If `True`, download data if it can't be found in the root folder. If this is
                `False` and the data isn't found, the process will crash. Not all datasets support
                downloading; for those, this argument does nothing.

        Raises:
            ValueError: If `food101` was requested without `resize`, or if an unknown dataset was
                requested.
        """
        if self.dataset == "flickr":
            train_data = datasets.ImageFolder(root=os.path.join(self.root, "flickr_train"),
                                              transform=train_transform)
            test_data = datasets.ImageFolder(root=os.path.join(self.root, "flickr_test"),
                                             transform=test_transform)
        elif self.dataset == "celeba":
            train_data = datasets.CelebA(root=self.root,
                                         split="train",
                                         transform=train_transform,
                                         download=download)
            test_data = datasets.CelebA(root=self.root,
                                        split="test",
                                        transform=test_transform,
                                        download=download)
        elif self.dataset == "svhn":
            train_data_base = datasets.SVHN(root=self.root,
                                            split="train",
                                            transform=train_transform,
                                            download=download)
            extra_data = datasets.SVHN(root=self.root,
                                       split="extra",
                                       transform=train_transform,
                                       download=download)
            train_data = torch.utils.data.ConcatDataset((train_data_base, extra_data))
            test_data = datasets.SVHN(root=self.root,
                                      split="test",
                                      transform=test_transform,
                                      download=download)
        elif self.dataset == "emnist":
            train_data = datasets.EMNIST(root=self.root,
                                         split="balanced",
                                         train=True,
                                         transform=train_transform,
                                         download=download)
            test_data = datasets.EMNIST(root=self.root,
                                        split="balanced",
                                        train=False,
                                        transform=test_transform,
                                        download=download)
        elif self.dataset == "stl10":
            train_data = datasets.STL10(root=self.root,
                                        split="train+unlabeled",
                                        transform=train_transform,
                                        download=download)
            test_data = datasets.STL10(root=self.root,
                                       split="test",
                                       transform=test_transform,
                                       download=download)
        elif self.dataset == "food101":
            if self.resize is None:
                raise ValueError("food101 dataset requires resize argument set to guarantee a "
                                 "common image size.")
            train_data = datasets.Food101(root=self.root,
                                          split="train",
                                          transform=train_transform,
                                          download=download)
            test_data = datasets.Food101(root=self.root,
                                         split="test",
                                         transform=test_transform,
                                         download=download)
        else:
            if self.dataset == "mnist":
                constructor = datasets.MNIST
            elif self.dataset == "fashion":
                constructor = datasets.FashionMNIST
            elif self.dataset == "cifar10":
                constructor = datasets.CIFAR10
            elif self.dataset == "cifar100":
                constructor = datasets.CIFAR100
            else:
                raise ValueError(f"Unknown dataset {self.dataset}!")
            train_data = constructor(root=self.root,
                                     train=True,
                                     transform=train_transform,
                                     download=True)
            test_data = constructor(root=self.root,
                                    train=False,
                                    transform=test_transform,
                                    download=True)

        return train_data, test_data

    def _make_transforms(self) -> tuple[Transform, Transform]:
        """Create all train/test transforms.
        
        There is a number of "always-on" transforms that depend on the dataset. Other than that,
        `augment_transforms` are added to training, `additional_transforms `to both, as is
        normalization. Note that dataset statistics must have already been computed at this point if
        normalization is requested.

        Returns:
            Tuple of train and test transform (bundled via Compose).

        Raises:
            ValueError: If normalization is requested, but parameters have not been computed for
                this dataset, or if an invalid `normalizer` was requested.
        """
        train_transforms = self.necessary_transforms()
        test_transforms = self.necessary_transforms()
    
        if self.augment_transforms is not None:
            train_transforms += self.augment_transforms
        if self.additional_transforms is not None:
            train_transforms += self.additional_transforms
            test_transforms += self.additional_transforms

        if self.do_normalize is not None:
            if self.do_normalize in STATISTICS_NORMALIZERS:
                if not self.prepared:
                    raise ValueError("Cannot create normalization transforms before prepare_data "
                                     "has been called!")
                normalizer = Normalize(self.statistics.mean, self.statistics.stddev)
            elif isinstance(self.do_normalize, float):
                normalizer = Logit(alpha=self.do_normalize)
            else:
                raise ValueError(f"Invalid normalization {self.do_normalize}. Honestly you should "
                                 "never have gotten this far!")
            train_transforms.append(normalizer)
            test_transforms.append(normalizer)
        else:
            normalizer = Identity()
        self.normalizer = normalizer  # saved so plotting methods can access the invert() method
        
        if self.dataset not in ["mnist", "emnist", "fashion", "svhn"] and self.do_flip:
            # mnist/svhn should not be flipped, since flipping digits/letters does not make sense
            train_transforms.append(RandomHorizontalFlip())
            
        return Compose(train_transforms), Compose(test_transforms)

    def necessary_transforms(self) -> list[Transform]:
        """Bundle a bunch of "must-have" transforms for different datasets.
        
        Returns:
            List of transforms (NOT `Compose()`!).
        """
        transforms = [ToImage()]
        if self.dataset in ["mnist", "emnist", "fashion"]:
            # mnist is padded to 32 x 32 for convenience
            transforms.append(Pad(2))
        if self.dataset == "emnist":
            # for some reason EMNIST has height/width transposed...
            transforms.append(Transpose())
        if self.dataset == "celeba":
            transforms.append(CenterCrop(140))
        if self.dataset == "food101":
            transforms.append(AdaptiveCenterCrop())
    
        if self.resize is not None:
            transforms.append(Resize(self.resize))
        transforms.append(ToDtype(torch.float32, scale=True))
        return transforms

    def get_statistics(self,
                       data: datasets.VisionDataset | None = None,
                       mode: Literal["global", "channel", "dim"] | None = None)\
                        -> tuple[torch.Tensor, torch.Tensor]:  # TODO typing
        """Compute statistics over the dataset.

        See `normalize` argument in `__init__` docs for the different "modes".
        
        Parameters:
            data: Optional dataset to use. If `None`, use the `self.train_data_simple` return
                dataset.
            mode: Optional mode to get statistics from. If not given, the default normalization mode 
                or this object is used; if no normalization was requested, this will crash if no
                mode is provided.

        Returns:
            Mean and standard deviation for the desired mode.

        Raises:
            ValueError: If mode is not provided, but the DataModule does not have normalization set.
        """
        if data is None:
            data = self.train_data_simple()
        if mode is None:
            if self.do_normalize is None:
                raise ValueError("get_statistics with mode=None requires normalization to be set " 
                                 "to get a default mode.")
            mode = self.do_normalize.split("_")[-1]

        savedir = Path(self.root) / "statistics"
        os.makedirs(savedir, exist_ok=True)
        save_path = savedir / f"{self.dataset}_{mode}.npz"

        if os.path.exists(save_path):
            rank_zero_info("Loading existing statistics from file...\n")
            mean_std = np.load(save_path)
            mean = torch.tensor(mean_std["mean"], dtype=torch.float32)
            stddev = torch.tensor(mean_std["stddev"], dtype=torch.float32)
        else:
            rank_zero_info("No statistics found; computing...\n")
            statistics_loader = DataLoader(data,
                                           batch_size=self.batch_size,
                                           pin_memory=True,
                                           num_workers=self.num_workers)
            mean, stddev = dataset_mean_std(statistics_loader, mode=mode, save_path=save_path)
        return mean, stddev

    def train_data_simple(self,
                          download: bool = False) -> datasets.VisionDataset:
        """Create and return a basic version of the training data.
        
        This applies only necessary transforms. Can be used as a dummy to guarantee data is
        downloaded, or to compute statistics, where you don't want to tranform the original data, or
        if you want un-augmented training data for whatever reason.

        Parameters:
            download: If `True`, download data if supported. Will crash if this is `False` and data
                has not been downloaded previously. Not all datasets support download; those must be
                manually downloaded in any case.
        """
        transform = Compose(self.necessary_transforms())
        train_data, _ = self._make_datasets(transform,
                                            transform,
                                            download=download)
        return train_data

    @override
    def prepare_data(self):
        train_data = self.train_data_simple(download=True)

        if self.do_normalize is not None:
            if isinstance(self.do_normalize, float):
                if not 0 < self.do_normalize < 1:
                    raise ValueError("Logit transform requires alpha to be in the range (0, 1), "
                                     f"but got {self.do_normalize}")
            elif isinstance(self.do_normalize, str):
                if self.do_normalize.startswith("mean_std"):
                    rank_zero_info("\nMean-Std normalization requested.")
                    mean, stddev = self.get_statistics(train_data)
                elif self.do_normalize == "center":
                    rank_zero_info("\nCentering to range [-1, 1] requested.")
                    mean, stddev = torch.tensor(0.5), torch.tensor(0.5)
                else:
                    raise ValueError(f"Invalid normalization {self.do_normalize}. Allowed are "
                                     "'center', 'mean_std_<mode>', or a float value for logit.")
                self.statistics = DatasetStatistics(mean=mean, stddev=stddev)
            else:
                raise ValueError(f"Invalid normalization {self.do_normalize}. Allowed are "
                                 "'center', 'mean_std_<mode>', or a float value for logit.")

        self.prepared = True

    @override
    def setup(self,
              stage: str | None = None):
        train_transform, test_transform = self._make_transforms()
        self.train_data, self.test_data = self._make_datasets(train_transform,
                                                              test_transform,
                                                              download=False)

    @override
    def train_dataloader(self) -> DataLoader[tuple[ImageBatchFloat, LabelBatch]]:
        if self.cutmixup is not None:
            rank_zero_info("\nWARNING! You have enabled cutmix/mixup. This only makes sense for "
                           "training classifier networks. Hope you know what you are doing!\n")
            transforms = [Lambda(lambda x: x)]
            mixup, cutmix = self.cutmixup
            if mixup is not None:
                transforms.append(MixUp(num_classes=self.n_classes, alpha=mixup))
            if cutmix is not None:
                transforms.append(CutMix(num_classes=self.n_classes, alpha=cutmix))
            cutmixup = RandomChoice(transforms)
    
            def collate_fn(batch: list[tuple[ImageFloat, ScalarIndex]]) -> tuple[ImageBatchFloat,
                                                                                 LabelBatch]:
                return cutmixup(*default_collate(batch))
        else:
            collate_fn = None
            
        # NOTE drop_last=True for training drops the last "leftover" batch if it's smaller than the
        # batch size -> guarantees consistent batch size always
        return DataLoader(self.train_data,
                          batch_size=self.batch_size,
                          pin_memory=True,
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers,
                          drop_last=True,
                          shuffle=True,
                          collate_fn=collate_fn)
    
    @override
    def val_dataloader(self) -> DataLoader[tuple[ImageBatchFloat, LabelBatch]]:
        return self.basic_dataloader(self.test_data)
    
    @override
    def test_dataloader(self) -> DataLoader[tuple[ImageBatchFloat, LabelBatch]]:
        return self.val_dataloader()
    
    @override
    def predict_dataloader(self) -> DataLoader[tuple[ImageBatchFloat, LabelBatch]]:
        return self.val_dataloader()

    def basic_dataloader(self,
                         dataset: datasets.VisionDataset) -> DataLoader[tuple[ImageBatchFloat, LabelBatch]]:
        """Returns a simple no-frills dataloader for the input dataset.

        This dataloader is not shuffled, does not drop the last batch if it's smaller, and uses the
        default collate function.
        
        Parameters:
            dataset: The dataset to get a dataloader for.

        Returns:
            A dataloader for the dataset.
        """
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          pin_memory=True,
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers)
