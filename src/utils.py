"""
Utilities for Convolutional Kernel Networks (CKN).
Includes:
1. Reproducibility setup (Seeds).
2. Data Loading (STL-10 & FashionMNIST).
3. Patch Extraction (Sampling small squares from images).
4. Preprocessing (Standardization & Whitening).

Reference: Mairal et al. (2014), Convolutional Kernel Networks.
"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataclasses import dataclass
from typing import Tuple, Optional

def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Default is 42.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class DataSpec:
    """
    Data specification container.

    Attributes:
        num_classes (int): Number of classes in the dataset.
        channels (int): Number of image channels.
        image_size (int): Height/Width of the image (assumed square).
    """
    num_classes: int
    channels: int
    image_size: int

def get_stl10_loaders(
    root: str,
    batch_size: int,
    num_workers: int = 2,
    augment: bool = True,
    use_unlabeled: bool = False
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], DataSpec]:
    """
    Prepares DataLoaders for the STL-10 dataset.

    Args:
        root (str): Path to the data directory.
        batch_size (int): Number of samples per batch.
        num_workers (int, optional): Number of subprocesses for data loading. Default is 2.
        augment (bool, optional): Whether to apply data augmentation on training set. Default is True.
        use_unlabeled (bool, optional): Whether to load the unlabeled split. Default is False.

    Returns:
        Tuple[DataLoader, DataLoader, Optional[DataLoader], DataSpec]: 
            Returns (train_loader, test_loader, unlabeled_loader, data_spec).
            unlabeled_loader is None if use_unlabeled is False.
    """
    mean = (0.4467, 0.4398, 0.4066)
    std = (0.2603, 0.2566, 0.2713)

    base_transform = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    
    if augment:
        train_tf = transforms.Compose([
            transforms.RandomCrop(96, padding=4),
            transforms.RandomHorizontalFlip(),
        ] + base_transform)
    else:
        train_tf = transforms.Compose(base_transform)

    test_tf = transforms.Compose(base_transform)

    train_set = datasets.STL10(root=root, split='train', download=True, transform=train_tf)
    test_set = datasets.STL10(root=root, split='test', download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    unlabeled_loader = None
    if use_unlabeled:
        unlabeled_set = datasets.STL10(root=root, split='unlabeled', download=True, transform=test_tf)
        unlabeled_loader = DataLoader(unlabeled_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    spec = DataSpec(num_classes=10, channels=3, image_size=96)
    
    return train_loader, test_loader, unlabeled_loader, spec

class PatchExtractor:
    """
    Extracts random patches from a dataset for unsupervised dictionary learning.

    Attributes:
        patch_size (int): Spatial dimension of the patches (square).
        total_patches (int): Target number of patches to extract.
    """
    def __init__(self, patch_size: int, total_patches: int = 100000):
        """
        Initializes the PatchExtractor.

        Args:
            patch_size (int): Size of the extraction patch.
            total_patches (int, optional): Total number of patches to collect. Default is 100000.
        """
        self.patch_size = patch_size
        self.total_patches = total_patches

    def sample(self, loader: DataLoader, device='cpu') -> torch.Tensor:
        """
        Performs the extraction of random patches from the provided DataLoader.

        Args:
            loader (DataLoader): The data source.
            device (str, optional): Device for computation. Default is 'cpu'.

        Returns:
            torch.Tensor: A tensor of shape (total_patches, channels * patch_size^2).
        """
        patches = []
        count = 0
        
        for data, _ in loader:
            if count >= self.total_patches:
                break
            
            data = data.to(device)
            unfolded = F.unfold(data, kernel_size=self.patch_size)
            unfolded = unfolded.permute(0, 2, 1).contiguous().view(-1, unfolded.shape[1])
            
            n_samples_batch = min(1000, unfolded.shape[0])
            indices = torch.randperm(unfolded.shape[0])[:n_samples_batch]
            
            patches.append(unfolded[indices].cpu())
            count += indices.shape[0]

        all_patches = torch.cat(patches, dim=0)
        if all_patches.shape[0] > self.total_patches:
            all_patches = all_patches[:self.total_patches]
            
        return all_patches

class Preprocessing:
    """
    Handles preprocessing of image patches, specifically centering and ZCA whitening.

    Attributes:
        mean (torch.Tensor): Mean vector of the training data.
        whitening_matrix (torch.Tensor): The computed whitening matrix.
        reg (float): Regularization parameter for ZCA whitening.
    """
    def __init__(self, whitening_reg=0.001):
        """
        Initializes the Preprocessing module.

        Args:
            whitening_reg (float, optional): Regularization epsilon. Default is 0.001.
        """
        self.mean = None
        self.whitening_matrix = None
        self.reg = whitening_reg

    def fit(self, patches: torch.Tensor):
        """
        Computes the mean and the whitening matrix from the input patches.

        Args:
            patches (torch.Tensor): Input data of shape (N, dim).
        """
        self.mean = patches.mean(dim=0)
        patches_centered = patches - self.mean

        cov = torch.mm(patches_centered.t(), patches_centered) / (patches_centered.shape[0] - 1)
        U, S, _ = torch.svd(cov)
        
        inv_sqrt_S = torch.diag(1.0 / torch.sqrt(S + self.reg))
        self.whitening_matrix = torch.mm(torch.mm(U, inv_sqrt_S), U.t())

    def transform(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Applies centering and whitening to the input data.

        Args:
            patches (torch.Tensor): Input data to transform.

        Returns:
            torch.Tensor: Whitened data.

        Raises:
            RuntimeError: If fit() has not been called previously.
        """
        if self.mean is None or self.whitening_matrix is None:
            raise RuntimeError("Preprocessing must be fit first")
        
        patches_centered = patches - self.mean
        return torch.mm(patches_centered, self.whitening_matrix)