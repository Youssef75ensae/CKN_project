import torch
import numpy as np
import random
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    
    Args:
        seed (int): The seed value to use. Default is 42.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class DatasetSpec:
    """
    Data structure holding dataset metadata.
    
    Attributes:
        channels (int): Number of image channels.
        image_size (int): Height/Width of the images.
        num_classes (int): Number of classification categories.
    """
    def __init__(self, channels, image_size, num_classes):
        self.channels = channels
        self.image_size = image_size
        self.num_classes = num_classes

class SyntheticBars(Dataset):
    """
    Generates a synthetic dataset of images containing either vertical or horizontal bars.
    
    Args:
        size (int): Number of samples to generate.
        img_size (int): Dimension of the square images.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, size=1000, img_size=32, transform=None):
        self.size = size
        self.img_size = img_size
        self.transform = transform
        self.data = []
        self.targets = []
        
        for _ in range(size):
            img = torch.rand(1, img_size, img_size) * 0.2 
            label = random.randint(0, 1)
            
            start = random.randint(5, img_size - 5)
            thickness = random.randint(2, 4)
            
            if label == 0: 
                img[:, :, start:start+thickness] += 0.8
            else: 
                img[:, start:start+thickness, :] += 0.8
                
            self.data.append(torch.clamp(img, 0, 1))
            self.targets.append(label)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, target

def get_stl10_loaders(root='./data', batch_size=128, augment=False, use_unlabeled=True):
    """
    Creates DataLoaders for the STL-10 dataset with standardization.
    
    Args:
        root (str): Directory to store data.
        batch_size (int): Batch size for training and testing.
        augment (bool): Whether to apply random augmentation.
        use_unlabeled (bool): Whether to return a loader for the unlabeled split.
        
    Returns:
        tuple: (train_loader, test_loader, unlabeled_loader, dataset_spec)
    """
    mean = (0.4467, 0.4398, 0.4066)
    std = (0.2603, 0.2566, 0.2713)

    if augment:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(96, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.STL10(root=root, split='train', download=True, transform=transform_train)
    test_set = datasets.STL10(root=root, split='test', download=True, transform=transform_test)
    
    unlabeled_loader = None
    if use_unlabeled:
        unlabeled_set = datasets.STL10(root=root, split='unlabeled', download=True, transform=transform_test)
        unlabeled_loader = DataLoader(unlabeled_set, batch_size=batch_size, shuffle=True, num_workers=2)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    spec = DatasetSpec(channels=3, image_size=96, num_classes=10)
    return train_loader, test_loader, unlabeled_loader, spec

def get_fashion_mnist_loaders(root='./data', batch_size=128):
    """
    Creates DataLoaders for the Fashion-MNIST dataset.
    
    Args:
        root (str): Directory to store data.
        batch_size (int): Batch size.
        
    Returns:
        tuple: (train_loader, test_loader, unlabeled_loader, dataset_spec)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    
    unlabeled_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    spec = DatasetSpec(channels=1, image_size=28, num_classes=10)
    return train_loader, test_loader, unlabeled_loader, spec

def get_synthetic_loaders(batch_size=128):
    """
    Creates DataLoaders for the SyntheticBars dataset with normalization.
    
    Normalization is applied to center the data, which is critical for 
    the convergence of Spherical K-Means.
    
    Args:
        batch_size (int): Batch size.
        
    Returns:
        tuple: (train_loader, test_loader, unlabeled_loader, dataset_spec)
    """
    transform = transforms.Normalize((0.5,), (0.5,))
    
    train_set = SyntheticBars(size=2000, img_size=32, transform=transform)
    test_set = SyntheticBars(size=500, img_size=32, transform=transform)
    
    unlabeled_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    spec = DatasetSpec(channels=1, image_size=32, num_classes=2)
    return train_loader, test_loader, unlabeled_loader, spec