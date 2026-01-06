import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CKNLayer(nn.Module):
    """
    Implements a Convolutional Kernel Network (CKN) layer.
    
    Reference: Mairal et al., "Convolutional Kernel Networks", NIPS 2014.

    This layer performs:
    1. Unsupervised learning of filters via Spherical K-Means.
    2. Convolution with learned filters.
    3. Non-linear mapping (Gaussian kernel approximation).
    4. Spatial subsampling (Pooling).

    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of filters (dictionary size).
        patch_size (int): Spatial size of the convolution kernel.
        subsampling (int): Spatial subsampling factor.
        kernel_args (float): Kernel parameter (alpha = 1/sigma^2).
        weight (torch.nn.Parameter): The learned filters.
    """
    def __init__(self, in_channels: int, out_channels: int, patch_size: int, 
                 subsampling: int = 2, kernel_args: float = 0.5):
        """
        Initializes the CKNLayer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output filters.
            patch_size (int): Size of the extraction patch.
            subsampling (int, optional): Pooling stride. Default is 2.
            kernel_args (float, optional): Kernel parameter. Default is 0.5.
        """
        super(CKNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.subsampling = subsampling
        self.kernel_args = kernel_args
        
        self.patch_dim = in_channels * patch_size * patch_size
        
        self.weight = nn.Parameter(torch.Tensor(out_channels, self.patch_dim))
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initializes weights with a standard normal distribution.
        """
        nn.init.normal_(self.weight, mean=0.0, std=0.1)

    def sample_patches(self, x: torch.Tensor, n_patches: int) -> torch.Tensor:
        """
        Extracts random patches from the input feature map for dictionary learning.

        Args:
            x (torch.Tensor): Input batch (B, C, H, W).
            n_patches (int): Number of patches to extract.

        Returns:
            torch.Tensor: Extracted patches (n_patches, patch_dim).
        """
        unfolded = F.unfold(x, kernel_size=self.patch_size)
        unfolded = unfolded.transpose(1, 2).contiguous().view(-1, self.patch_dim)
        
        n_total = unfolded.size(0)
        if n_total > n_patches:
            indices = torch.randperm(n_total)[:n_patches]
            return unfolded[indices]
        else:
            return unfolded

    def unsup_train(self, patches: torch.Tensor, n_iter: int = 20):
        """
        Performs Spherical K-Means to learn the filters (centroids).

        Args:
            patches (torch.Tensor): Training patches (N, patch_dim).
            n_iter (int, optional): Number of iterations. Default is 20.
        """
        if not torch.is_tensor(patches):
            patches = torch.from_numpy(patches).float()
            
        device = self.weight.device
        patches = patches.to(device)

        # Normalize patches to unit sphere
        patches_norm = torch.norm(patches, dim=1, keepdim=True)
        patches = patches / (patches_norm + 1e-8)

        # Initialize centroids
        n_samples = patches.size(0)
        indices = torch.randperm(n_samples)[:self.out_channels]
        centroids = patches[indices]

        # Spherical K-Means
        for _ in range(n_iter):
            dots = torch.mm(patches, centroids.t())
            _, labels = dots.max(dim=1)

            one_hot = torch.zeros(n_samples, self.out_channels, device=device)
            one_hot.scatter_(1, labels.view(-1, 1), 1)

            new_centroids = torch.mm(one_hot.t(), patches)

            centroids_norm = torch.norm(new_centroids, dim=1, keepdim=True)
            centroids = new_centroids / (centroids_norm + 1e-8)
            
            # Re-init empty clusters
            mask_empty = (centroids_norm.view(-1) < 1e-8)
            if mask_empty.sum() > 0:
                new_indices = torch.randperm(n_samples)[:mask_empty.sum()]
                centroids[mask_empty] = patches[new_indices]

        self.weight.data.copy_(centroids)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass: Convolution -> Non-Linearity -> Pooling.

        Approximation: exp(alpha * (<w, x> - 1))
        """
        w = F.normalize(self.weight, p=2, dim=1)
        
        # 1. Convolution
        out = F.conv2d(x, w.view(self.out_channels, self.in_channels, self.patch_size, self.patch_size))
        
        # 2. Kernel Activation (Gaussian approximation)
        out = torch.exp(2.0 * self.kernel_args * (out - 1.0))
        
        # 3. Pooling
        if self.subsampling > 1:
            out = F.avg_pool2d(out, kernel_size=self.subsampling, stride=self.subsampling)
            out = out * math.sqrt(self.subsampling * self.subsampling)

        return out

class Linear(nn.Module):
    """
    Wrapper for nn.Linear to handle flattening automatically.
    """
    def __init__(self, in_features: int, out_features: int):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.linear(x)