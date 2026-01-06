import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CKNLayer(nn.Module):
    """
    Implements a Convolutional Kernel Network (CKN) layer with strict patch normalization.
    
    To ensure consistency with Spherical K-Means, input patches must be projected onto
    the unit sphere during the forward pass. This is achieved by dividing the convolution
    output by the L2-norm of the input patches.

    Attributes:
        in_channels (int): Input channels.
        out_channels (int): Output filters.
        patch_size (int): Kernel spatial size.
        subsampling (int): Pooling stride.
        kernel_args (float): Gaussian kernel parameter (alpha).
        weight (nn.Parameter): Learned filters (centroids).
    """
    def __init__(self, in_channels: int, out_channels: int, patch_size: int, 
                 subsampling: int = 2, kernel_args: float = 0.5):
        super(CKNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.subsampling = subsampling
        self.kernel_args = kernel_args
        
        self.patch_dim = in_channels * patch_size * patch_size
        
        # Filters (Centroids)
        self.weight = nn.Parameter(torch.Tensor(out_channels, self.patch_dim))
        
        # Fixed kernel for computing local patch norms (sum of squares)
        # Shape: (1, in_channels, k, k) filled with 1.0
        self.register_buffer('ones_kernel', torch.ones(1, in_channels, patch_size, patch_size))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.1)

    def sample_patches(self, x: torch.Tensor, n_patches: int) -> torch.Tensor:
        """Extracts and normalizes patches for unsupervised training."""
        unfolded = F.unfold(x, kernel_size=self.patch_size)
        unfolded = unfolded.transpose(1, 2).contiguous().view(-1, self.patch_dim)
        
        if unfolded.size(0) > n_patches:
            indices = torch.randperm(unfolded.size(0))[:n_patches]
            unfolded = unfolded[indices]
            
        return unfolded

    def unsup_train(self, patches: torch.Tensor, n_iter: int = 20):
        """Spherical K-Means training."""
        if not torch.is_tensor(patches):
            patches = torch.from_numpy(patches).float()
        
        device = self.weight.device
        patches = patches.to(device)

        # Normalize patches (Project to Sphere)
        patches = F.normalize(patches, p=2, dim=1)

        # Initialize centroids
        n_samples = patches.size(0)
        indices = torch.randperm(n_samples)[:self.out_channels]
        centroids = patches[indices]

        for _ in range(n_iter):
            dots = torch.mm(patches, centroids.t())
            _, labels = dots.max(dim=1)

            one_hot = torch.zeros(n_samples, self.out_channels, device=device)
            one_hot.scatter_(1, labels.view(-1, 1), 1)

            new_centroids = torch.mm(one_hot.t(), patches)
            centroids = F.normalize(new_centroids, p=2, dim=1)
            
            # Handle empty clusters
            mask_empty = (torch.norm(new_centroids, dim=1) < 1e-8)
            if mask_empty.sum() > 0:
                new_indices = torch.randperm(n_samples)[:mask_empty.sum()]
                centroids[mask_empty] = patches[new_indices]

        self.weight.data.copy_(centroids)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with patch normalization.
        Computes: exp(alpha * ( <w, x>/||x|| - 1 ))
        """
        # 1. Normalize Filters (Project to Sphere)
        w = F.normalize(self.weight, p=2, dim=1)
        w_spatial = w.view(self.out_channels, self.in_channels, self.patch_size, self.patch_size)
        
        # 2. Compute Patch Norms ||x|| efficiently via Convolution
        # ||x||^2 = sum(x^2) over the patch
        x_sq = x.pow(2)
        patch_norms_sq = F.conv2d(x_sq, self.ones_kernel, stride=1)
        patch_norms = torch.sqrt(patch_norms_sq + 1e-6) # Avoid division by zero
        
        # 3. Convolution (Dot Product <w, x>)
        dot_products = F.conv2d(x, w_spatial, stride=1)
        
        # 4. Cosine Similarity (<w, x> / ||x||)
        # We divide the dot product by the patch norm to get cosine similarity in [-1, 1]
        cosine_sim = dot_products / patch_norms
        
        # 5. Kernel Activation (Gaussian approximation)
        # exp( alpha * (cosine - 1) ) -> Ranges from 0 to 1. No explosion possible.
        out = torch.exp(2.0 * self.kernel_args * (cosine_sim - 1.0))
        
        # 6. Pooling
        if self.subsampling > 1:
            out = F.avg_pool2d(out, kernel_size=self.subsampling, stride=self.subsampling)
            out = out * math.sqrt(self.subsampling * self.subsampling)

        return out

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.linear(x)