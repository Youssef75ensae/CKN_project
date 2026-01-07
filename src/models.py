import torch
import torch.nn as nn
import torch.nn.functional as F

class CKNLayer(nn.Module):
    """
    Implements a single layer of a Convolutional Kernel Network (CKN).
    
    This layer performs three main operations:
    1. Patch extraction from the input tensor.
    2. Non-linear mapping using a Gaussian kernel approximation.
    3. Spatial pooling.
    
    Attributes:
        patch_dim (int): Dimensionality of the extracted patches (C * H_k * W_k).
        weight (torch.Tensor): Learnable filters (centroids) for the kernel approximation.
        scale (torch.Tensor): Scaling factor associated with the kernel bandwidth (sigma).
    """
    def __init__(self, in_channels, out_channels, filter_size, subsampling, sigma=1.0):
        super(CKNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_size
        self.subsampling = subsampling
        self.sigma = sigma
        
        self.unfold = nn.Unfold(kernel_size=filter_size, stride=1, padding=filter_size//2)
        
        patch_dim = in_channels * filter_size * filter_size
        self.weight = nn.Parameter(torch.randn(out_channels, patch_dim))
        
        self.register_buffer('scale', torch.tensor(1.0 / (sigma ** 2)))

    def sample_patches(self, x, n_patches=1000):
        """
        Extracts a random subset of patches from the input batch.
        Used for unsupervised dictionary learning.
        
        Args:
            x (torch.Tensor): Input batch of images [B, C, H, W].
            n_patches (int): Number of patches to sample.
            
        Returns:
            torch.Tensor: Sampled patches of shape [n_patches, patch_dim].
        """
        patches = self.unfold(x)  
        patches = patches.permute(0, 2, 1).reshape(-1, patches.size(1))  
        
        if patches.size(0) > n_patches:
            indices = torch.randperm(patches.size(0))[:n_patches]
            patches = patches[indices]
            
        return patches

    def normalize_patches(self, patches):
        """
        Normalizes patches to have unit L2 norm.
        """
        norm = patches.norm(p=2, dim=1, keepdim=True)
        return patches / (norm + 1e-8)

    def unsup_train(self, patches, n_iterations=20):
        """
        Performs Spherical K-Means clustering to learn the layer filters.
        
        Args:
            patches (torch.Tensor): Tensor of sampled patches.
            n_iterations (int): Number of K-Means iterations.
        """
        patches = self.normalize_patches(patches)
        
        n_samples = patches.size(0)
        indices = torch.randperm(n_samples)[:self.out_channels]
        centroids = patches[indices]
        centroids = self.normalize_patches(centroids)
        
        for _ in range(n_iterations):
            dots = torch.matmul(patches, centroids.t())
            labels = dots.argmax(dim=1)
            
            new_centroids = torch.zeros_like(centroids)
            for k in range(self.out_channels):
                mask = (labels == k)
                if mask.sum() > 0:
                    cluster_points = patches[mask]
                    mean_vec = cluster_points.mean(dim=0)
                    new_centroids[k] = mean_vec
                else:
                    new_centroids[k] = patches[torch.randint(0, n_samples, (1,))]
            
            centroids = self.normalize_patches(new_centroids)
            
        self.weight.data = centroids

    def forward(self, x):
        B, C, H, W = x.shape
        patches = self.unfold(x) 
        
        w_norm = F.normalize(self.weight, p=2, dim=1)
        
        patches_t = patches.transpose(1, 2)
        
        patches_norm = torch.norm(patches_t, p=2, dim=2, keepdim=True) + 1e-8
        patches_t = patches_t / patches_norm
        
        out = torch.matmul(patches_t, w_norm.t())
        
        out = torch.exp(self.scale * (out - 1.0))
        
        out = out.transpose(1, 2).view(B, self.out_channels, H, W)
        
        if self.subsampling > 1:
            out = F.avg_pool2d(out, kernel_size=self.subsampling, stride=self.subsampling)
            
        return out

class CKNSequential(nn.Module):
    """
    A sequential model stacking multiple CKN layers.
    Optionally includes a linear classification head for supervised tasks.
    
    This class automatically computes the flattened dimension of the feature maps
    to initialize the linear classifier correctly, regardless of input image size.
    """
    def __init__(self, in_channels, hidden_channels_list, filter_sizes, subsamplings, 
                 image_size, kernel_args_list, use_linear_classifier=False, out_features=10):
        super(CKNSequential, self).__init__()
        
        self.layers = nn.ModuleList()
        current_channels = in_channels
        
        for i, (out_channels, k_size, subsampling, sigma) in enumerate(zip(
            hidden_channels_list, filter_sizes, subsamplings, kernel_args_list)):
            
            layer = CKNLayer(
                in_channels=current_channels,
                out_channels=out_channels,
                filter_size=k_size,
                subsampling=subsampling,
                sigma=sigma
            )
            self.layers.append(layer)
            current_channels = out_channels

        self.use_linear_classifier = use_linear_classifier
        self.classifier = None

        if use_linear_classifier:
            # Dynamic calculation of the flattened dimension
            with torch.no_grad():
                dummy_input = torch.zeros(1, in_channels, image_size, image_size)
                dummy_out = dummy_input
                for layer in self.layers:
                    dummy_out = layer(dummy_out)
                
                flat_dim = dummy_out.view(1, -1).size(1)
            
            self.classifier = nn.Linear(flat_dim, out_features)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        if self.use_linear_classifier:
            x = x.flatten(1)
            x = self.classifier(x)
            
        return x