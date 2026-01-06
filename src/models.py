import torch
import torch.nn as nn
from typing import List, Union, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.layers import CKNLayer, Linear

class CKNSequential(nn.Module):
    """
    A sequential container for Convolutional Kernel Networks (CKN).

    Attributes:
        layers (nn.ModuleList): The sequence of CKN layers.
        classifier (nn.Module): The final linear classifier.
    """
    def __init__(
        self, 
        in_channels: int, 
        hidden_channels_list: List[int], 
        filter_sizes: List[int], 
        subsamplings: List[int], 
        image_size: int,
        kernel_args_list: Optional[List[float]] = None,
        use_linear_classifier: bool = True,
        out_features: int = 10
    ):
        """
        Initializes the CKN architecture and dynamically computes the linear layer input size.

        Args:
            in_channels (int): Number of input channels.
            hidden_channels_list (List[int]): Number of filters for each layer.
            filter_sizes (List[int]): Patch size for each layer.
            subsamplings (List[int]): Subsampling factor for each layer.
            image_size (int): Spatial dimension of the input image (required for dimension inference).
            kernel_args_list (List[float], optional): Kernel parameters.
            use_linear_classifier (bool, optional): Add a final Linear layer. Default is True.
            out_features (int, optional): Number of classes.
        """
        super(CKNSequential, self).__init__()
        
        assert len(hidden_channels_list) == len(filter_sizes) == len(subsamplings), \
            "Hyperparameter lists must have the same length."

        self.layers = nn.ModuleList()
        current_channels = in_channels

        # 1. Build CKN Layers
        for i in range(len(hidden_channels_list)):
            kernel_arg = kernel_args_list[i] if kernel_args_list else 0.5
            
            layer = CKNLayer(
                in_channels=current_channels,
                out_channels=hidden_channels_list[i],
                patch_size=filter_sizes[i],
                subsampling=subsamplings[i],
                kernel_args=kernel_arg
            )
            self.layers.append(layer)
            current_channels = hidden_channels_list[i]

        # 2. Compute the Flattened Dimension via a Dummy Pass
        # This ensures the Linear layer matches the spatial output of the CKN stack exactly.
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, image_size, image_size)
            dummy_output = self.features(dummy_input)
            # Flatten dimension: Channels * Height_out * Width_out
            self.flat_features = dummy_output.view(1, -1).size(1)

        # 3. Final Classifier
        self.classifier = None
        if use_linear_classifier:
            self.classifier = Linear(self.flat_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        
        if self.classifier is not None:
            return self.classifier(features)
        
        return features

    def features(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def train_unsupervised(self, dataloader: DataLoader, n_patches: int = 50000, device: str = 'cuda'):
        """
        Performs greedy layer-wise unsupervised training using Spherical K-Means.
        """
        self.to(device)
        self.eval() 

        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                print(f"Training Layer {i+1}/{len(self.layers)} Unsupervised...")
                
                collected_patches = []
                count = 0
                
                pbar = tqdm(total=n_patches, desc=f"Sampling (L{i+1})")
                
                for x, _ in dataloader:
                    if count >= n_patches:
                        break
                    
                    x = x.to(device)
                    
                    # Project data through previous layers
                    for prev_layer_idx in range(i):
                        x = self.layers[prev_layer_idx](x)
                    
                    n_per_batch = 1000 
                    patches = layer.sample_patches(x, n_patches=n_per_batch)
                    
                    collected_patches.append(patches.cpu())
                    count += patches.size(0)
                    pbar.update(patches.size(0))
                
                pbar.close()

                all_patches = torch.cat(collected_patches, dim=0)
                if all_patches.size(0) > n_patches:
                    all_patches = all_patches[:n_patches]
                
                layer.unsup_train(all_patches.to(device))