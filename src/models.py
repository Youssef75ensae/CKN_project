import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.layers import CKNLayer, Linear

class CKNSequential(nn.Module):
    """
    Implements a Convolutional Kernel Network (CKN) architecture for image classification.

    This module acts as a sequential container that stacks multiple `CKNLayer` instances
    followed by a linear classification head. It implements the hybrid training protocol
    proposed by Mairal et al. (2014):
    
    1.  Unsupervised Pre-trainin: The convolutional filters are learned layer by layer
        using a Spherical K-Means algorithm on extracted patches. This approximates the
        geometry of the data manifold without using labels.
    
    2.  Supervised Fine-tuning: Once the filters are fixed, the resulting feature map
        is flattened, normalized via Batch Normalization, and passed to a linear classifier
        trained via Stochastic Gradient Descent (SGD) using class labels.

    Attributes:
        layers (nn.ModuleList): The sequence of unsupervised CKN layers.
        classifier_head (nn.Sequential): The supervised component (BatchNorm + Linear).
        flat_features (int): The dimensionality of the feature vector before classification.
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
        Initializes the CKN architecture and dynamically infers the feature dimension.

        Args:
            in_channels (int): Number of input image channels (e.g., 3 for RGB).
            hidden_channels_list (List[int]): Number of filters (dictionary size) for each CKN layer.
            filter_sizes (List[int]): Spatial size of the patch extraction for each layer.
            subsamplings (List[int]): Pooling factor (stride) for each layer.
            image_size (int): Height/Width of the input images. Used to calculate the final feature vector size.
            kernel_args_list (List[float], optional): Kernel parameters (alpha) for the Gaussian approximation.
            use_linear_classifier (bool): Whether to append the classification head.
            out_features (int): Number of target classes.
        """
        super(CKNSequential, self).__init__()
        
        # Validation of hyperparameter consistency
        assert len(hidden_channels_list) == len(filter_sizes) == len(subsamplings), \
            "Hyperparameter lists (channels, sizes, subsamplings) must have the same length."

        self.layers = nn.ModuleList()
        current_channels = in_channels

        # Construction of the unsupervised convolutional layers
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

        # Dynamic inference of the output dimension
        # We perform a dummy forward pass to determine the exact size of the flattened feature map.
        # This avoids manual calculation errors related to padding/strides.
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, image_size, image_size)
            dummy_output = self.features(dummy_input)
            self.flat_features = dummy_output.view(1, -1).size(1)

        # Construction of the supervised classifier
        # Batch Normalization is crucial here to scale the kernel features before the linear layer.
        self.classifier_head = None
        if use_linear_classifier:
            self.classifier_head = nn.Sequential(
                nn.BatchNorm1d(self.flat_features),
                nn.Linear(self.flat_features, out_features)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the computation performed at every call.
        """
        features = self.features(x)
        features = features.view(features.size(0), -1)
        
        if self.classifier_head is not None:
            return self.classifier_head(features)
        return features

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagates the input through the sequence of CKN layers to extract features.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def train_unsupervised(self, dataloader: DataLoader, n_patches: int = 50000, device: str = 'cuda'):
        """
        Executes the greedy layer-wise unsupervised pre-training.

        For each layer, the method:
        1. Projects the data through the previously trained layers.
        2. Samples random patches from the resulting feature maps.
        3. Performs Spherical K-Means to learn the layer's filters (centroids).
        """
        self.to(device)
        self.eval() 

        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                print(f"Training Layer {i+1}/{len(self.layers)} Unsupervised...")
                
                collected_patches = []
                count = 0
                
                # We limit the sampling batch size to avoid OOM errors
                pbar = tqdm(total=n_patches, desc=f"Sampling (L{i+1})")
                for x, _ in dataloader:
                    if count >= n_patches: break
                    x = x.to(device)
                    
                    # Forward pass through frozen previous layers
                    for prev_layer_idx in range(i):
                        x = self.layers[prev_layer_idx](x)
                    
                    patches = layer.sample_patches(x, n_patches=1000)
                    collected_patches.append(patches.cpu())
                    count += patches.size(0)
                    pbar.update(patches.size(0))
                pbar.close()

                # Concatenate and crop to exact number of required patches
                all_patches = torch.cat(collected_patches, dim=0)
                if all_patches.size(0) > n_patches:
                    all_patches = all_patches[:n_patches]
                
                layer.unsup_train(all_patches.to(device))

    def train_classifier(self, train_loader: DataLoader, test_loader: DataLoader, 
                         epochs: int = 20, lr: float = 0.01, device: str = 'cuda'):
        """
        Trains the final linear classifier in a supervised manner.

        Crucially, this method freezes the parameters of the CKN layers to preserve
        the structure learned during the unsupervised phase. Only the BatchNorm
        statistics and the Linear weights are updated.
        """
        if self.classifier_head is None:
            raise ValueError("Model has no classifier head to train.")

        self.to(device)
        
        # Freezing CKN layers
        for layer in self.layers:
            for param in layer.parameters():
                param.requires_grad = False
        
        # Optimizer targets only the classifier head
        optimizer = optim.Adam(self.classifier_head.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        print(f"\nStarting Supervised Training (LR={lr})...")

        # Initial Feature Statistics Check (Debugging)
        first_batch, _ = next(iter(train_loader))
        with torch.no_grad():
            feats = self.features(first_batch.to(device)).view(first_batch.size(0), -1)
            print(f"DEBUG: Feature Statistics - Mean: {feats.mean().item():.4f}, Std: {feats.std().item():.4f}")
            if feats.std().item() < 1e-6:
                print("WARNING: Low feature variance detected. Model may fail to converge.")

        # Training Loop
        for epoch in range(epochs):
            self.train() 
            
            total_loss = 0
            correct = 0
            total = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            train_acc = 100. * correct / total
            test_acc = self.evaluate(test_loader, device)
            
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/total:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    def evaluate(self, dataloader: DataLoader, device: str = 'cuda') -> float:
        """
        Computes the accuracy of the model on the provided dataset.
        """
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return 100. * correct / total