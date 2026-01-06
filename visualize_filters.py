import torch
import matplotlib.pyplot as plt
import numpy as np
import math
from src.utils import seed_everything, get_stl10_loaders
from src.models import CKNSequential

def visualize_first_layer_filters():
    """
    Orchestrates the visualization of filters learned by the first CKN layer.
    
    This function performs the following steps:
    1. Loads a subset of the STL-10 unsupervised dataset.
    2. Initializes a CKN model (only the first layer is relevant here).
    3. Samples image patches from the data and performs unsupervised learning (Spherical K-Means) 
       to train the first layer's filters.
    4. Extracts the learned weights (centroids) from the layer.
    5. Calls the plotting function to generate and save the visualization.
    """
    # Setup device and seed for reproducibility
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(42)
    print(f"Device: {device}")

    print("Loading data for filter visualization...")
    root_dir = './data'
    
    # We utilize the unlabeled split of STL-10, as the filters are learned without labels
    _, _, unlabeled_loader, spec = get_stl10_loaders(
        root=root_dir, 
        batch_size=128, 
        augment=False, 
        use_unlabeled=True
    )

    # Initialize the CKN architecture
    hidden_channels = [64, 128]
    filter_sizes = [5, 5]
    subsamplings = [2, 2]
    
    model = CKNSequential(
        in_channels=spec.channels,
        hidden_channels_list=hidden_channels,
        filter_sizes=filter_sizes,
        subsamplings=subsamplings,
        image_size=spec.image_size,
        kernel_args_list=[1.0, 1.0],
        use_linear_classifier=False, 
        out_features=10
    )
    model.to(device)

    print("Learning filters for Layer 1 (this should be fast)...")
    
    # Access the first layer directly to perform greedy training
    layer1 = model.layers[0]
    layer1.eval() 
    
    # Configuration for sampling patches
    n_patches = 50000
    collected_patches = []
    count = 0
    
    # Sample patches from the dataset
    with torch.no_grad():
        for x, _ in unlabeled_loader:
            if count >= n_patches: break
            x = x.to(device)
            patches = layer1.sample_patches(x, n_patches=1000)
            collected_patches.append(patches.cpu())
            count += patches.size(0)
    
    # Concatenate and crop to the exact budget
    all_patches = torch.cat(collected_patches, dim=0)[:n_patches]
    
    # Run the unsupervised optimization (Spherical K-Means)
    layer1.unsup_train(all_patches.to(device))
    print("Filters learned.")

    # Extract the learned weights from the layer parameters
    weights = None
    for param in layer1.parameters():
        weights = param.data
        break 
    
    if weights is None:
        print("Error: Could not find weights in Layer 1.")
        return

    print(f"Weights shape: {weights.shape}") # Expected: [64, 75]
    
    # Generate the plot
    # We pass explicit filter dimensions (3 channels, 5x5 kernel) to reshape correctly
    plot_filters(weights.cpu(), in_channels=3, kernel_size=5)

def plot_filters(tensor, in_channels=3, kernel_size=5):
    """
    Visualizes a tensor of convolutional filters as a grid of images.
    
    The input tensor is typically flattened (e.g., [N, C*H*W]). This function reshapes
    it back to [N, C, H, W] before plotting.
    
    Args:
        tensor (torch.Tensor): Tensor representing filters. Shape [N, D] where D = C*H*W.
        in_channels (int): Number of input channels (e.g., 3 for RGB).
        kernel_size (int): Spatial dimension of the filter (e.g., 5 for 5x5).
    """
    # Limit the number of filters to display to keep the plot readable
    num_filters = min(tensor.shape[0], 64)
    tensor = tensor[:num_filters]
    
    # Min-Max normalization to map weights to pixel intensities [0, 1]
    min_val = tensor.min()
    max_val = tensor.max()
    tensor = (tensor - min_val) / (max_val - min_val)
    
    # Calculate grid dimensions
    n_cols = 8
    n_rows = math.ceil(num_filters / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))
    fig.suptitle("Learned Filters - Layer 1 (Unsupervised CKN)", fontsize=16)
    
    for i in range(n_rows * n_cols):
        ax = axes.flat[i]
        if i < num_filters:
            # 1. Get the flattened filter
            flat_filter = tensor[i]
            
            # 2. Reshape to (Channels, Height, Width) -> (3, 5, 5)
            reshaped_filter = flat_filter.view(in_channels, kernel_size, kernel_size)
            
            # 3. Transpose dimensions to (Height, Width, Channels) for Matplotlib
            img = reshaped_filter.permute(1, 2, 0).numpy()
            
            ax.imshow(img)
            ax.axis('off')
        else:
            # Hide unused subplots
            ax.axis('off') 
            
    plt.tight_layout()
    plt.subplots_adjust(top=0.92) 
    plt.savefig('learned_filters.png', dpi=300)
    print("Success: Visualization saved as 'learned_filters.png'")

if __name__ == "__main__":
    visualize_first_layer_filters()