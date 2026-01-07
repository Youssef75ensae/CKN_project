import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from src.models import CKNLayer
from src.utils import seed_everything, get_stl10_loaders, get_fashion_mnist_loaders, get_synthetic_loaders

def normalize_image(img):
    """
    Normalizes a filter tensor to the range [0, 1] for visualization purposes.
    
    Args:
        img (numpy.ndarray): Input filter array.
        
    Returns:
        numpy.ndarray: Normalized filter ready for plotting.
    """
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    return img

def visualize_filters(dataset_name='stl10', num_filters=64, save_path='learned_filters.png'):
    """
    Instantiates a single CKN layer, trains it unsupervised on the specified dataset,
    and visualizes the learned convolutional filters.

    This function serves to qualitatively assess the features learned by the model
    (e.g., edges, textures) without running the full supervised training pipeline.

    Args:
        dataset_name (str): The target dataset ('stl10', 'fashion', 'synthetic').
        num_filters (int): Number of filters to visualize.
        save_path (str): Filename for the output image.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(42)
    print(f"Visualizing filters for dataset: {dataset_name}")

    if dataset_name == 'stl10':
        train_loader, _, unlabeled_loader, spec = get_stl10_loaders(batch_size=128)
        filter_size = 5
    elif dataset_name == 'fashion':
        train_loader, _, unlabeled_loader, spec = get_fashion_mnist_loaders(batch_size=128)
        filter_size = 3 
    elif dataset_name == 'synthetic':
        train_loader, _, unlabeled_loader, spec = get_synthetic_loaders(batch_size=128)
        filter_size = 5
    else:
        raise ValueError("Unknown dataset")

    layer = CKNLayer(
        in_channels=spec.channels,
        out_channels=num_filters,
        filter_size=filter_size,
        subsampling=1,
        sigma=1.0
    ).to(device)

    print("Collecting patches and training filters...")
    if unlabeled_loader:
        data_iter = iter(unlabeled_loader)
        images, _ = next(data_iter)
        images = images.to(device)
        
        patches = layer.sample_patches(images, n_patches=50000)
        layer.unsup_train(patches, n_iterations=20)
    else:
        print("Error: No unlabeled loader found.")
        return

    filters = layer.weight.data.cpu().numpy()
    
    filters = filters.reshape(num_filters, spec.channels, filter_size, filter_size)

    print(f"Plotting {num_filters} filters...")
    n_cols = 8
    n_rows = num_filters // n_cols + (1 if num_filters % n_cols else 0)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    
    for i, ax in enumerate(axes.flatten()):
        if i < num_filters:
            f = filters[i]
            
            if spec.channels == 1:
                img = f[0]
                img = normalize_image(img)
                ax.imshow(img, cmap='gray', interpolation='nearest')
            else:
                img = np.transpose(f, (1, 2, 0))
                img = normalize_image(img)
                ax.imshow(img, interpolation='nearest')
                
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')
            
    final_filename = f"learned_filters_{dataset_name}.png"
    plt.suptitle(f"Learned CKN Filters - {dataset_name.upper()}", fontsize=16)
    plt.savefig(final_filename, bbox_inches='tight', pad_inches=0.1)
    print(f"Saved visualization to {final_filename}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='stl10', choices=['stl10', 'fashion', 'synthetic'])
    args = parser.parse_args()
    
    visualize_filters(dataset_name=args.dataset)