import time
import torch
from src.utils import seed_everything, get_stl10_loaders
from src.models import CKNSequential

def main():
    """
    Main execution entry point for the CKN project.

    This script performs the following operations:
    1. Configures the computing device and random seeds for reproducibility.
    2. Loads the STL-10 dataset (unlabeled split).
    3. Initializes a 2-layer CKNSequential model.
    4. Executes greedy layer-wise unsupervised pre-training.
    5. Validates the architecture integrity with a forward pass.
    """
    seed = 42
    batch_size = 128
    n_patches_training = 50000
    
    hidden_channels = [64, 128]
    filter_sizes = [6, 3]
    subsamplings = [2, 2]
    kernel_args = [0.5, 0.5]

    seed_everything(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading STL-10 dataset...")
    root_dir = './data'
    _, _, unlabeled_loader, spec = get_stl10_loaders(
        root=root_dir, 
        batch_size=batch_size, 
        augment=False,
        use_unlabeled=True
    )
    print(f"Data loaded. Input shape: {spec.channels}x{spec.image_size}x{spec.image_size}")

    print("Initializing CKN model...")
    model = CKNSequential(
        in_channels=spec.channels,
        hidden_channels_list=hidden_channels,
        filter_sizes=filter_sizes,
        subsamplings=subsamplings,
        image_size=spec.image_size,
        kernel_args_list=kernel_args,
        use_linear_classifier=True,
        out_features=spec.num_classes
    )
    model.to(device)

    print("Starting unsupervised pre-training...")
    t0 = time.time()
    
    model.train_unsupervised(
        dataloader=unlabeled_loader,
        n_patches=n_patches_training,
        device=device
    )
    
    duration = time.time() - t0
    print(f"Pre-training completed in {duration:.2f}s")

    print("Verifying forward pass...")
    dummy_input = torch.randn(2, spec.channels, spec.image_size, spec.image_size).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Execution finished successfully.")

if __name__ == "__main__":
    main()