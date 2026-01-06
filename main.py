import time
import torch
from src.utils import seed_everything, get_stl10_loaders
from src.models import CKNSequential

def main():
    """
    Main execution script for the CKN project.
    Pipeline: Unsupervised Dictionary Learning -> Supervised Linear Classification.
    """
    seed = 42
    batch_size = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    hidden_channels = [64, 128]
    filter_sizes = [6, 3]
    subsamplings = [2, 2]
    
    # Kernel parameters (alpha) reduced to prevent feature saturation
    # Layer 1: 0.5, Layer 2: 0.5
    kernel_args = [0.5, 0.5]
    
    n_patches_unsup = 50000
    epochs_sup = 20
    lr_sup = 0.01

    seed_everything(seed)
    print(f"Device: {device}")

    print("Loading STL-10 dataset...")
    root_dir = './data'
    train_loader, test_loader, unlabeled_loader, spec = get_stl10_loaders(
        root=root_dir, 
        batch_size=batch_size, 
        augment=True,
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

    print("\nPhase 1: Unsupervised Dictionary Learning")
    t0 = time.time()
    model.train_unsupervised(
        dataloader=unlabeled_loader,
        n_patches=n_patches_unsup,
        device=device
    )
    print(f"Phase 1 completed in {time.time() - t0:.2f}s")

    print("\nPhase 2: Supervised Classifier Training")
    t0 = time.time()
    model.train_classifier(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs_sup,
        lr=lr_sup,
        device=device
    )
    print(f"Phase 2 completed in {time.time() - t0:.2f}s")
    print("Execution finished successfully.")

if __name__ == "__main__":
    main()