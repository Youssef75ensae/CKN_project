import time
import torch
from src.utils import seed_everything, get_stl10_loaders
from src.models import CKNSequential

def main():
    """
    Main execution script for the CKN project.
    """
    seed = 42
    batch_size = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    hidden_channels = [64, 128]
    filter_sizes = [5, 5]
    subsamplings = [2, 2]
    
    # CORRECTION CRITIQUE ICI :
    # On remonte alpha pour que le noyau soit discriminant.
    # Avec 0.05, exp(alpha * ...) valait toujours 1.
    # Avec 1.0, on aura de la variance.
    kernel_args = [1.0, 1.0]
    
    n_patches_unsup = 100000
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
    print(f"Data loaded. Input: {spec.channels}x{spec.image_size}x{spec.image_size}")

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
    t_unsup = time.time() - t0
    print(f"Phase 1 completed in {t_unsup:.2f}s")

    print("\nPhase 2: Supervised Classifier Training")
    t0 = time.time()
    model.train_classifier(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs_sup,
        lr=lr_sup,
        device=device
    )
    t_sup = time.time() - t0
    print(f"Phase 2 completed in {t_sup:.2f}s")
    print(f"Total Runtime: {t_unsup + t_sup:.2f}s")
    print("Execution finished successfully.")

if __name__ == "__main__":
    main()