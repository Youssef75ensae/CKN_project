import time
import torch
from src.utils import seed_everything, get_stl10_loaders
from src.models import CKNSequential
from src.cnn_baseline import CNNBaseline

def main():
    """
    Main execution script comparing CKN (Ours) vs CNN Baseline.
    """
    seed = 42
    batch_size = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    hidden_channels = [64, 128]
    filter_sizes = [5, 5]
    subsamplings = [2, 2]
    
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

    print("\n[Model A] CKN (2-layer)")
    
    kernel_args = [1.0, 1.0]
    n_patches_unsup = 100000
    epochs_ckn = 30
    lr_ckn = 0.01

    ckn_model = CKNSequential(
        in_channels=spec.channels,
        hidden_channels_list=hidden_channels,
        filter_sizes=filter_sizes,
        subsamplings=subsamplings,
        image_size=spec.image_size,
        kernel_args_list=kernel_args,
        use_linear_classifier=True,
        out_features=spec.num_classes
    )
    ckn_model.to(device)

    print("Phase 1: Unsupervised Dictionary Learning")
    t0 = time.time()
    ckn_model.train_unsupervised(unlabeled_loader, n_patches_unsup, device)
    t_ckn_unsup = time.time() - t0
    print(f"Unsupervised time: {t_ckn_unsup:.2f}s")

    print("Phase 2: Supervised Linear Training")
    t0 = time.time()
    ckn_model.train_classifier(train_loader, test_loader, epochs=epochs_ckn, lr=lr_ckn, device=device)
    t_ckn_sup = time.time() - t0
    print(f"CKN Total Runtime: {t_ckn_unsup + t_ckn_sup:.2f}s")

    print("\n[Model B] CNN Baseline")

    epochs_cnn = 30
    lr_cnn = 3e-4
    weight_decay_cnn = 1e-4

    cnn_model = CNNBaseline(
        in_channels=spec.channels,
        hidden_channels_list=hidden_channels,
        kernel_sizes=filter_sizes,
        subsamplings=subsamplings,
        out_features=spec.num_classes
    )
    
    n_params = sum(p.numel() for p in cnn_model.parameters() if p.requires_grad)
    print(f"CNN Trainable Parameters: {n_params/1e6:.2f}M")

    t0 = time.time()
    cnn_model.train_model(
        train_loader, 
        test_loader, 
        epochs=epochs_cnn, 
        lr=lr_cnn, 
        weight_decay=weight_decay_cnn, 
        device=device
    )
    t_cnn_total = time.time() - t0
    print(f"CNN Total Runtime: {t_cnn_total:.2f}s")

    print("Experiment complete.")

if __name__ == "__main__":
    main()