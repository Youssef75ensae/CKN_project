import time
import torch
import pandas as pd
from src.utils import seed_everything, get_stl10_loaders
from src.models import CKNSequential

def train_and_evaluate(config, train_loader, test_loader, unlabeled_loader, device):
    """
    Executes a complete training pipeline (Unsupervised + Supervised) for a specific 
    hyperparameter configuration.
    
    This function initializes a fresh CKN model based on the provided dictionary `config`,
    measures the wall-clock time for both the unsupervised dictionary learning phase
    and the supervised classification phase, and finally computes the test accuracy.
    """
    seed_everything(42)
    
    print(f"\n[Running Config] {config['name']}")
    print(f"Params: Filters={config['hidden_channels']}, Alpha={config['kernel_args']}, "
          f"Patches={config['n_patches']}")

    # Initialize Model with specific config
    model = CKNSequential(
        in_channels=3,
        hidden_channels_list=config['hidden_channels'],
        filter_sizes=[5, 5],
        subsamplings=[2, 2],
        image_size=96,
        kernel_args_list=config['kernel_args'],
        use_linear_classifier=True,
        out_features=10
    )
    model.to(device)

    # Phase 1: Unsupervised Dictionary Learning
    t0 = time.time()
    model.train_unsupervised(
        dataloader=unlabeled_loader, 
        n_patches=config['n_patches'], 
        device=device
    )
    t_unsup = time.time() - t0

    # Phase 2: Supervised Linear Classifier Training
    t0 = time.time()
    model.train_classifier(
        train_loader=train_loader, 
        test_loader=test_loader, 
        epochs=config['epochs'], 
        lr=0.01, 
        device=device
    )
    t_sup = time.time() - t0

    # Final Evaluation
    acc = model.evaluate(test_loader, device)
    
    return {
        "Experiment": config['name'],
        "Accuracy": acc,
        "Time_Unsup": t_unsup,
        "Time_Sup": t_sup,
        "Total_Time": t_unsup + t_sup,
        "Filters": str(config['hidden_channels']),
        "Kernel_Alpha": str(config['kernel_args']),
        "N_Patches": config['n_patches']
    }

def main():
    """
    Orchestrates the ablation study defined in the experimental report.
    
    This function defines a list of experimental configurations targeting three 
    sensitivity axes:
    1. Capacity: Varying the number of filters (dictionary size).
    2. Selectivity: Varying the kernel bandwidth parameter (alpha).
    3. Budget: Varying the number of patches used for K-Means clustering.
    
    Results are aggregated into a DataFrame and saved to 'ablation_results.csv'.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load Data Once
    root_dir = './data'
    train_loader, test_loader, unlabeled_loader, spec = get_stl10_loaders(
        root=root_dir, batch_size=128, augment=True, use_unlabeled=True
    )

    # Define Ablations based on Report Section 4.7
    experiments = [
        # --- Baseline (Reference) ---
        {
            "name": "Baseline",
            "hidden_channels": [64, 128],
            "kernel_args": [1.0, 1.0],
            "n_patches": 100000,
            "epochs": 20
        },
        
        # --- Axis 1: Capacity (Number of Filters) ---
        {
            "name": "Low Capacity",
            "hidden_channels": [32, 64],
            "kernel_args": [1.0, 1.0],
            "n_patches": 100000,
            "epochs": 20
        },
        {
            "name": "High Capacity",
            "hidden_channels": [128, 256],
            "kernel_args": [1.0, 1.0],
            "n_patches": 100000,
            "epochs": 20
        },

        # --- Axis 2: Selectivity (Kernel Alpha) ---
        # Note: Low Alpha corresponds to High Sigma (broad kernel, low selectivity)
        {
            "name": "Low Selectivity (Alpha=0.5)",
            "hidden_channels": [64, 128],
            "kernel_args": [0.5, 0.5],
            "n_patches": 100000,
            "epochs": 20
        },
        # Note: High Alpha corresponds to Low Sigma (sharp kernel, high selectivity)
        {
            "name": "High Selectivity (Alpha=1.5)",
            "hidden_channels": [64, 128],
            "kernel_args": [1.5, 1.5],
            "n_patches": 100000,
            "epochs": 20
        },

        # --- Axis 3: Unsupervised Budget (K-Means samples) ---
        {
            "name": "Low Budget (10k)",
            "hidden_channels": [64, 128],
            "kernel_args": [1.0, 1.0],
            "n_patches": 10000,
            "epochs": 20
        },
        {
            "name": "High Budget (200k)",
            "hidden_channels": [64, 128],
            "kernel_args": [1.0, 1.0],
            "n_patches": 200000,
            "epochs": 20
        },
    ]

    results = []
    
    for config in experiments:
        res = train_and_evaluate(config, train_loader, test_loader, unlabeled_loader, device)
        results.append(res)
        # Save intermediate results to ensure data persistence
        pd.DataFrame(results).to_csv("ablation_results.csv", index=False)
        print(f"-> Result: {res['Accuracy']:.2f}% (Saved)")

    # Display Final Summary Table
    df = pd.DataFrame(results)
    print("\n=== Final Ablation Results ===")
    print(df[['Experiment', 'Accuracy', 'Total_Time']].to_markdown())

if __name__ == "__main__":
    main()