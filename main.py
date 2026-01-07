import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from src.models import CKNSequential
from src.utils import seed_everything, get_stl10_loaders, get_fashion_mnist_loaders, get_synthetic_loaders

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of supervised training on the linear classification head.
    The convolutional filters remain fixed during this phase to adhere to the CKN protocol.

    Args:
        model (nn.Module): The CKN model structure.
        loader (DataLoader): The training data iterator.
        criterion (loss): The loss function (CrossEntropyLoss).
        optimizer (optim): The optimizer (Adam).
        device (torch.device): Computation device (CPU or CUDA).

    Returns:
        tuple: (average_loss, accuracy_percentage)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return running_loss / total, 100. * correct / total

def evaluate(model, loader, criterion, device):
    """
    Evaluates the model performance on a hold-out test set.

    Args:
        model (nn.Module): The trained CKN model.
        loader (DataLoader): The test data iterator.
        criterion (loss): The loss function.
        device (torch.device): Computation device.

    Returns:
        tuple: (average_loss, accuracy_percentage)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / total, 100. * correct / total

def main(dataset_name='stl10', epochs=20):
    """
    Executes the full training pipeline:
    1. Dataset loading and standardization.
    2. Model initialization with architecture adapted to input resolution.
    3. Unsupervised dictionary learning (Spherical K-Means) for feature extraction.
    4. Supervised training of the linear classifier.

    Args:
        dataset_name (str): The target dataset ('stl10', 'fashion', 'synthetic').
        epochs (int): Number of epochs for the supervised optimization phase.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(42)
    print(f"Using device: {device}")
    print(f"Selected Dataset: {dataset_name}")

    learning_rate = 0.001

    if dataset_name == 'stl10':
        train_loader, test_loader, unlabeled_loader, spec = get_stl10_loaders(batch_size=128)
        hidden_channels = [64, 128]
        filter_sizes = [5, 5]
        subsamplings = [2, 2]
        
    elif dataset_name == 'fashion':
        train_loader, test_loader, unlabeled_loader, spec = get_fashion_mnist_loaders(batch_size=128)
        hidden_channels = [64, 128]
        filter_sizes = [3, 3] 
        subsamplings = [2, 2]
        
    elif dataset_name == 'synthetic':
        train_loader, test_loader, unlabeled_loader, spec = get_synthetic_loaders(batch_size=128)
        hidden_channels = [32, 64]
        filter_sizes = [5, 5]
        subsamplings = [2, 2]
        learning_rate = 0.0005 
        
    else:
        raise ValueError("Unknown dataset provided.")

    print("Initializing CKN Model...")
    ckn_model = CKNSequential(
        in_channels=spec.channels,
        hidden_channels_list=hidden_channels,
        filter_sizes=filter_sizes,
        subsamplings=subsamplings,
        image_size=spec.image_size,
        kernel_args_list=[1.0] * len(hidden_channels),
        use_linear_classifier=True,
        out_features=spec.num_classes
    ).to(device)

    print("\n--- Phase 1: Unsupervised Learning (CKN) ---")
    if unlabeled_loader:
        n_patches_max = 50000 
        data_for_unsup = []
        count = 0
        
        for x, _ in unlabeled_loader:
            if count >= 1000: break 
            data_for_unsup.append(x)
            count += x.size(0)
        data_for_unsup = torch.cat(data_for_unsup)[:1000].to(device)

        input_data = data_for_unsup
        for i, layer in enumerate(ckn_model.layers):
            print(f"Training Layer {i+1} Unsupervised...")
            patches = layer.sample_patches(input_data, n_patches=n_patches_max)
            layer.unsup_train(patches)
            
            with torch.no_grad():
                input_data = layer(input_data)
        print("Unsupervised training complete.")

    print(f"\n--- Phase 2: Supervised Training (Linear Head) | LR: {learning_rate} ---")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ckn_model.classifier.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(ckn_model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(ckn_model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CKN on various datasets.")
    parser.add_argument('--dataset', type=str, default='stl10', choices=['stl10', 'fashion', 'synthetic'], 
                        help="Dataset to use for training.")
    parser.add_argument('--epochs', type=int, default=15, 
                        help="Number of epochs for the supervised phase.")
    args = parser.parse_args()
    
    if args.dataset == 'synthetic' and args.epochs == 15:
        args.epochs = 30

    main(dataset_name=args.dataset, epochs=args.epochs)