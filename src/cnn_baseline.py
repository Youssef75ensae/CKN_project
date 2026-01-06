import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class CNNBaseline(nn.Module):
    """
    Standard CNN architecture for comparison with CKN.
    
    This model implements the classical design pattern described by Aurelien Geron (Chap. 14):
    a stack of convolutional blocks (Convolution -> ReLU -> Pooling) that progressively 
    reduce spatial resolution while increasing depth. It concludes with a Global Average 
    Pooling layer to minimize parameters before the final linear classifier.
    """
    def __init__(self, in_channels: int, hidden_channels_list: list, kernel_sizes: list, 
                 subsamplings: list, out_features: int = 10):
        """
        Dynamically builds the CNN architecture based on the provided hyperparameters.
        
        Args:
            in_channels (int): Number of input image channels (e.g., 3 for RGB).
            hidden_channels_list (list): Number of filters for each convolutional block.
            kernel_sizes (list): Kernel size for each convolution.
            subsamplings (list): Pooling factors (strides) for each block.
            out_features (int): Number of output classes (e.g., 10 for STL-10).
        """
        super(CNNBaseline, self).__init__()
        
        self.features = nn.Sequential()
        current_channels = in_channels
        
        for i in range(len(hidden_channels_list)):
            # Add Convolutional Layer
            self.features.add_module(f"conv_{i}", nn.Conv2d(
                in_channels=current_channels,
                out_channels=hidden_channels_list[i],
                kernel_size=kernel_sizes[i],
                padding=0 
            ))
            
            # Add ReLU Activation
            self.features.add_module(f"relu_{i}", nn.ReLU())
            
            # Add Pooling Layer (if subsampling factor > 1)
            # Uses AveragePooling to maintain consistency with the CKN's information loss profile
            if subsamplings[i] > 1:
                self.features.add_module(f"pool_{i}", nn.AvgPool2d(
                    kernel_size=subsamplings[i],
                    stride=subsamplings[i]
                ))
            
            current_channels = hidden_channels_list[i]
        
        # Global Average Pooling: Reduces each feature map to a single mean value
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final Linear Classifier
        self.classifier = nn.Linear(current_channels, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass: Features -> Global Avg Pool -> Flatten -> Classify.
        """
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_model(self, train_loader: DataLoader, test_loader: DataLoader, 
                    epochs: int = 30, lr: float = 3e-4, weight_decay: float = 1e-4, 
                    device: str = 'cuda'):
        """
        Executes the full supervised training loop using the Adam optimizer and CrossEntropyLoss.
        Prints loss and accuracy metrics at the end of every epoch.
        """
        self.to(device)
        self.train()
        
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        print(f"\nStarting CNN Baseline Training (Adam, lr={lr})...")
        
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
        Computes the model accuracy on a given dataset.
        Sets the model to evaluation mode and disables gradient calculation to save memory.
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