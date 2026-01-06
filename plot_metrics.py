import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_logs(log_data, model_name):
    """
    Parses the raw training logs to extract performance metrics.

    This function scans the log string line by line using a regular expression
    to find the Epoch number, Loss, Train Accuracy, and Test Accuracy.
    It returns a list of dictionaries, where each dictionary corresponds to
    the metrics of a single epoch.

    Args:
        log_data (str): The raw string containing the training logs.
        model_name (str): The name of the model (e.g., 'CKN', 'CNN Baseline').

    Returns:
        list: A list of dictionaries containing structured metric data.
    """
    data = []
    # Regex to capture: Epoch X/Y | Loss: Z | Train Acc: A% | Test Acc: B%
    pattern = r"Epoch (\d+)/\d+ \| Loss: ([\d\.]+) \| Train Acc: ([\d\.]+)% \| Test Acc: ([\d\.]+)%"
    
    for line in log_data.strip().split('\n'):
        match = re.search(pattern, line)
        if match:
            data.append({
                'Model': model_name,
                'Epoch': int(match.group(1)),
                'Loss': float(match.group(2)),
                'Train Accuracy': float(match.group(3)),
                'Test Accuracy': float(match.group(4))
            })
    return data

def main():
    """
    Main execution function.
    
    1. Defines the raw logs for both CKN and CNN models.
    2. Parses these logs into a unified pandas DataFrame.
    3. Generates and saves a comparative plot for Training Loss and Test Accuracy.
    """
    
    # Raw logs from the CKN (Ours) execution
    ckn_logs = """
    Epoch 1/30 | Loss: 2.3638 | Train Acc: 19.40% | Test Acc: 25.32%
    Epoch 2/30 | Loss: 2.0575 | Train Acc: 25.66% | Test Acc: 24.86%
    Epoch 3/30 | Loss: 2.0442 | Train Acc: 27.26% | Test Acc: 26.98%
    Epoch 4/30 | Loss: 2.0073 | Train Acc: 29.06% | Test Acc: 28.57%
    Epoch 5/30 | Loss: 2.0115 | Train Acc: 28.62% | Test Acc: 22.50%
    Epoch 6/30 | Loss: 2.0045 | Train Acc: 28.10% | Test Acc: 30.35%
    Epoch 7/30 | Loss: 1.9627 | Train Acc: 30.06% | Test Acc: 29.46%
    Epoch 8/30 | Loss: 1.9289 | Train Acc: 30.34% | Test Acc: 27.90%
    Epoch 9/30 | Loss: 1.9297 | Train Acc: 31.20% | Test Acc: 31.82%
    Epoch 10/30 | Loss: 1.9635 | Train Acc: 29.88% | Test Acc: 29.59%
    Epoch 11/30 | Loss: 1.9243 | Train Acc: 31.20% | Test Acc: 33.24%
    Epoch 12/30 | Loss: 1.9134 | Train Acc: 31.52% | Test Acc: 29.09%
    Epoch 13/30 | Loss: 1.8730 | Train Acc: 32.92% | Test Acc: 29.91%
    Epoch 14/30 | Loss: 1.9024 | Train Acc: 31.58% | Test Acc: 33.01%
    Epoch 15/30 | Loss: 1.8623 | Train Acc: 33.64% | Test Acc: 31.44%
    Epoch 16/30 | Loss: 1.8858 | Train Acc: 31.70% | Test Acc: 32.24%
    Epoch 17/30 | Loss: 1.8345 | Train Acc: 32.68% | Test Acc: 26.71%
    Epoch 18/30 | Loss: 1.9089 | Train Acc: 32.30% | Test Acc: 29.36%
    Epoch 19/30 | Loss: 1.8489 | Train Acc: 32.96% | Test Acc: 33.76%
    Epoch 20/30 | Loss: 1.8585 | Train Acc: 33.42% | Test Acc: 32.08%
    Epoch 21/30 | Loss: 1.8451 | Train Acc: 33.76% | Test Acc: 31.99%
    Epoch 22/30 | Loss: 1.8170 | Train Acc: 34.90% | Test Acc: 24.74%
    Epoch 23/30 | Loss: 1.8736 | Train Acc: 33.36% | Test Acc: 29.14%
    Epoch 24/30 | Loss: 1.8648 | Train Acc: 32.06% | Test Acc: 30.98%
    Epoch 25/30 | Loss: 1.8952 | Train Acc: 33.70% | Test Acc: 29.62%
    Epoch 26/30 | Loss: 1.8325 | Train Acc: 35.20% | Test Acc: 33.69%
    Epoch 27/30 | Loss: 1.8181 | Train Acc: 34.26% | Test Acc: 32.08%
    Epoch 28/30 | Loss: 1.8259 | Train Acc: 33.36% | Test Acc: 33.46%
    Epoch 29/30 | Loss: 1.8057 | Train Acc: 35.26% | Test Acc: 32.33%
    Epoch 30/30 | Loss: 1.8423 | Train Acc: 32.78% | Test Acc: 32.62%
    """

    # Raw logs from the CNN Baseline execution
    cnn_logs = """
    Epoch 1/30 | Loss: 2.1702 | Train Acc: 20.36% | Test Acc: 24.18%
    Epoch 2/30 | Loss: 2.0464 | Train Acc: 24.00% | Test Acc: 25.70%
    Epoch 3/30 | Loss: 1.9975 | Train Acc: 25.64% | Test Acc: 28.07%
    Epoch 4/30 | Loss: 1.9463 | Train Acc: 28.20% | Test Acc: 28.51%
    Epoch 5/30 | Loss: 1.9018 | Train Acc: 29.18% | Test Acc: 31.35%
    Epoch 6/30 | Loss: 1.8576 | Train Acc: 30.92% | Test Acc: 32.70%
    Epoch 7/30 | Loss: 1.8091 | Train Acc: 32.12% | Test Acc: 34.36%
    Epoch 8/30 | Loss: 1.7719 | Train Acc: 32.92% | Test Acc: 34.00%
    Epoch 9/30 | Loss: 1.7421 | Train Acc: 34.72% | Test Acc: 35.34%
    Epoch 10/30 | Loss: 1.7193 | Train Acc: 34.30% | Test Acc: 35.58%
    Epoch 11/30 | Loss: 1.6999 | Train Acc: 34.66% | Test Acc: 33.29%
    Epoch 12/30 | Loss: 1.6871 | Train Acc: 35.00% | Test Acc: 36.49%
    Epoch 13/30 | Loss: 1.6700 | Train Acc: 36.08% | Test Acc: 35.66%
    Epoch 14/30 | Loss: 1.6553 | Train Acc: 35.96% | Test Acc: 36.88%
    Epoch 15/30 | Loss: 1.6485 | Train Acc: 36.62% | Test Acc: 36.29%
    Epoch 16/30 | Loss: 1.6331 | Train Acc: 36.38% | Test Acc: 35.36%
    Epoch 17/30 | Loss: 1.6257 | Train Acc: 37.14% | Test Acc: 39.02%
    Epoch 18/30 | Loss: 1.6118 | Train Acc: 38.82% | Test Acc: 37.62%
    Epoch 19/30 | Loss: 1.6148 | Train Acc: 38.16% | Test Acc: 39.41%
    Epoch 20/30 | Loss: 1.6009 | Train Acc: 39.76% | Test Acc: 39.24%
    Epoch 21/30 | Loss: 1.5930 | Train Acc: 39.56% | Test Acc: 39.77%
    Epoch 22/30 | Loss: 1.5883 | Train Acc: 39.74% | Test Acc: 38.54%
    Epoch 23/30 | Loss: 1.5811 | Train Acc: 39.98% | Test Acc: 39.39%
    Epoch 24/30 | Loss: 1.5819 | Train Acc: 39.66% | Test Acc: 39.74%
    Epoch 25/30 | Loss: 1.5714 | Train Acc: 40.96% | Test Acc: 41.25%
    Epoch 26/30 | Loss: 1.5596 | Train Acc: 41.62% | Test Acc: 40.85%
    Epoch 27/30 | Loss: 1.5558 | Train Acc: 41.26% | Test Acc: 39.85%
    Epoch 28/30 | Loss: 1.5555 | Train Acc: 41.72% | Test Acc: 40.27%
    Epoch 29/30 | Loss: 1.5389 | Train Acc: 41.50% | Test Acc: 41.24%
    Epoch 30/30 | Loss: 1.5375 | Train Acc: 42.94% | Test Acc: 41.06%
    """

    # Create the DataFrame
    df_ckn = pd.DataFrame(parse_logs(ckn_logs, 'CKN (Ours)'))
    df_cnn = pd.DataFrame(parse_logs(cnn_logs, 'CNN Baseline'))
    df = pd.concat([df_ckn, df_cnn])

    # Setup plot style
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Training Loss
    sns.lineplot(ax=axes[0], data=df, x='Epoch', y='Loss', hue='Model', marker='o')
    axes[0].set_title('Training Loss over Epochs')
    axes[0].set_ylabel('Cross Entropy Loss')

    # Plot 2: Test Accuracy
    sns.lineplot(ax=axes[1], data=df, x='Epoch', y='Test Accuracy', hue='Model', marker='o')
    axes[1].set_title('Test Accuracy over Epochs')
    axes[1].set_ylabel('Accuracy (%)')

    # Save the figure to disk
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300)
    print("Plot saved as 'training_curves.png'")

if __name__ == "__main__":
    main()