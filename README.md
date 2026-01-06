# Convolutional Kernel Networks (CKN) on STL-10

This repository contains a PyTorch implementation of Convolutional Kernel Networks (CKN) applied to the STL-10 dataset. The primary objective of this project is to explore the efficacy of unsupervised dictionary learning (specifically Spherical K-Means) combined with Gaussian kernel operations, and to compare its performance against a standard supervised Convolutional Neural Network (CNN) baseline.

## Project Structure

The project is organized as follows:

* **`src/`**: Contains the core source code for the models and utilities.
    * `models.py`: Contains the custom implementation of `CKNLayer` (patch extraction, Gaussian kernel, pooling) and `CKNSequential`.
    * `cnn_baseline.py`: A standard supervised CNN architecture used for benchmarking.
    * `utils.py`: Helper functions for random seeding and loading the STL-10 dataset.
* **`main.py`**: The primary script that executes the training pipeline. It trains the unsupervised CKN features, trains the linear classifier, and trains the CNN baseline.
* **`ablation_study.py`**: Performs sensitivity analysis on hyperparameters (Capacity, Selectivity, Budget).
* **`plot_ablation.py`**: Generates comparative bar charts based on the results of the ablation study.
* **`plot_metrics.py`**: Parses training logs to generate Loss and Accuracy curves.
* **`visualize_filters.py`**: Visualizes the convolutional filters learned by the unsupervised layer.

## Requirements

The codebase requires Python 3.8 or higher. The specific dependencies can be installed via pip:

```bash
pip install torch torchvision numpy pandas matplotlib seaborn