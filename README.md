# Convolutional Kernel Networks (CKN) on STL-10

This repository contains a PyTorch implementation of Convolutional Kernel Networks (CKN). The project explores the effectiveness of unsupervised dictionary learning (specifically Spherical K-Means) combined with Gaussian kernel operations. The model is evaluated on the STL-10 dataset, as well as Fashion-MNIST and a custom synthetic dataset to demonstrate robustness and theoretical validity.

## Project Structure

The project is organized as follows:

* **`src/`**: Contains the core source code.
    * `models.py`: Custom implementation of `CKNLayer` (patch extraction, Gaussian kernel, pooling) and `CKNSequential`. Features dynamic layer sizing for variable input resolutions.
    * `utils.py`: Utilities for reproducibility (seeding), data loading (STL-10, Fashion-MNIST), and synthetic data generation.
* **`main.py`**: The primary execution script. It orchestrates the pipeline: dataset loading, unsupervised feature learning, and supervised linear classification.
* **`ablation_study.py`**: Performs sensitivity analysis on hyperparameters (Capacity, Selectivity, Budget).
* **`plot_ablation.py`**: Generates comparative figures based on ablation results.
* **`plot_metrics.py`**: Visualizes training dynamics (Loss/Accuracy).
* **`visualize_filters.py`**: Visualizes the filters learned by the unsupervised layer.

## Requirements

The codebase requires Python 3.8 or higher. Dependencies can be installed via pip:

```bash
pip install torch torchvision numpy pandas matplotlib seaborn