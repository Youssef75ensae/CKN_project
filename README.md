# Convolutional Kernel Networks (CKN) on STL-10

This repository hosts the final project for the Advanced Machine Learning course (taught by Austin Stromme at ENSAE Paris).
It provides a reproducible PyTorch implementation of Convolutional Kernel Networks (CKN), bridging the gap between Reproducing Kernel Hilbert Spaces (RKHS) and Deep Learning. The project explores the effectiveness of unsupervised dictionary learning (specifically Spherical K-Means) combined with Gaussian kernel operations.

## Project Structure

The project is organized to separate the core model logic from experimental scripts and results:

* **`src/`**: Contains the core library code.
    * `models.py`: Custom implementation of `CKNLayer` (patch extraction, Gaussian kernel, pooling) and `CKNSequential`. Features dynamic layer sizing for variable input resolutions.
    * `cnn_baseline.py`: Implementation of the supervised CNN baseline used for performance comparison.
    * `utils.py`: Utilities for reproducibility (seeding), data loading (STL-10, Fashion-MNIST), and synthetic data generation.

* **`scripts/`**: Contains standalone scripts for analysis and visualization.
    * `ablation_study.py`: Performs sensitivity analysis on hyperparameters (Capacity, Selectivity, Budget).
    * `visualize_filters.py`: Visualizes the filters learned by the unsupervised layer.
    * `plot_ablation.py`: Generates comparative figures based on ablation results.
    * `plot_metrics.py`: Visualizes training dynamics (Loss/Accuracy).

* **`figures/`**: Stores generated images (learned filters, training curves).
* **`results/`**: Stores quantitative outputs (CSV logs from ablation studies).

* **`main.py`**: The primary execution script at the root. It orchestrates the pipeline: dataset loading, unsupervised feature learning, and supervised linear classification.

## Requirements

The codebase requires Python 3.8 or higher. Dependencies can be installed via pip:

~~~bash
pip install torch torchvision numpy pandas matplotlib seaborn
~~~

## Instructions for Reproduction

To replicate the experiments presented in the report, follow the steps below.

### 1. Training and Evaluation

The model supports three experimental configurations. Run the command corresponding to the desired experiment:

#### A. Standard Benchmark (STL-10)

Main experiment evaluating performance on color images (96x96).

~~~bash
python main.py --dataset stl10 --epochs 20
~~~

#### B. Robustness Test (Fashion-MNIST)

Evaluates adaptability to grayscale images (28x28) with different feature scales.

~~~bash
python main.py --dataset fashion --epochs 15
~~~

#### C. Theoretical Validation (Synthetic Data)

Trains on a custom dataset of vertical vs. horizontal bars to validate edge detection capabilities.

~~~bash
python main.py --dataset synthetic
~~~

### 2. Generating Learning Curves

To visualize the training dynamics of the most recent run:

~~~bash
python plot_metrics.py
~~~

Output: Generates training_curves.png.

### 3. Filter Visualization

To qualitatively assess the features learned during the unsupervised phase:

~~~bash
python visualize_filters.py
~~~

Output: Generates learned_filters.png.

### 4. Ablation Study

To evaluate sensitivity to hyperparameters (Number of filters, Kernel bandwidth alpha, Sampling budget):

~~~bash
python ablation_study.py
~~~

Output: Generates ablation_results.csv.

### 5. Plotting Ablation Results

Once the ablation CSV is generated:

~~~bash
python plot_ablation.py
~~~

Output: Generates ablation_study.png.

## Code Originality Statement

In accordance with project guidelines:

* **Original Implementation**: The `CKNLayer` and `CKNSequential` classes in `src/models.py` are implemented from scratch. We consulted the repository `claying/CKN-Pytorch-image` to understand the tensor dimensions of the Gaussian kernel operations, but the final PyTorch implementation (leveraging `torch.nn.Unfold` and custom modularity) is original.
* **Unsupervised Learning**: The optimization loop (Spherical K-Means) is a custom implementation designed to normalize patches and update centroids without external clustering libraries.
* **Custom Data Pipeline**: The `SyntheticBars` class in `src/utils.py` is a custom implementation created to generate procedural data for theoretical validation.
* **Visualization**: All plotting scripts were created specifically for this project.
* **Baseline Implementation**: The CNN baseline architecture and training loop (`cnn_baseline.py`) are adapted from standard practices described in Aurélien Géron's book, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*.

## References

* Mairal, J., Koniusz, P., Harchaoui, Z., & Schmid, C. (2014). Convolutional Kernel Networks. Advances in Neural Information Processing Systems (NIPS).
* Coates, A., Ng, A., & Lee, H. (2011). An Analysis of Single-Layer Networks in Unsupervised Feature Learning. AISTATS.
* Géron, A. (2019). Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow. O'Reilly Media.
