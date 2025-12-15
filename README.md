# Adaptive Simulated Annealing for Clustering & Optimization

A robust, publication-ready implementation of **Simulated Annealing (SA)** for clustering and optimization tasks, featuring adaptive temperature calibration and a hybrid "polishing" mechanism.

## 📌 Project Overview

This repository contains the code for the paper *"Adaptive Simulated Annealing with Polishing for Clustering: A Comprehensive Multi-Dataset Analysis"*. It demonstrates how Simulated Annealing can overcome local optima in complex optimization landscapes where traditional methods like K-means often fail.

### Key Features
*   **Adaptive Temperature Calibration**: Automatically determines the optimal starting temperature ($T_0$) based on dataset-specific energy barriers.
*   **Greedy Polishing Phase**: A hybrid approach that uses K-means refinement after SA cooling to ensure micro-convergence.
*   **Uphill Exploration**: Probabilistic acceptance of worse solutions allows the algorithm to escape local optima.
*   **Multi-Dataset Benchmarking**: Validated on 14+ datasets including UCI Real-world data (Iris, Wine, Breast Cancer, Glass) and complex synthetic benchmarks (Grid, Chaotic, Blobs).

## 📂 Files

*   **`Simulated_Annealing_Publication.ipynb`**: The main Jupyter Notebook containing:
    *   Full algorithm implementation (SA + Polishing).
    *   Comparison logic against K-means (Single Random Init).
    *   Automated data loading for 14 datasets.
    *   Visualization code for convergence plots and 2D cluster maps.
*   **`simulated_annealing_publication_code.py`**: A Python script version of the notebook for command-line execution or import.

## 🚀 Getting Started

### Prerequisites
*   Python 3.8+
*   `numpy`
*   `pandas`
*   `matplotlib`
*   `scikit-learn`

### Installation
```bash
git clone https://github.com/yourusername/Adaptive-Simulated-Annealing-Clustering.git
cd Adaptive-Simulated-Annealing-Clustering
pip install numpy pandas matplotlib scikit-learn
```

### Usage
Run the Jupyter Notebook to reproduce the paper's results:
```bash
jupyter notebook Simulated_Annealing_Publication.ipynb
```
Or run the script directly:
```bash
python3 simulated_annealing_publication_code.py
```

## 📊 Results Summary

The Adaptive SA+Polishing algorithm consistently matches or outperforms standard K-means, with dramatic improvements on complex geometries:
*   **Grid Dataset**: **87.6% improvement** in distortion (Correctly finds $4\times4$ grid structure).
*   **1D Motivating Example**: **15.7% improvement** (Escapes local trap).
*   **Detailed Exploration**: The code includes specific visualizations showing the "Uphill Moves" mechanism.

## 📜 License
This project is open source and available under the [MIT License](LICENSE).
