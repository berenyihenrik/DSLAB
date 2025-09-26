# TopoMAD Reimplementation

This notebook (`TopoMAD Reimplementation.ipynb`) provides a full reimplementation of the TopoMAD anomaly detection algorithm for multivariate time-series data with graph-based topology.

## Overview

TopoMAD combines temporal modeling (LSTM) and spatial modeling (graph neural networks) to detect anomalies in datasets where each variable/component is associated with a node in a graph. The model leverages both the time-series behavior and the relationships between nodes.

## Main Features

- **Data Preprocessing:** Loads and normalizes multivariate time-series data and graph topology.
- **Graph LSTM VAE Model:** Implements a variational autoencoder with GraphLSTM cells (GCN, GAT, WL1, or Linear) for encoding and decoding sequences.
- **Training & Validation:** Trains the model using a combination of reconstruction and KL-divergence losses, with early stopping and teacher forcing.
- **Anomaly Detection:** Computes anomaly scores for each time window and identifies anomalies using a percentile threshold.
- **Evaluation:** Aligns predicted anomalies with ground truth and calculates the F1 score for performance assessment.

## Usage

**Data Preparation:**

- Place your time-series CSV and topology pickle file in the specified directory.

**Model Training:**

- Run the notebook to preprocess data, initialize the model, and train.

**Evaluation:**

- The notebook outputs anomaly indices and F1 score for the detection results.

## Key Classes & Functions

- `Algorithm`: Abstract base for anomaly detection algorithms.
- `PyTorchUtils`: Device and tensor utilities for PyTorch.
- `GraphLSTM`: Graph-based LSTM implementation.
- `GraphLSTM_VAE`: Variational autoencoder with GraphLSTM backbone.
- `GraphLSTM_VAE_AD`: Main anomaly detection class with fit, predict, and interpret methods.

## Output

- **Anomaly Indices:** List of detected anomaly time points.
- **F1 Score:** Quantitative measure of detection accuracy.

---

For more details, see the code and comments in `TopoMAD Reimplementation.ipynb`.
