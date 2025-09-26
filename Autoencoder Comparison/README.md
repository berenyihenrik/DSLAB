# Autoencoder Comparison

This folder contains notebooks and documentation for comparing different autoencoder architectures on time-series anomaly detection tasks.

## Contents

- `AutoEncoders Compared.ipynb`:  
  Main notebook for comparing various autoencoder models (e.g., vanilla autoencoder, LSTM autoencoder, convolutional autoencoder) on generic time-series datasets.

- `AutoEncoders Compared swat.ipynb`:  
  Notebook focused on applying and comparing autoencoder models specifically on the SWaT (Secure Water Treatment) dataset for anomaly detection.

## Overview

The notebooks provide implementations and experiments for evaluating the performance of different autoencoder-based approaches for detecting anomalies in time-series data. Each notebook includes:

- Data loading and preprocessing
- Model definitions for multiple autoencoder types
- Training and validation routines
- Anomaly scoring and thresholding
- Performance evaluation (e.g., F1 score, precision, recall)
- Visualization of results

## Usage

1. **Prepare Data:**  
   Ensure your time-series data (or SWaT dataset) is available and correctly referenced in the notebooks.

2. **Run Experiments:**  
   Open and execute the notebooks to train models, compare results, and visualize anomaly detection performance.

3. **Review Results:**  
   Use the provided metrics and plots to assess which autoencoder architecture works best for your application.

## Output

- Comparative metrics for each autoencoder model
- Anomaly indices and scores
- Plots of reconstruction errors and detected anomalies

---

For details, see the code and comments in each notebook.
