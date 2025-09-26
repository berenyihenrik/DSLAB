# AutoEncoder Implementations

This folder contains notebooks implementing various autoencoder architectures for time-series anomaly detection.

## Contents

- `AE.ipynb`:  
  Implementation of a standard (vanilla) autoencoder for time-series data.

- `LSTM AE.ipynb`:  
  Implementation of an LSTM-based autoencoder, designed to capture temporal dependencies in sequential data.

- `LSTM VAE.ipynb`:  
  Implementation of an LSTM-based variational autoencoder (VAE), combining sequence modeling with probabilistic latent representations.

## Overview

Each notebook provides:

- Data loading and preprocessing steps
- Model architecture definition
- Training and validation routines
- Anomaly scoring and thresholding
- Performance evaluation and visualization

## Usage

1. **Prepare Data:**  
   Ensure your time-series dataset is available and referenced correctly in the notebooks.

2. **Run Notebooks:**  
   Open and execute the notebooks to train the models and perform anomaly detection.

3. **Review Results:**  
   Analyze the output metrics and plots to evaluate model performance.

## Output

- Reconstruction errors and anomaly scores
- Detected anomaly indices
- Visualizations of model results

---

For further details, see the code and comments in each notebook.
