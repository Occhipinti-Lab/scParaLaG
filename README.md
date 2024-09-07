# scParaLaG: A Parallel Residual and Layer-Attentive Graph Neural Network for Multimodal Single Cell Analysis.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Model Architecture](#model-architecture)


## Introduction
**scParaLaG** is a state-of-the-art Graph Neural Network (GNN) designed for multimodal single-cell analysis, utilizing parallel residual connections and layer-attentive mechanisms. This model efficiently integrates data from multiple modalities, such as RNA and protein, addressing challenges of modality prediction in cases where only unimodal data is available for a given cell.

This is particularly relevant in scenarios where:

1. Technical limitations or cost constraints prevent simultaneous measurement of multiple modalities for all cells.
2. There's a need to computationally infer missing modalities to create a more comprehensive multi-omic profile.
3. Researchers aim to leverage the complementary information provided by different molecular layers (e.g., transcriptomics, proteomics, epigenomics) to gain a more holistic understanding of cellular states and functions.
4. There's a requirement to harmonize and integrate heterogeneous single-cell datasets collected using different technologies or measuring different molecular modalities.

By accurately predicting modality pairs, especially in the absence of pathway data, scParaLaG enables more robust multi-omic analyses and helps overcome the sparsity and incompleteness often encountered in single-cell multi-modal datasets.

## Features
- **Parallel Residual Connections**: Stabilizes deep model training for better performance.
- **Layer-Attentive Mechanism**: Focuses on relevant layers in the network to capture essential biological features.
- **Graph-Based Representation**: Models cell-cell interactions effectively.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Occhipinti-Lab/scParaLaG.git
    cd scParaLaG
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Install PyYAML for configuration management:
    ```bash
    pip install pyyaml
    ```

## Usage

1. **Create or modify the `config.yaml`** file with your desired parameters. Example configuration:
    ```yaml
    subtask: "openproblems_bmmc_cite_phase2_rna_subset"
    batch_size: 520
    act: "leaky_relu"
    conv_flow: ['gat']
    agg_flow: [null]
    device: "cpu"
    learning_rate: 0.00004
    hidden_size: 512
    num_heads: 2
    n_neigbours: 3
    n_components: 1200
    metric: "nan_euclidean"
    dropout_rate: 0.2
    num_epochs: 500
    seed: 1
    es_patience: 50
    es_min_delta: 0.01
    es_rate_threshold: 0.0008
    layer_dim_ex: 3
    verbose: true
    sample: true
    preprocessing_type: "SVD"
    ```

2. **Run the model** using the YAML configuration:
    ```bash
    python main.py --config config.yaml
    ```

## Configuration

The model's behavior is controlled via a `config.yaml` file. Below are the available configuration parameters:

- **subtask**: Name of the dataset subtask.
- **batch_size**: Number of samples per batch.
- **act**: Activation function (`relu`, `relu6`, `gelu`, `leaky_relu`).
- **conv_flow**: Convolutional layer flow (`gat`, `sage`, `gconv`).
- **agg_flow**: Aggregation flow (`None`, `mean`).
- **device**: Device for computation (`cpu`, `cuda`).
- **learning_rate**: Learning rate for the optimizer.
- **hidden_size**: Size of the hidden layers.
- **num_heads**: Number of attention heads in the model.
- **n_neigbours**: Number of neighbors for graph construction.
- **n_components**: Number of components for feature reduction (SVD).
- **metric**: Distance metric for neighbors.
- **dropout_rate**: Dropout rate for regularization.
- **num_epochs**: Number of training epochs.
- **seed**: Random seed for reproducibility.
- **es_patience**: Early stopping patience.
- **es_min_delta**: Minimum delta for early stopping.
- **es_rate_threshold**: Threshold for early stopping rate.
- **layer_dim_ex**: Layer dimension expansion factor.
- **verbose**: Whether to print detailed output (`True`, `False`).
- **sample**: Whether to sample the data during training.
- **preprocessing_type**: Preprocessing method (`SVD`, `None`).

## Model Architecture

The **scParaLaG** architecture includes:
- **Graph Construction**: Graph-based representation of cell-cell interactions.
- **Parallel Residual Layer**: For raw combination of different layers outputs.
- **Layer Attention Module**: Selects the most relevant feature(s).