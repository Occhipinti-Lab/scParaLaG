# scParaLaG: A Parallel Residual, Layer-Attentive Graph Neural Network for Multimodal Single Cell Analysis



## Features

- **Multimodal Single-Cell Analysis**: Maps between GEX, ADT, and ATAC data.
- **Parallel Residual Layers**: Improves performance by capturing high-order information across layers.
- **Layer-Attentive Mechanism**: Enhances learning from data with limited biological knowledge.
- **Efficient Training**: Optimized for performance with reduced computational resource requirements.
- **Benchmarking**: Outperforms traditional models in the NeurIPS competition dataset, achieving the lowest RMSE loss.

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

## Usage

1. Prepare your multimodal single-cell data (e.g., GEX, ADT, ATAC) and place it in the `data/` directory.
2. Run the model with the default configuration:
    ```bash
    python main.py -- your-desired-config
    ```


## Model Architecture

- **Graph Construction**: Creates cell-cell graphs to model interactions.
- **Parallel Residual Networks**: Stabilizes training and improves performance across modalities.
- **Layer Attention**: Learns important features from multiple layers for better predictions.
- **Prediction Head**: Outputs results such as gene expression or chromatin accessibility predictions.

## Benchmarking and Results

scParaLaG was benchmarked against leading models within the **PyDance** package and evaluated on the NeurIPS competition dataset. It demonstrated superior performance, achieving the lowest RMSE loss across all tasks and proving its robustness in multimodal single-cell analysis, especially in scenarios lacking prior biological knowledge.