# scParaLaG: Parallel Residual and Layer-Attentive Graph Neural Network for Multimodal Single Cell Analysis

**scParaLaG** is an advanced Graph Neural Network (GNN) designed for multimodal single-cell analysis. It leverages parallel residual connections and layer-attentive mechanisms to efficiently integrate data from diverse modalities (e.g., RNA, protein). A key capability of scParaLaG is its ability to predict missing modalities for cells where only unimodal data is available, thereby facilitating more comprehensive multi-omic studies.

This tool is particularly useful for:
* Inferring missing modalities to create complete multi-omic profiles.
* Integrating heterogeneous single-cell datasets.
* Gaining a holistic understanding of cellular states by leveraging complementary molecular information.

## Key Features
-   **Parallel Residual Connections**: Enhances deep model stability and performance.
-   **Layer-Attentive Mechanism**: Focuses on biologically relevant features across network layers.
-   **Graph-Based Representation**: Effectively models complex cell-cell interactions.

## Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/Occhipinti-Lab/scParaLaG.git](https://github.com/Occhipinti-Lab/scParaLaG.git)
    cd scParaLaG
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Quick Start

1.  **Configure**: Modify the `config.yaml` file to set your desired parameters, including dataset details, model hyperparameters, and processing options.
2.  **Run the model**:
    ```bash
    python run.py --config config.yaml
    ```

For detailed configuration options, please refer to the comments and structure within the `config.yaml` file.
