"""
Ablation Study for scParaLaG Model

Note: The actual ablation study was conducted as an iterative process, exploring various
parameters and their combinations incrementally. This script provides a comprehensive
template that includes all the parameters tested throughout the study, with the exception
of batch normalization. Batch normalization was excluded from this template as its 
inclusion would require structural changes to the model.

The parameters explored in this template include:
- Activation functions
- Convolutional layer flows
- Aggregation flows
- Dropout rates
- L2 regularization strengths

While this template offers a broad overview of the parameters examined, in practice,
the ablation study was performed in stages, with insights from each iteration informing
subsequent experiments.
"""

from argparse import Namespace
import torch
from dance.datasets.multimodality import ModalityPredictionDataset
from graph_transform import GraphCreator
from train import scParaLaGWrapper
import pandas as pd

def run_ablation_study():
    base_args = {
        "subtask": "openproblems_bmmc_cite_phase2_mod2",
        "batch_size": 520,
        "act": "leaky_relu",
        "conv_flow": ['gat'],
        "agg_flow": [None],
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "learning_rate": 0.000064,
        "hidden_size": 512,
        "num_heads": 3,
        "n_neigbours": 3,
        "n_components": 1200,
        "metric": 'euclidean',
        "dropout_rate": 0.2,
        "num_epochs": 500,
        "seed": 1,
        "es_patience": 40,
        "es_min_delta": 0.01,
        "es_rate_threshold": 0.0008,
        "layer_dim_ex": 3,
        "verbose": True,
        "sample": True,
        "preprocessing_type": "None"
    }

    ablation_params = {
        "act": ["relu", "gelu", "leaky_relu", "tanh"], 
        "conv_flow": [['gat'], ['gat', 'gconv'], ['sage'], ['gconv'], ['gat', 'sage']],
        "agg_flow": [[None], ['mean'], [None, 'mean']],
        "dropout_rate": [0.2],
        "l2_lambda": [0.0, 0.01, 0.001, 0.0001]
    }

    results = []

    dataset = ModalityPredictionDataset(base_args["subtask"], preprocess=None)
    data = dataset.load_data()

    for act in ablation_params["act"]:
        for conv_flow in ablation_params["conv_flow"]:
            for agg_flow in ablation_params["agg_flow"]:
                for dropout_rate in ablation_params["dropout_rate"]:
                    for l2_lambda in ablation_params["l2_lambda"]:
                        current_args = base_args.copy()
                        current_args.update({
                            "act": act,
                            "conv_flow": conv_flow,
                            "agg_flow": agg_flow,
                            "dropout_rate": dropout_rate
                        })

                        data, train_label, val_label, test_label, ftl_shape = GraphCreator(
                            current_args["preprocessing_type"], current_args["n_neigbours"],
                            current_args["n_components"], current_args["metric"])(data)

                        if current_args["preprocessing_type"] == "SVD":
                            current_args["FEATURE_SIZE"] = current_args["n_components"]
                        else:
                            current_args["FEATURE_SIZE"] = ftl_shape[0]
                        current_args["OUTPUT_SIZE"] = ftl_shape[1]

                        train_graph, val_graph, test_graph = data.data.uns['gtrain'], data.data.uns['gval'], data.data.uns['gtest']

                        model = scParaLaGWrapper(Namespace(**current_args))
                        model.fit(train_graph, val_graph, test_graph, torch.tensor(train_label),
                                  torch.tensor(val_label), torch.tensor(test_label),
                                  num_epochs=current_args["num_epochs"], batch_size=current_args["batch_size"],
                                  verbose=current_args["verbose"], es_patience=current_args["es_patience"],
                                  es_min_delta=current_args["es_min_delta"],
                                  es_rate_threshold=current_args["es_rate_threshold"],
                                  learning_rate=current_args["learning_rate"],
                                  sample=current_args["sample"],
                                  l2_lambda=l2_lambda)

                        test_score = model.score(test_graph, torch.tensor(test_label))
                        results.append({
                            "activation": act,
                            "conv_flow": str(conv_flow),
                            "agg_flow": str(agg_flow),
                            "dropout_rate": dropout_rate,
                            "l2_lambda": l2_lambda,
                            "test_score": test_score
                        })
    return results

if __name__ == "__main__":
    results = run_ablation_study()
    df_results = pd.DataFrame(results)
    print(df_results.sort_values('test_score', ascending=True))
