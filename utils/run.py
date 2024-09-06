"""
----------------------------------------------------------------------------------------------------------------------------------------
Description:                                                                                                                            |
    This module is used for building, customizing, and training of models made with the scParaLaG Framework.                            |
                                                                                                                                        |
Copyright:                                                                                                                              |
    Copyright © 2024. All rights reserved.                                                                                              |
                                                                                                                                        |
License:                                                                                                                                |
    This script is licensed under the MIT License.                                                                                      |
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software                                       |
    and associated documentation files (the "Software"), to deal in the Software without restriction,                                   |
    including without limitation the rights to use, copy, modify, merge, publish, distribute,                                           |
    sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,                   |
    subject to the following conditions:                                                                                                |
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.      |
                                                                                                                                        |
Disclaimer:                                                                                                                             |
    This software is provided 'as is' and without any express or implied warranties.                                                    |
    The author or the copyright holders make no representations about the suitability of this software for any purpose.                 |
                                                                                                                                        |
Contact:                                                                                                                                |
    For any queries or issues related to this script, please contact fchumeh@gmail.com.                                                 |
----------------------------------------------------------------------------------------------------------------------------------------
"""

import argparse
from argparse import Namespace
import torch
import yaml
from dance.datasets.multimodality import ModalityPredictionDataset
from graph_transform import GraphCreator
from train import scParaLaGWrapper

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def validate_config(config):
    required_params = {
        "subtask": str,
        "batch_size": int,
        "act": str,
        "conv_flow": list,
        "agg_flow": list,
        "device": str,
        "learning_rate": float,
        "hidden_size": int,
        "num_heads": int,
        "n_neigbours": int,
        "n_components": int,
        "metric": str,
        "dropout_rate": float,
        "num_epochs": int,
        "seed": int,
        "es_patience": int,
        "es_min_delta": float,
        "es_rate_threshold": float,
        "layer_dim_ex": int,
        "verbose": bool,
        "sample": bool,
        "preprocessing_type": str
    }
    
    for param, param_type in required_params.items():
        if param not in config:
            raise ValueError(f"Missing required parameter: {param}")
        if not isinstance(config[param], param_type):
            raise TypeError(f"Parameter {param} should be of type {param_type}, but is {type(config[param])}")

    return config


def pipeline(**kwargs):
    subtask = kwargs["subtask"]
    dataset = ModalityPredictionDataset(subtask, preprocess=None)
    data = dataset.load_data()

    if kwargs["subtask"]=="openproblems_bmmc_cite_phase2_mod2":
        kwargs["preprocessing_type"] = "None"

    data, train_label, val_label, test_label, ftl_shape = GraphCreator(
        kwargs["preprocessing_type"], kwargs["n_neigbours"],
        kwargs["n_components"],
        kwargs["metric"])(data)

    if kwargs["preprocessing_type"]=="SVD":
        kwargs["FEATURE_SIZE"] = kwargs["n_components"]
    else:
        kwargs["FEATURE_SIZE"] = ftl_shape[0]
    kwargs["OUTPUT_SIZE"] = ftl_shape[1]

    train_graph, val_graph, test_graph = data.data.uns['gtrain'], data.data.uns['gval'], data.data.uns['gtest']

    model = scParaLaGWrapper(Namespace(**kwargs))
    model.fit(train_graph, val_graph, test_graph, torch.tensor(train_label),
              torch.tensor(val_label), torch.tensor(test_label),
              num_epochs=kwargs["num_epochs"], batch_size=kwargs["batch_size"],
              verbose=kwargs["verbose"], es_patience=kwargs["es_patience"],
              es_min_delta=kwargs["es_min_delta"],
              es_rate_threshold=kwargs["es_rate_threshold"],
              learning_rate=kwargs["learning_rate"],
              sample=kwargs["sample"])



def pipeline(**kwargs):
    subtask = kwargs["subtask"]
    dataset = ModalityPredictionDataset(subtask, preprocess=None)
    data = dataset.load_data()

    if kwargs["subtask"] == "openproblems_bmmc_cite_phase2_mod2":
        kwargs["preprocessing_type"] = "None"

    data, train_label, val_label, test_label, ftl_shape = GraphCreator(
        kwargs["preprocessing_type"], kwargs["n_neigbours"],
        kwargs["n_components"],
        kwargs["metric"])(data)

    if kwargs["preprocessing_type"] == "SVD":
        kwargs["FEATURE_SIZE"] = kwargs["n_components"]
    else:
        kwargs["FEATURE_SIZE"] = ftl_shape[0]
    kwargs["OUTPUT_SIZE"] = ftl_shape[1]

    train_graph, val_graph, test_graph = data.data.uns['gtrain'], data.data.uns['gval'], data.data.uns['gtest']

    model = scParaLaGWrapper(Namespace(**kwargs))
    model.fit(train_graph, val_graph, test_graph, torch.tensor(train_label),
              torch.tensor(val_label), torch.tensor(test_label),
              num_epochs=kwargs["num_epochs"], batch_size=kwargs["batch_size"],
              verbose=kwargs["verbose"], es_patience=kwargs["es_patience"],
              es_min_delta=kwargs["es_min_delta"],
              es_rate_threshold=kwargs["es_rate_threshold"],
              learning_rate=kwargs["learning_rate"],
              sample=kwargs["sample"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    validated_config = validate_config(config)

    torch.set_num_threads(1)
    pipeline(**validated_config)
