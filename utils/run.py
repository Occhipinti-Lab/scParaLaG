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

from dance.datasets.multimodality import ModalityPredictionDataset
from graph_transform import GraphCreator
from train import scParaLaGWrapper


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



parser = argparse.ArgumentParser()
parser.add_argument("-st", "--subtask", default="openproblems_bmmc_cite_phase2_rna")
parser.add_argument("-bs", "--batch_size", default=520, type=int)
parser.add_argument("-ac", "--act", default="leaky_relu",
                    choices=["relu", "relu6", "gelu", "leaky_relu"])
parser.add_argument("-convf", "--conv_flow", nargs='*',
                    choices=['gat', 'sage', 'gconv'], default=['gat'],
                    help="List of convolutional layer flow 'gat', 'sage' or 'gconv'")
parser.add_argument("-aggf", "--agg_flow", nargs='*', choices=[None, 'mean'],
                    default=[None],
                    help="List of aggregation flow 'mean' or 'None'")
parser.add_argument("-device", "--device", default="cpu", choices=["cpu", "cuda"])
parser.add_argument("-lr", "--learning_rate", type=float, default=0.00004)
parser.add_argument("-hid", "--hidden_size", type=int, default=512)
parser.add_argument("-nh", "--num_heads", type=int, default=2)
parser.add_argument("-nneig", "--n_neigbours", type=int, default=3)
parser.add_argument("-ncomp", "--n_components", type=int, default=1200)
parser.add_argument("-m", "--metric", type=str, default='euclidean')
parser.add_argument("-dr", "--dropout_rate", type=float, default=0.2)
parser.add_argument("-ne", "--num_epochs", type=int, default=500)
parser.add_argument("-sd", "--seed", type=int, default=1)
parser.add_argument("-pat", "--es_patience", type=int, default=50)
parser.add_argument("-esmd", "--es_min_delta", type=int, default=0.01)
parser.add_argument("-esrt", "--es_rate_threshold", type=int, default=0.0008)
parser.add_argument("-lde", "--layer_dim_ex", type=int, default=3)
parser.add_argument("-v", "--verbose", type=bool, default=True)
parser.add_argument("-sam", "--sample", type=bool, default=True)
parser.add_argument("-prep", "--preprocessing_type", default="SVD", choices=["None", "SVD"])

args, unk = parser.parse_known_args()
torch.set_num_threads(1)
pipeline(**vars(args))
