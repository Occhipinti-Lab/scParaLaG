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

import argparse, os, sys, yaml, torch
from argparse import Namespace
from dance.datasets.multimodality import ModalityPredictionDataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graph_transform import GraphCreator
from train import scParaLaGWrapper

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def validate_config(cfg):
    required = {
        "subtask": str, "batch_size": int, "act": str,
        "conv_flow": list, "agg_flow": list, "device": str,
        "learning_rate": float, "hidden_size": int, "num_heads": int,
        "n_neigbours": int, "n_components": int, "metric": str,
        "dropout_rate": float, "num_epochs": int, "seed": int,
        "es_patience": int, "es_min_delta": float, "es_rate_threshold": float,
        "layer_dim_ex": int, "verbose": bool, "sample": bool,
        "preprocessing_type": str, "variant": str, "num_clusters": int,
        "processed": bool
    }
    for k, t in required.items():
        if k not in cfg and k != "processed":
            raise ValueError(f"Missing config: {k}")
        if k in cfg and not isinstance(cfg[k], t):
            if k == "agg_flow" and cfg[k] == [None]: continue
            raise TypeError(f"{k} must be {t}, got {type(cfg[k])}")
    if cfg.get("agg_flow") is None:
        cfg["agg_flow"] = [None]
    return cfg

def load_or_process_data(cfg, data):
    name = f"processed_data_{cfg['subtask']}"
    folder = "./processed_data"
    paths = {k: os.path.join(folder, f"{name}_{k}.pt") for k in 
             ["train_graph", "val_graph", "test_graph", "train_label", "val_label", "test_label", "feature_size", "output_size"]}
    flag_path = os.path.join(folder, f"{name}_processed.flag")
    
    if os.path.exists(flag_path) and all(os.path.exists(p) for p in paths.values()):
        data.data.uns['gtrain'] = torch.load(paths["train_graph"])
        data.data.uns['gval'] = torch.load(paths["val_graph"])
        data.data.uns['gtest'] = torch.load(paths["test_graph"])
        return data, *(torch.load(paths[k]).numpy() for k in ["train_label", "val_label", "test_label"]), \
               torch.load(paths["feature_size"]).item(), torch.load(paths["output_size"]).item(), True
    else:
        os.makedirs(folder, exist_ok=True)
        g = GraphCreator(cfg["preprocessing_type"], cfg["n_neigbours"], cfg["n_components"], cfg["metric"])
        data, y_train, y_val, y_test, shape = g(data)
        feature_size, output_size = (cfg["n_components"], shape[1]) if cfg["preprocessing_type"] == "SVD" else shape
        torch.save(data.data.uns['gtrain'], paths["train_graph"])
        torch.save(data.data.uns['gval'], paths["val_graph"])
        torch.save(data.data.uns['gtest'], paths["test_graph"])
        torch.save(torch.tensor(y_train), paths["train_label"])
        torch.save(torch.tensor(y_val), paths["val_label"])
        torch.save(torch.tensor(y_test), paths["test_label"])
        torch.save(torch.tensor(feature_size), paths["feature_size"])
        torch.save(torch.tensor(output_size), paths["output_size"])
        with open(flag_path, 'w'): pass
        return data, y_train, y_val, y_test, feature_size, output_size, True

def pipeline(cfg):
    dataset = ModalityPredictionDataset(cfg["subtask"], preprocess='feature_selection')
    data = dataset.load_data()
    if cfg["subtask"] == "openproblems_bmmc_cite_phase2_mod2":
        cfg["preprocessing_type"] = "None"
    
    data, y_train, y_val, y_test, cfg["FEATURE_SIZE"], cfg["OUTPUT_SIZE"], _ = load_or_process_data(cfg, data)
    
    for seed in range(1, cfg.get("num_seeds_loop", 6)):
        cfg["seed"] = seed
        try:
            from dance.models.multimodal.scparalag_gcn import scParaLaGWrapper
            model = scParaLaGWrapper(Namespace(**cfg))
        except ImportError:
            print("Missing scParaLaGWrapper.")
            continue
        model.fit(data.data.uns['gtrain'], data.data.uns['gval'], data.data.uns['gtest'],
                  torch.tensor(y_train), torch.tensor(y_val), torch.tensor(y_test),
                  num_epochs=cfg["num_epochs"], batch_size=cfg["batch_size"],
                  verbose=cfg["verbose"], es_patience=cfg["es_patience"],
                  es_min_delta=cfg["es_min_delta"], es_rate_threshold=cfg["es_rate_threshold"],
                  learning_rate=cfg["learning_rate"], sample=cfg["sample"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args, _ = parser.parse_known_args()
    
    config = vars(parser.parse_args())
    if os.path.exists(args.config):
        config.update(load_config(args.config))
    
    config = validate_config(config)
    torch.set_num_threads(1)
    pipeline(config)
