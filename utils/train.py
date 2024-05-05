"""
----------------------------------------------------------------------------------------------------------------------------------------
Description:                                                                                                                            |
    This module is used for the custom training of scParaLaG models.                                                                    |
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
import sys
import os
import math
import warnings
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.scParaLaG import scParaLaG

# Ignore all warnings
warnings.filterwarnings("ignore")


class CustomEarlyStopping:
    """
    Custom early stopping mechanism for training scParaLaG.

    This class provides a way to stop training early if the loss does not improve sufficiently over a given number of epochs.
    It tracks the best loss and the number of epochs since there was a significant improvement.

    Attributes
    ----------
    patience : int
        The number of epochs to wait for improvement before stopping the training.
    min_delta : float
        The minimum change in the loss to qualify as an improvement.
    rate_threshold : float
        The minimum rate of improvement that must be exceeded to reset the patience countdown.
    best_loss : float
        The best loss encountered during training.
    best_epoch : int
        The epoch number where the best loss was encountered.
    epochs_since_improvement : int
        The number of epochs since the last significant improvement in loss.
    prev_loss : float
        The loss from the previous epoch.

    Methods
    -------
    update(current_loss, current_epoch)
        Updates the state based on the loss of the current epoch.

    should_stop()
        Determines if training should be stopped based on the early stopping criteria.

    Parameters
    ----------
    patience : int, optional
        The number of epochs with no improvement after which training will be stopped. Default is 10.
    min_delta : float, optional
        Minimum change in the monitored quantity to qualify as an improvement. Default is 0.001.
    rate_threshold : float, optional
        Minimum rate of improvement to reset the patience. Default is 0.001.

    Usage
    -----
    - Instantiate this class and use the 'update' method after each epoch to update the internal state.
    - Use the 'should_stop' method to check if the training should be stopped.
    """
    def __init__(self, patience=10, min_delta=0.001, rate_threshold=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.rate_threshold = rate_threshold
        self.best_loss = float('inf')
        self.best_epoch = -1
        self.epochs_since_improvement = 0
        self.prev_loss = float('inf')

    def update(self, current_loss, current_epoch):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.best_epoch = current_epoch
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1

        rate_of_improvement = (self.prev_loss - current_loss) / self.prev_loss
        if rate_of_improvement < self.rate_threshold:
            self.epochs_since_improvement += 1  # Additional penalty for slow improvement
        print(f'Patience grace period: {self.patience - self.epochs_since_improvement}')

        self.prev_loss = current_loss

    def should_stop(self):
        return self.epochs_since_improvement >= self.patience

class scParaLaGWrapper:
    def __init__(self, args):
        """
        Initialize the scFlowGNNWrapper with provided arguments.

        Parameters
        ----------
        args : object
            Contains all the necessary configuration and model parameters.
        """
        self.args = args
        self.model = scParaLaG(self.args).to(self.args.device)

    def predict(self, graph, idx=None):
        """
        Predict the outputs for a given graph and node features.

        Parameters
        ----------
        graph : dgl.DGLGraph
            The graph input, containing nodes and edges representing the data.
        idx : Iterable[int], optional
            Node indices for which predictions are to be made. If None, predictions are made for all nodes.

        Returns
        -------
        outputs : torch.Tensor
            The predicted outputs as a tensor.
        """
        self.model.eval()
        with torch.no_grad():
            if idx is not None:
                graph = graph.subgraph(idx).to(self.args.device)
            else:
                graph = graph.to(self.args.device)
            outputs = self.model(graph, graph.ndata['feat'])
        return outputs

    def score(self, graph, labels, idx=None):
        """
        Compute the Root Mean Square Error (RMSE) score for the model's predictions.

        Parameters
        ----------
        graph : dgl.DGLGraph
            The graph input, containing nodes and edges.
        labels : torch.Tensor
            True labels for the nodes.
        idx : Iterable[int], optional
            Node indices to be used for scoring. If None, all nodes are used.

        Returns
        -------
        RMSE : float
            The computed RMSE score.
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.predict(graph, idx)
            relevant_labels = labels[idx] if idx is not None else labels
            mse_loss = F.mse_loss(predictions, relevant_labels.to(self.args.device))
            RMSE = math.sqrt(mse_loss.item())
            return RMSE

    def fit(self, train_graph, val_graph, test_graph, train_label, val_label,
            test_label, num_epochs=500, batch_size=520,verbose=True,
            es_patience=20, es_min_delta=0.01, es_rate_threshold=0.001,
            learning_rate=0.000028, sample=True):
        """
        Train and validate the model using the provided training and testing graphs and labels.

        Parameters
        ----------
        train_graph : dgl.DGLGraph
            The graph representing the training data.
        val_graph : dgl.DGLGraph
            The graph representing the validation data.
        test_graph : dgl.DGLGraph
            The graph representing the testing data.
        train_label : torch.Tensor
            Labels corresponding to the training data.
        val_label : torch.Tensor
            Labels corresponding to the validation data.
        test_label : torch.Tensor
            Labels corresponding to the testing data.
        num_epochs : int
            Number of epochs for training the model.
        batch_size : int, optional
            Size of the batches used during training.
        verbose : bool, optional
            Flag to enable printing detailed logs.
        es_patience : int, optional
            Patience parameter for early stopping.
        es_min_delta : int, optional
            Minimum delta parameter for early stopping.
        es_rate_threshold : int, optional
            Rate threshold parameter for early stopping.
        learning_rate : float, optional
            Initial learning rate.
        sample : bool, optional
            Flag to enable batch sampling.

        Returns
        -------
        None
        """
        torch.manual_seed(self.args.seed)
        if self.args.device == "cuda":
            torch.cuda.manual_seed_all(self.args.seed)
        early_stopping = CustomEarlyStopping(patience=es_patience,
                                             min_delta=es_min_delta,
                                             rate_threshold=es_rate_threshold)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        save_dir = "model_checkpoints"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0
            num_nodes = len(train_graph.ndata['feat'])
            indices = torch.randperm(num_nodes) if sample else torch.arange(num_nodes)

            for i in range(0, num_nodes, batch_size):
                batch_indices = indices[i:i + batch_size]
                subgraph = train_graph.subgraph(batch_indices).to(self.args.device)
                batch_labels = train_label[batch_indices].to(self.args.device)

                optimizer.zero_grad()
                outputs = self.model(subgraph, subgraph.ndata['feat'])
                loss = F.mse_loss(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = math.sqrt(epoch_loss / (len(indices) / batch_size))

            # Validation and Test Phase
            val_loss = self.score(val_graph, val_label)
            test_loss = self.score(test_graph, test_label)

            if verbose:
                print('---------------------------------')
                print(f'* Epoch {epoch + 1}/{num_epochs}')
                print(f'* Training Loss: {avg_loss:.4f} ')
                print(f'* Validation Loss: {val_loss:.4f}')
                print(f'* Test Loss: {test_loss:.4f}')
                print('---------------------------------')

            # Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_since_improvement = 0
                torch.save(self.model.state_dict(), os.path.join(
                    save_dir, 'best_model.pth'))
                print(f'Model saved: Epoch {epoch+1}, Validation Loss: {val_loss}')

            # Update early stopping with current validation loss
            early_stopping.update(val_loss, epoch)

            # Check if early stopping is triggered
            if early_stopping.should_stop():
                print(f"Early stopping triggered at epoch {epoch}")
                break

        # Load best model weights
        self.model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))
        self.model.eval()
        result = pd.DataFrame({'rmse': [], 'seed': [], 'subtask': [], 'method': []})
        test_graph = test_graph.to(self.args.device)
        test_feat = test_graph.ndata['feat'].to(self.args.device)
        test_label = test_label.to(self.args.device)
        outputs = self.model(test_graph, test_feat).to(self.args.device)

        rmse = math.sqrt(F.mse_loss(outputs, test_label))


        result = result.append(
            {
                'rmse': rmse,
                'seed': self.args.seed,
                'subtask': self.args.subtask,
                'method': 'scParaLaG',
            }, ignore_index=True)
        print(result)