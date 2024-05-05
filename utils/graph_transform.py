"""
----------------------------------------------------------------------------------------------------------------------------------------
Description:                                                                                                                            |
    This module is used for creating graphs suitable for scParaLaG models.                                                              |
                                                                                                                                        |
Copyright:                                                                                                                              |
    Copyright © 2024. All rights reserved.                                                                                              |                                                                            |
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

import dgl
import torch
import scipy.stats
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from dance.data.base import Data

class GraphCreator:
    def __init__(self, preprocess_type, n_neighbors=20, n_components=1200,
                 metric='euclidean'):
        """
        Initialize the GraphCreator class for constructing k-nearest neighbor graphs.

        Parameters
        ----------
        preprocess_type : str
            Specifies the type of preprocessing to perform on the data. Options are 'None' and 'SVD'.
        n_neighbors : int, optional
            The number of nearest neighbors to consider for graph construction.
        n_components : int, optional
            Number of components to retain if SVD is used for preprocessing.
        metric : str, optional
            The distance metric to use for kneighbors graph construction. Defaults to 'euclidean'.

        Raises
        ------
        ValueError
            If the preprocess_type is not among the allowed options.
        """
        if preprocess_type not in ['None', 'SVD']:
            raise ValueError("preprocess_type must be 'None' or 'SVD'")
        self.preprocess_type = preprocess_type
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.metric = metric

    def _build_knn_graph(self, features):
        """
        Build a k-nearest neighbor graph using the provided features.

        Parameters
        ----------
        features : array-like
            The feature set based on which the k-nearest neighbor graph is to be constructed.

        Returns
        -------
        graph : dgl.DGLGraph
            The constructed graph where nodes represent samples and edges represent neighbor relationships.
        """
        A = kneighbors_graph(features, n_neighbors=self.n_neighbors,
                             metric=self.metric, mode='connectivity',
                             include_self=True)
        graph = dgl.from_scipy(A)
        return graph

    def _create_graphs(self, train_features, val_features, test_features):
        """
        Create training and testing graphs using the provided feature sets.

        Parameters
        ----------
        train_features : array-like
            The feature set for training data.
        val_features : array-like
            The feature set for validation data.
        test_features : array-like
            The feature set for testing data.

        Returns
        -------
        tuple
            A tuple containing the training, validation and testing graphs (dgl.DGLGraph, dgl.DGLGraph, dgl.DGLGraph).
        """
        train_graph = self._build_knn_graph(train_features)
        val_graph = self._build_knn_graph(val_features)
        test_graph = self._build_knn_graph(test_features)
        return train_graph, val_graph, test_graph

    def __call__(self, data: Data) -> Data:
        """
        Call method to process the data and create graphs.

        Parameters
        ----------
        data : Data
            The data object containing training and testing data.

        Returns
        -------
        Data
            The updated data object with training and testing graphs added.
        train_label : torch.Tensor
            Labels corresponding to the training data.
        val_label : torch.Tensor
            Labels corresponding to the validation data.
        test_label : torch.Tensor
            Labels corresponding to the testing data.
        ftl_shape : tuple
            Feature and Label size of the dataset.
        """
        input, label = data.get_train_data(return_type="numpy")
        test_input, test_label = data.get_test_data(return_type="numpy")
        train_input, val_input, train_label, val_label = train_test_split(
            input, label, test_size=0.05, random_state=42)



        if self.preprocess_type == 'SVD':
            embedder = TruncatedSVD(n_components=self.n_components)
            train_input = embedder.fit_transform(
                scipy.sparse.csr_matrix(train_input))
            val_input = embedder.transform(scipy.sparse.csr_matrix(val_input))
            test_input = embedder.transform(scipy.sparse.csr_matrix(test_input))

        train_graph, val_graph, test_graph = self._create_graphs(
            train_input, val_input, test_input)
        train_graph.ndata['feat'] = torch.tensor(train_input, dtype=torch.float32)
        val_graph.ndata['feat'] = torch.tensor(val_input, dtype=torch.float32)
        test_graph.ndata['feat'] = torch.tensor(test_input, dtype=torch.float32)

        data.data.uns['gtrain'] = train_graph
        data.data.uns['gval'] = val_graph
        data.data.uns['gtest'] = test_graph
        ftl_shape = (train_input.shape[1], train_label.shape[1])

        return data, train_label, val_label, test_label, ftl_shape
    