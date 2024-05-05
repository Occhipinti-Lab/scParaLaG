"""
----------------------------------------------------------------------------------------------------------------------------------------
Description:                                                                                                                            |
    This module contains the scParaLaG framework for building scParaLaG models.                                                         |
                                                                                                                                        |
Copyright:                                                                                                                              |
    Copyright © 2024. All rights reserved.                                                                                              |                                                                             |
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


"""
******************************************************** IMPORTANT DESCRIPTION **********************************************************

Description of `conv_flow` and `agg_flow` Parameters and Their Usage

1. `conv_flow` (list of str):
   This parameter defines the sequence of convolution layer types to be utilized in the network.
   The supported layer types include:
    - 'gat': Graph Attention Network layer.
    - 'sage': GraphSAGE layer.
    - 'gconv': Graph Convolution Network layer.

2. `agg_flow` (list of str):
   This parameter specifies the aggregation methods corresponding to each layer type defined in `conv_flow`.
   The supported aggregation methods are:
    - For 'gat':
      - 'mean'
      - 'None'
    - For 'sage':
      - 'mean'
      - 'sum'
      - 'max_pool'
      - 'mean_pool'
      - 'lstm'
    - For 'gconv':
      - 'None' (Note: The standard GCN does not utilize varying aggregation methods).

Important Note:
Please ensure that each entry in `agg_flow` corresponds exactly to the layer type at the same index in `conv_flow`.
This alignment is crucial for the proper functioning of the network layers and their respective aggregation methods.
"""

import os
os.environ['DGLBACKEND'] = 'pytorch'
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
from dgl.nn.pytorch.conv import GATConv, GraphConv, SAGEConv

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class LayerAttention(nn.Module):
    """
    A layer to compute attention weights over different layers' outputs.

    This layer calculates the attention weights for a set of layer outputs and produces a weighted sum of these outputs.

    Attributes
    ----------
    attention_weights : nn.Parameter
        The learnable attention weights.

    Methods
    -------
    forward(layer_outputs)
        Applies attention mechanism on the given layer outputs.

    Parameters
    ----------
    num_layers : int
        Number of layers for which attention weights are to be learned.
    num_feat : int
        The size of each feature vector.

    Usage
    -----
    - Instantiate this class and pass the outputs of different layers to the forward method to get a weighted sum.
    """
    def __init__(self, num_layers, num_feat):
        super(LayerAttention, self).__init__()
        self.attention_weights = nn.Parameter(torch.empty(num_layers, num_feat))
        nn.init.kaiming_normal_(self.attention_weights, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, layer_outputs):
        normalized_weights = F.softmax(self.attention_weights, dim=0)
        weighted_sum = sum(w * output for w, output in zip(normalized_weights, layer_outputs))
        return weighted_sum

class scParaLaG(nn.Module):
    """
    A model for graph neural network with parameterized layer aggregation.

    This model includes convolutional, residual, and linear layers with an attention mechanism for aggregating
    the outputs of these layers.

    Attributes
    ----------
    conv_layers : nn.ModuleList
        A list of convolutional layers.
    residual_layers : nn.ModuleList
        A list of residual layers corresponding to the convolutional layers.
    linear_layers : nn.ModuleList
        A list of linear layers for transforming the output of the convolutional and residual layers.
    layer_attention : LayerAttention
        The layer attention mechanism for aggregating layer outputs.
    final_linear : nn.Linear
        The final linear layer for output transformation.
    dropout : nn.Dropout
        The dropout layer for regularization.

    Methods
    -------
    forward(g, features, conv_flow, agg_flow)
        Propagates input through the model layers and returns the final output.

    Parameters
    ----------
    num_heads : int
        Number of attention heads.
    FEATURE_SIZE : int
        Size of input features.
    OUTPUT_SIZE : int
        Size of output features.
    hidden_size : int
        Size of hidden layer features.
    conv_flow : list
        List of convolution layer types.
    agg_flow : list
        List of aggregation types for each convolution layer.
    act : str
        Activation function to use.
    dropout_rate : float
        Dropout rate for regularization.

    Usage
    -----
    - Instantiate this class and call the forward method with a graph, its features, and the convolution and aggregation flows.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args


        num_heads = self.args.num_heads
        self.activation = self.get_activation_function(self.args.act)

        self.conv_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.residual_layers = nn.ModuleList()
        self.layer_attention = LayerAttention(len(self.args.conv_flow) + self.args.layer_dim_ex, self.args.hidden_size)
        self.final_linear = nn.Linear(self.args.hidden_size, self.args.OUTPUT_SIZE)
        self.dropout = nn.Dropout(self.args.dropout_rate)

        in_feats = self.args.FEATURE_SIZE
        out_feats = self.args.hidden_size

        for i, (layer_type, aggregate_type) in enumerate(zip(self.args.conv_flow, self.args.agg_flow)):
            conv_layer = self.layer_factory(layer_type, in_feats, out_feats, aggregate_type, num_heads)
            self.conv_layers.append(conv_layer)

            out_in_dim = out_feats * num_heads if (layer_type == 'gat' and aggregate_type is None) else out_feats
            linear_layer = nn.Linear(in_feats, out_in_dim)
            self.linear_layers.append(linear_layer)

            residual_layer = nn.Linear(out_in_dim, self.args.hidden_size)
            self.residual_layers.append(residual_layer)

    def forward(self, g, features):
        layer_outputs = []

        for i, (conv_layer, residual_layer, linear_layer) in enumerate(
          zip(self.conv_layers, self.residual_layers, self.linear_layers)):
            conv_h = conv_layer(g, features)

            if self.args.conv_flow[i] == 'gat':
                if self.args.agg_flow[i] == 'mean':
                    conv_h = conv_h.mean(1)
                elif self.args.agg_flow[i] is None:
                    conv_h = conv_h.flatten(1)

            linear_h = linear_layer(features)
            conv_linear = conv_h + linear_h

            conv_linear_h = residual_layer(conv_linear)
            conv_linear_h = self.activation(conv_linear_h)
            conv_linear_h = self.dropout(conv_linear_h)
            layer_outputs.append(conv_linear_h)

        h = self.layer_attention(layer_outputs)
        h = self.final_linear(h)
        return h

    @staticmethod
    def layer_factory(layer_type, in_feats, out_feats, aggregate_type=None,
                      num_heads=None):
        """
        Factory method to create a layer based on the specified type.

        This method creates a graph convolutional layer of a specified type. It supports various types of layers like GATConv,
        GraphConv, and SAGEConv.

        Parameters
        ----------
        layer_type : str
            The type of layer to create ('gat', 'gconv', 'sage').
        in_feats : int
            The number of input features.
        out_feats : int
            The number of output features.
        aggregate_type : str, optional
            The type of aggregator for Conv layers.
        num_heads : int, optional
            The number of attention heads for GATConv layers.

        Returns
        -------
        nn.Module
            The created graph convolutional layer.

        Raises
        ------
        ValueError
            If the layer type is unknown.

        Usage
        -----
        - This method is used internally by the scParaLaG class to instantiate layers based on configuration parameters.
        """
        if layer_type == 'gat':
            return GATConv(in_feats=in_feats, out_feats=out_feats,
                           num_heads=num_heads, activation=F.leaky_relu)
        elif layer_type == 'gconv':
            return GraphConv(in_feats=in_feats, out_feats=out_feats)
        elif layer_type == 'sage':
            return SAGEConv(in_feats=in_feats, out_feats=out_feats,
                            aggregator_type=aggregate_type)
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

    @staticmethod
    def get_activation_function(name):
      """
      Get the specified activation function.

      This method returns an activation function based on the given name. It supports various activation functions like ReLU,
      Sigmoid, Tanh, Leaky ReLU, etc.

      Parameters
      ----------
      name : str
          The name of the activation function.

      Returns
      -------
      callable
          The corresponding activation function.

      Usage
      -----
      - This method is used internally by the scParaLaG class to set the activation function based on configuration parameters.
      """
      activation_functions = {
            'relu': F.relu,
            'relu6': F.relu6,
            'sigmoid': torch.sigmoid,
            'tanh': torch.tanh,
            'leaky_relu': F.leaky_relu,
            'selu': F.selu,
            'gelu': F.gelu,
            'rrelu': F.rrelu
      }
      return activation_functions.get(name)
    