# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 00:49:51 2026

@author: Diogenis
"""

import torch
from torch import nn

class MLP(nn.Module):
    
    def __init__(self, input_size, layer_info: list):
        super().__init__()
        self.layers = nn.ModuleList()
        
        self.input_size = input_size
        for size, activation_type in layer_info:
            self.layers.append(nn.Linear(self.input_size, size))
            self.input_size = size
            self.layers.append(activation_type)
        
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        self.to(self.device)

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
            
        return output

def MLP_loss(model, x, y_exp, y_pred, Da, n, a, b):
    loss = torch.mean((y_exp - y_pred) ** 2)
    return loss
