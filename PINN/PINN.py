# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 15:20:47 2026

@author: Diogenis
"""

import torch
from torch import nn

class PINN(nn.Module):
    
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

def PINN_loss(model, x, y_exp, y_pred, Da, n, a, b):
    dCAdz = torch.autograd.grad(outputs=y_pred, inputs=x, create_graph=True, grad_outputs=torch.ones_like(y_pred))
    bc_pred = model(torch.tensor([0.0], device=next(model.parameters()).device))
    loss_PFR = torch.mean((1.0 / Da * dCAdz[0] + y_pred ** n) ** 2)
    loss_data = torch.mean((y_exp - y_pred) ** 2)
    loss_bc = (bc_pred[0] - 1) ** 2
    
    loss = loss_data + a * loss_PFR + b * loss_bc
    return loss

def ground_truth(x, Da, n):
    y_truth = torch.zeros_like(x)
    
    if n > 1:
        y_truth = (1 + Da * (n - 1) * x) ** (1 / (1 - n))
    elif n == 0:
        y_truth = 1 - Da * x
    else:
        y_truth = torch.exp(-Da * x)

    return y_truth

def train_loop(train_loader, model: nn.Module, loss, optimizer, reactor_params, hyperparams, epoch):
    Da, n = reactor_params
    a, b = hyperparams
    model.train()
    
    for x, y in train_loader:
        x.requires_grad = True
        y_pred = model(x)
        l = loss(model, x, y, y_pred, Da, n, a, b)
        l.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if epoch % 100 == 0:
            l1, current = l.item(), epoch
            print(f"Loss: {l1:>7f}  [{current:>5d}]")

def test_loop(test_loader, model: nn.Module, loss, reactor_params, hyperparams):
    Da, n = reactor_params
    a, b = hyperparams
    model.eval()
    
    l = []
    for x, y in test_loader:
        x.requires_grad = True
        y_pred = model(x)
        l.append(loss(model, x, y, y_pred, Da, n, a, b).item())
    
    avg_loss = sum(l) / len(l)
    print(f"Avg loss: {avg_loss:>8f} \n")
