# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 21:21:49 2026

@author: Diogenis
"""

from PINN import PINN, ground_truth, train_loop, PINN_loss
from MLP import MLP, MLP_loss
from torch import nn
import torch

Da = 1.0
n = 2

x = torch.linspace(0, 1, 20).unsqueeze(1)
y = ground_truth(x, Da, n)
offset = torch.randn_like(y) * 0.1
y_gauss = y + offset

from torch.utils.data import TensorDataset, random_split, DataLoader
dataset = TensorDataset(x, y_gauss)
train_size = int(1.0 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=train_size)

epochs = 2000
a = 5.0
b = 100
learning_rate = 1e-3

x_high_res = torch.linspace(0, 1, 10000).unsqueeze(1)
evolution_history_PINN = []
evolution_history_MLP = []

modelPINN = PINN(1, [(32, nn.Tanh()), (32, nn.Tanh()), (32, nn.Tanh()), (1, nn.Identity())])
modelMLP = MLP(1, [(32, nn.Tanh()), (32, nn.Tanh()), (32, nn.Tanh()), (1, nn.Identity())])
optimizerPINN = torch.optim.Adam(modelPINN.parameters(), lr=learning_rate)
optimizerMLP = torch.optim.Adam(modelMLP.parameters(), lr=learning_rate)

for epoch in range(epochs):
    print(f"Epoch {epoch}\n-------------------------------")
    train_loop(train_loader, modelPINN, PINN_loss, optimizerPINN, (Da, n), (a, b), epoch)
    train_loop(train_loader, modelMLP, MLP_loss, optimizerMLP, (Da, n), (a, b), epoch)
    
    if epoch % 20 == 0:
        y_snap_PINN = modelPINN(x_high_res).detach()
        y_snap_MLP = modelMLP(x_high_res).detach()
        evolution_history_PINN.append(y_snap_PINN)
        evolution_history_MLP.append(y_snap_MLP)

import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.1)
ax.set_xlabel('Dimensionless Reactor Length (z)')
ax.set_ylabel('Dimensionless Concentration (C_A)')
ax.grid(True, linestyle='--', alpha=0.6)

y_true_high_res = ground_truth(x_high_res, Da, n)
ax.plot(x_high_res.numpy(), y_true_high_res.numpy(), 'k-', linewidth=2.5, label='Analytical Ground Truth', zorder=10)

x_train = train_dataset.dataset.tensors[0][train_dataset.indices]
y_train = train_dataset.dataset.tensors[1][train_dataset.indices]
ax.scatter(x_train.numpy(), y_train.numpy(), color='red', s=50, label='Noisy Sensor Data', zorder=11)

pinn_line, = ax.plot([], [], color='gold', linewidth=3, label='PINN Prediction', zorder=13)
mlp_line, = ax.plot([], [], color='blue', linewidth=3, label='MLP Prediction', zorder=12)
title = ax.set_title('PINN Evolution: Overcoming Sensor Noise (Epoch: 0)')
ax.legend()

def update(frame):
    y_snap_PINN = evolution_history_PINN[frame]
    y_snap_MLP = evolution_history_MLP[frame]
    pinn_line.set_data(x_high_res.numpy(), y_snap_PINN.numpy())
    mlp_line.set_data(x_high_res.numpy(), y_snap_MLP.numpy())
    
    title.set_text(f'PINN Evolution: Overcoming Sensor Noise (Epoch: {frame * 20})')
    return pinn_line, mlp_line, title

ani = animation.FuncAnimation(fig, update, frames=len(evolution_history_PINN), blit=True, interval=200)

# ani.save('pinn_training.gif', writer='pillow', fps=15)
# print("Animation saved as pinn_training.gif!")

plt.show()
