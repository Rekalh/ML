# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 14:55:32 2026

@author: Diogenis
"""

import numpy as np
import matplotlib.pyplot as plt


# Function we want to approximate
def f(x):
    return np.sin(x ** 2)

# ReLU activator function
def ReLU(x):
    return np.maximum(0, x)

# ReLU derivative
def ReLU_prime(x):
    return (x > 0).astype(float)

# Input
Np = 1000
X = np.linspace(-2 * np.pi, 2 * np.pi, Np).reshape((1, Np))

# Number of neurons
Nneurons1 = 150
Nneurons2 = 50

# Number of epochs
Nepochs = 50000

# Learning rate
a = 0.001

# Initialize the weights randomly
w1 = np.random.randn(Nneurons1, 1) * np.sqrt(2.0 / 1)
w2 = np.random.randn(Nneurons2, Nneurons1) * np.sqrt(2.0 / Nneurons1)
v = np.random.randn(Nneurons2, 1) * np.sqrt(1.0 / Nneurons2)

# Initialize biases
random_activation_points_1 = np.random.uniform(np.min(X), np.max(X), size=(Nneurons1, 1))
b1 = -w1 * random_activation_points_1
b2 = np.ones((Nneurons2, 1)) * 0.05

b_out = np.zeros((1,1))

# Train NN
y_predHistory = np.zeros((Np, Nepochs))
y = f(X)

# Add momentum (Dynamic learning rate)
v_dw1 = np.zeros_like(w1)
v_db1 = np.zeros_like(b1)
v_dw2 = np.zeros_like(w2)
v_db2 = np.zeros_like(b2)
v_dv = np.zeros_like(v)
v_db_out = np.zeros_like(b_out)

beta = 0.9
for e in range(Nepochs):
    
    # Find prediction
    Z1 = np.dot(w1, X) + b1
    A1 = ReLU(Z1)
    Z2 = np.dot(w2, A1) + b2
    A2 = ReLU(Z2)
    y_pred = np.dot(v.T, A2) + b_out
    
    # Back propagation calculations
    err = y_pred - y
    err_h2 = np.dot(v, err) * ReLU_prime(Z2)
    err_h1 = np.dot(w2.T, err_h2) * ReLU_prime(Z1)
    dv = np.dot(err, A2.T).T / Np
    dw1 = err_h1.dot(X.T) / Np
    db1 = np.sum(err_h1, axis=1, keepdims=True) / Np
    dw2 = err_h2.dot(A1.T) / Np
    db2 = np.sum(err_h2, axis=1, keepdims=True) / Np
    db_out = np.sum(err, axis=1, keepdims=True) / Np

    # Gradient descend calculations
    v_dw1 = beta * v_dw1 + a * dw1
    v_db1 = beta * v_db1 + a * db1
    v_dw2 = beta * v_dw2 + a * dw2
    v_db2 = beta * v_db2 + a * db2
    v_dv = beta * v_dv + a * dv
    v_db_out = beta * v_db_out + a * db_out

    # Update weights
    v = v - v_dv
    w1 = w1 - v_dw1
    b1 = b1 - v_db1
    w2 = w2 - v_dw2
    b2 = b2 - v_db2
    b_out = b_out - v_db_out

    y_predHistory[:, e] = y_pred

# %% Plotting (Thanks Gemini)

# Plot final result
plt.figure()
plt.title(f'Epochs: {Nepochs}, Learning rate: {a}, Neurons: {(Nneurons1, Nneurons2)}')
plt.plot(X.flatten(), f(X).flatten(), label='True')
plt.plot(X.flatten(), y_predHistory[:, -1].flatten(), label='Best NN Prediction')
plt.legend()
plt.show()

import matplotlib.animation as animation

# Initialize the canvas
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(X[0, 0], X[0, -1])
ax.set_ylim(np.min(y), np.max(y))
ax.grid(True)
ax.set_title(f"Training: approximating f(x) with {(Nneurons1, Nneurons2)} Neurons, a = {a}", fontsize=14)

# Plot static background (The true function target)
ax.plot(X.flatten(), y.flatten(), label="True value", color="black", linewidth=2)

# Create an empty line placeholder for the dynamic network prediction
pred_line, = ax.plot([], [], label="NN Prediction", color="orange", linestyle="-", linewidth=2)
ax.legend(loc="upper right")

# Set up an on-screen text tracker for our epochs
epoch_text = ax.text(0.05, 0.9, "", transform=ax.transAxes, fontsize=12, fontweight='bold')

# Animation configurations
step = 20                        # Skip every 20 epochs to keep playback fast
num_frames = Nepochs // step     # 10000 / 20 = 500 frames total

# The sequential update handler called by FuncAnimation
def update_plot(frame):
    epoch_idx = frame * step
    
    # Rigid protection boundary check against array index clipping
    if epoch_idx >= Nepochs:
        epoch_idx = Nepochs - 1
        
    # Feed coordinates to line graph object
    pred_line.set_data(X.flatten(), y_predHistory[:, epoch_idx])
    epoch_text.set_text(f"Epoch: {epoch_idx:,}")
    
    return pred_line, epoch_text

# Run the live loop
ani = animation.FuncAnimation(
    fig, 
    update_plot, 
    frames=num_frames, 
    interval=20,     # 20ms delay between frames 
    blit=True, 
    repeat=False
)

plt.show()
