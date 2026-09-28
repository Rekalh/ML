# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 14:55:32 2026

@author: Diogenis
"""

import numpy as np
import matplotlib.pyplot as plt


# Function we want to approximate
def f(x):
    return np.cos(3 * x) * x

# ReLU activator function
def sigma(x):
    return np.maximum(0, x)

# ReLU derivative
def sigma_prime(x):
    return (x > 0).astype(float)

# Input
Np = 50
X = np.linspace(-np.pi, np.pi).reshape((1, Np))

# Number of neurons
Nneurons = 150

# Number of epochs
Nepochs = 50000

# Learning rate
a = 0.01

# Initialize the weights randomly
w = np.random.randn(Nneurons, 1) * 1
v = np.random.randn(Nneurons, 1) * 0.1

# Initialize biases
random_activation_points = np.random.uniform(np.min(X), np.max(X), size=(Nneurons, 1))
b = -w * random_activation_points # This guarantees hinges are born

b_out = np.zeros((1,1))

# Train NN
y_predHistory = np.zeros((Np, Nepochs))
y = f(X)

# Add momentum (Dynamic learning rate)
v_dw = np.zeros_like(w)
v_dv = np.zeros_like(v)
v_db = np.zeros_like(b)
v_db_out = np.zeros_like(b_out)

beta = 0.9
for e in range(Nepochs):
    
    # Find prediction
    Z = np.dot(w, X) + b
    A = sigma(Z)
    y_pred = np.dot(v.T, A) + b_out
    
    # Back propagation calculations
    err = y_pred - y
    err_hidden = np.dot(v, err) * sigma_prime(Z)
    dv = np.dot(A, err.T) / Np
    dw = err_hidden.dot(X.T) / Np
    db = np.sum(err_hidden, axis=1, keepdims=True) / Np
    db_out = np.sum(err) / Np

    # Gradient descend calculations
    v_dw = beta * v_dw + a * dw
    v_dv = beta * v_dv + a * dv
    v_db = beta * v_db + a * db
    v_db_out = beta * v_db_out + a * db_out

    # Update weights
    v = v - v_dv
    w = w - v_dw
    b = b - v_db
    b_out = b_out - v_db_out

    y_predHistory[:, e] = y_pred

# %% Plotting (Thanks Gemini)

# Plot final result
plt.figure()
plt.title(f'Epochs: {Nepochs}, Learning rate: {a}, Neurons: {Nneurons}')
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
ax.set_title(f"Training: approximating f(x) with {Nneurons} Neurons, a = {a}", fontsize=14)

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
