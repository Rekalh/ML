# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 02:34:06 2026

@author: Diogenis
"""

import numpy as np
from sklearn.datasets import fetch_openml


# ReLU activator function
def ReLU(z):
    return np.maximum(0, z)

# ReLU derivative
def ReLU_prime(z):
    return (z > 0).astype(float)

def softmax(z):
    shift_z = z - np.max(z, axis=0, keepdims=True)
    exps = np.exp(shift_z)
    return exps / np.sum(exps, axis=0, keepdims=True)

# Input
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X_all = mnist.data.T / 255.0
Y_all_labels = mnist.target.astype(int)

X_train = X_all
y_train = Y_all_labels

X_test = X_all[:, 60000:]
y_test = Y_all_labels[60000:]

Np = X_train.shape[1]

# Number of neurons
Nneurons = 100

# Number of epochs
Nepochs = 500

# Learning rate
a = 0.1

# Initialize the weights randomly
w = np.random.randn(Nneurons, 784) * 0.01
v = np.random.randn(Nneurons, 10) * 0.1

# Initialize biases
b = np.zeros((Nneurons, 1))
b_out = np.zeros((10, 1))

# Train NN
X = X_train
y = np.zeros((10, Np))
y[y_train, np.arange(Np)] = 1.0

# Add momentum (Dynamic learning rate)
v_dw = np.zeros_like(w)
v_dv = np.zeros_like(v)
v_db = np.zeros_like(b)
v_db_out = np.zeros_like(b_out)

beta = 0.9
for e in range(Nepochs):
    
    # Find prediction
    Z = np.dot(w, X) + b
    A = ReLU(Z)
    y_pred = softmax(np.dot(v.T, A) + b_out)
    
    # Back propagation calculations
    err = y_pred - y
    err_hidden = np.dot(v, err) * ReLU_prime(Z)
    dv = np.dot(A, err.T) / Np
    dw = err_hidden.dot(X.T) / Np
    db = np.sum(err_hidden, axis=1, keepdims=True) / Np
    db_out = np.sum(err, axis=1, keepdims=True) / Np

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

    if e % 10 == 0:
        predictions = np.argmax(y_pred, axis=0)
        accuracy = np.mean(predictions == y_train) * 100
        loss = -np.sum(y * np.log(y_pred + 1e-15)) / Np
        print(f"Epoch {e:4d} | Loss: {loss:.4f} | Training Accuracy: {accuracy:.2f}%")

# Test model on remaining data
Z_test = np.dot(w, X_test) + b
A_test = ReLU(Z_test)
y_pred_test = softmax(np.dot(v.T, A_test) + b_out)

test_predictions = np.argmax(y_pred_test, axis=0)

test_accuracy = np.mean(test_predictions == y_test) * 100
print(f"Final Test Accuracy: {test_accuracy:.2f}%")

np.save('./digit_mlp/w.npy', w)
np.save('./digit_mlp/v.npy', v)
np.save('./digit_mlp/b.npy', b)
np.save('./digit_mlp/b_out.npy', b_out)
