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
Nneurons1 = 100
Nneurons2 = 50

# Number of epochs
Nepochs = 500

# Learning rate
a = 0.1

# Initialize the weights randomly
w1 = np.random.randn(Nneurons1, 784) * np.sqrt(2.0 / 784)
w2 = np.random.randn(Nneurons2, Nneurons1) * np.sqrt(2.0 / Nneurons1)
v = np.random.randn(Nneurons2, 10) * 0.1

# Initialize biases
b1 = np.zeros((Nneurons1, 1))
b2 = np.zeros((Nneurons2, 1))
b_out = np.zeros((10, 1))

# Train NN
X = X_train
y = np.zeros((10, Np))
y[y_train, np.arange(Np)] = 1.0

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
    y_pred = softmax(np.dot(v.T, A2) + b_out)
    
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

    if e % 10 == 0:
        predictions = np.argmax(y_pred, axis=0)
        accuracy = np.mean(predictions == y_train) * 100
        loss = -np.sum(y * np.log(y_pred + 1e-15)) / Np
        print(f"Epoch {e:4d} | Loss: {loss:.4f} | Training Accuracy: {accuracy:.2f}%")

# Test model on remaining data
Z1_test = np.dot(w1, X_test) + b1
A1_test = ReLU(Z1_test)
Z2_test = np.dot(w2, A1_test) + b2
A2_test = ReLU(Z2_test)
y_pred_test = softmax(np.dot(v.T, A2_test) + b_out)

test_predictions = np.argmax(y_pred_test, axis=0)

test_accuracy = np.mean(test_predictions == y_test) * 100
print(f"Final Test Accuracy: {test_accuracy:.2f}%")

np.save('./digit_mlp/w1.npy', w1)
np.save('./digit_mlp/b1.npy', b1)
np.save('./digit_mlp/w2.npy', w2)
np.save('./digit_mlp/b2.npy', b2)
np.save('./digit_mlp/v.npy', v)
np.save('./digit_mlp/b_out.npy', b_out)
