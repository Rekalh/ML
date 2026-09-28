# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 03:16:38 2026

@author: Diogenis
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml


# ReLU activator function
def ReLU(z):
    return np.maximum(0, z)

def softmax(z):
    shift_z = z - np.max(z, axis=0, keepdims=True)
    exps = np.exp(shift_z)
    return exps / np.sum(exps, axis=0, keepdims=True)

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X_all = mnist.data.T / 255.0

w = np.load('./digit_mlp/w.npy')
v = np.load('./digit_mlp/v.npy')
b = np.load('./digit_mlp/b.npy')
b_out = np.load('./digit_mlp/b_out.npy')

i = random.randint(0, X_all.shape[1] - 1)
random_X = X_all[:, i : i + 1]
random_digit = random_X.reshape((28, 28))

# Predict using the model
Z = np.dot(w, random_X) + b
A = ReLU(Z)
y_pred = softmax(np.dot(v.T, A) + b_out)

plt.figure()
plt.title(f'Prediction: {np.argmax(y_pred)}, Confidence: {np.max(y_pred) * 100:.2f}%')
plt.imshow(random_digit)
plt.show()
