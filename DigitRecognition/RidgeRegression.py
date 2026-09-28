# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 19:34:04 2026

@author: Diogenis
"""

# %% Ridge regression weight equation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing


def ridge(X, Y, alpha):
    N = X.shape[1]
    penalty = alpha * np.identity(N)
    penalty[0, 0] = 0
    w = np.dot(np.linalg.inv(np.dot(X.T, X) + penalty), np.dot(X.T, Y))
    return w

# %% Applying ridge regression on California housing dataset
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)

print(df)

X = df.to_numpy()
Y = housing.target

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0) # Input scaling
X = np.hstack((np.ones((X.shape[0], 1)), X)) # Add bias vector
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

alphas = np.logspace(-3, 3, 100)
w = np.zeros((alphas.shape[0], X.shape[1])).T
Y_pred = np.zeros((X_test.shape[0], alphas.shape[0]))

# Apply ridge for multiple alphas
for i in range(alphas.shape[0]):
    w[:, i] = ridge(X_train, Y_train, alphas[i])
    Y_pred[:, i] = np.dot(X_test, w[:, i])

# Find optimal alpha
mse = np.zeros(alphas.shape[0])
for i in range(alphas.shape[0]):
    mse[i] = np.mean((Y_test - Y_pred[:, i]) ** 2)

index_opt = np.argmin(mse)
alpha_opt = alphas[index_opt]

# Do final regression using optimal alpha
w_opt = ridge(X_train, Y_train, alpha_opt)
Y_pred_opt = np.dot(X_test, w_opt)

# %% Plotting

# MSE vs alpha plot
plt.figure()
plt.scatter(alphas, mse)
plt.xscale('log')
plt.show()

# Residuals plot
plt.figure()
plt.title('Residuals - Y_pred')
plt.scatter(Y_pred_opt, Y_test - Y_pred_opt, label=f'a = {alpha_opt}')
plt.legend()
plt.show()

# Predictions vs true values plot
plt.figure()
plt.title('Y_pred - Y_test')
plt.scatter(Y_test, Y_pred_opt, label=f'a = {alpha_opt}')
plt.plot(Y_test, Y_test, color='black', label='y = x')
plt.legend()
plt.show()
