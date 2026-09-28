# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:18:42 2026

@author: Diogenis
"""

import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw


# Neural network
def ReLU(z):
    return np.maximum(0, z)

def softmax(z):
    shift_z = z - np.max(z, axis=0, keepdims=True)
    exps = np.exp(shift_z)
    return exps / np.sum(exps, axis=0, keepdims=True)

w1 = np.load('./digit_mlp/w1.npy')
v = np.load('./digit_mlp/v.npy')
b1 = np.load('./digit_mlp/b1.npy')
w2 = np.load('./digit_mlp/w2.npy')
b2 = np.load('./digit_mlp/b2.npy')
b_out = np.load('./digit_mlp/b_out.npy')

# Input matrix
X = np.zeros((28, 28))

# Create window
root = tk.Tk()
root.title('Digit predictor')

# PIL image buffer
high_res_img = Image.new("L", (280, 280), 0)
draw = ImageDraw.Draw(high_res_img)

# Sketch on the canvas
def savePosn(event):
    global lastx, lasty
    lastx, lasty = event.x, event.y

def addLine(event):
    canvas.create_line(lastx, lasty, event.x, event.y, width=10, capstyle=tk.ROUND)
    draw.line([lastx, lasty, event.x, event.y], fill=255, width=10, joint="round")
    
    savePosn(event)
    updatePrediction()

def updatePrediction():
    img_28x28 = high_res_img.resize((28, 28), Image.Resampling.BILINEAR)
    X = np.array(img_28x28) / 255.0
    
    if not np.any(X):
        predictionLabel.config(text="Prediction: None, Confidence: 0.00%")
        return
    
    # Calculate weighted Center of Mass over the absolute matrix space
    total_mass = np.sum(X)
    if total_mass > 0:
        Y_indices, X_indices = np.indices((28, 28))
        r_com = int(np.sum(Y_indices * X) / total_mass)
        c_com = int(np.sum(X_indices * X) / total_mass)
        
        # Determine pixel translation offsets to line up with center (14, 14)
        shift_y = 14 - r_com
        shift_x = 14 - c_com
        
        # Seamlessly translate the array using roll
        X_centered = np.roll(X, shift=shift_y, axis=0)
        X_centered = np.roll(X_centered, shift=shift_x, axis=1)
    else:
        X_centered = np.zeros((28, 28))
    
    # Find prediction using model
    X_flat = X_centered.reshape((784, 1))
    Z1 = np.dot(w1, X_flat) + b1
    A1 = ReLU(Z1)
    Z2 = np.dot(w2, A1) + b2
    A2 = ReLU(Z2)
    y_pred = softmax(np.dot(v.T, A2) + b_out)

    digit = np.argmax(y_pred)
    prob = np.max(y_pred) * 100
    
    predictionLabel.config(text=f'Prediction: {digit}, Confidence: {prob:.2f}%')

def clearCanvas():
    global high_res_img, draw
    canvas.delete('all')
    X.fill(0.0)
    high_res_img = Image.new("L", (280, 280), 0)
    draw = ImageDraw.Draw(high_res_img)
    predictionLabel.config(text="Prediction: None, Confidence: 0.00%")

# Window widgets
canvas = tk.Canvas(root, height=280, width=280)
canvas.pack()
canvas.bind("<Button-1>", savePosn)
canvas.bind("<B1-Motion>", addLine)

predictionLabel = tk.Label(root, text='Start drawing')
predictionLabel.pack()

clearButton = tk.Button(root, text='Clear canvas', command=clearCanvas)
clearButton.pack()

root.mainloop()
