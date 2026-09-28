# Custom Multilayer Perceptron (MLP) for Digit Recognition

A NumPy-based Multilayer Perceptron built entirely from scratch to classify handwritten digits. The forward pass and backpropagation algorithms were derived and implemented manually without automatic differentiation libraries.

## Mathematical Foundation
The network utilizes a cross-entropy loss function, defined in matrix form as:

$$\mathcal{L} = -\frac{1}{N_p} \text{tr}(y^T \log \hat{y})$$

Backpropagation gradients were derived using matrix differential calculus and the trace operator. For example, the gradient of the loss with respect to the first hidden layer weights ($w_1$) was analytically evaluated as:

$$\nabla_{w_1} \mathcal{L} = \frac{1}{N_p} (w_2^T \mathcal{E}_{h2} \odot \sigma'_{ReLU}) X^T$$

*Note: The complete step-by-step matrix calculus derivations for all weight and bias gradients, including $\nabla_{b_1} \mathcal{L} = \frac{1}{N_p} \mathcal{E}_{h1}$, can be found in `docs/Digit recognition (2 hidden layers, MLP).pdf`*.
