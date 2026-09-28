# Physics-Informed Neural Network (PINN) for Plug Flow Reactor

A custom PINN built in PyTorch to model the concentration profile of a steady-state isothermal Plug Flow Reactor (PFR). This project compares a standard data-driven MLP against a physics-informed architecture constrained by the governing ordinary differential equation (ODE).

## The Physics Model
The physical system is a dimensionless, isothermal PFR governed by the following species balance:

$$\frac{1}{Da} \frac{dc_A}{dz} + c_A^n = 0$$

*   $c_A$: Dimensionless concentration
*   $z$: Dimensionless reactor length $[0, 1]$
*   $Da$: Damkohler number
*   $n$: Reaction order

## Analytical Ground Truth
To validate the model's accuracy, the analytical solution was derived. For a reaction order of $n > 1$, the exact concentration profile is[cite: 6]:

$$C_A(z) = \left[ 1 + Da(n - 1)z \right]^{\frac{1}{1 - n}}$$

*Note: The complete manual integration for ground truth solutions ($n=0, n=1, n>1$) is available in `docs/PINN.pdf`*[cite: 6].

## Results
The animation below demonstrates the training evolution. The standard MLP overfits to simulated sensor noise, while the PINN regularizes to the physical ODE and boundary conditions.

![PINN vs MLP Training](pinn_vs_mlp_training.gif)
