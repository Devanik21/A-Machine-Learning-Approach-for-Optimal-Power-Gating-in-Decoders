# Surrogate Modeling Methodology

## Rationale
Exhaustive SPICE simulations for VLSI decoders are computationally prohibitive, scaling exponentially with the number of design parameters (transistor widths, Vdd, Vth, etc.). We employ surrogate modeling to approximate the physical response surfaces (Power, Delay, Area).

## Algorithms Evaluated
*   **Random Forest Regressor**: Provides robust baseline and intrinsic feature importance via MDI.
*   **Gradient Boosting Regressor**: Captures non-linearities effectively, sequential error correction.
*   **Multilayer Perceptron (MLP)**: 3-hidden-layer feedforward network to model complex parameter interactions.
*   **Support Vector Regression (SVR)**: RBF kernel employed to model high-dimensional, non-linear relationships.

## Validation Strategy
We use 5-fold cross-validation coupled with a holdout test set (80/20 split). Primary metrics are R^2, RMSE, and MAE.
