# Bayesian Optimization Formulation

## Objective Function
The objective function is derived from the trained surrogate models rather than physical SPICE simulations. Let $f_p(x)$, $f_d(x)$, and $f_a(x)$ be the surrogate predictions for power, delay, and area respectively, given design vector $x$.

The multi-objective formulation is often scalarized using a weighted sum for Bayesian Optimization:
$$ g(x) = w_1 \cdot 	ext{norm}(f_p(x)) + w_2 \cdot 	ext{norm}(f_d(x)) + w_3 \cdot 	ext{norm}(f_a(x)) $$

## Acquisition Function
We utilize Expected Improvement (EI):
$$ 	ext{EI}(x) = \mathbb{E}[\max(0, g(x^*) - g(x))] $$
where $g(x^*)$ is the best observed value so far. The Gaussian Process prior is updated iteratively.
