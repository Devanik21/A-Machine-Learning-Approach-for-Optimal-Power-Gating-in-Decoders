# Power Gating Decoder ML Optimizer

### A Machine Learning-Driven Design Space Exploration Tool for Low-Power VLSI Decoder Optimization

**Author:** Devanik Debnath  
**GitHub:** [@Devanik21](https://github.com/Devanik21)  
**LinkedIn:** [linkedin.com/in/devanik](https://www.linkedin.com/in/devanik/)  
**Domain:** VLSI CAD · Machine Learning · Low-Power Digital Design

---

## Overview

This application is a Streamlit-based interactive platform that applies supervised machine learning to the problem of **multi-objective optimization of power-gated line decoders**. Rather than relying on exhaustive SPICE simulation sweeps — which are computationally prohibitive across large design spaces — the tool trains regression surrogates on a physics-informed dataset to predict Power (mW), Delay (ns), and Area (µm²) given a set of design parameters. It then uses those surrogates to drive Pareto frontier identification and Bayesian optimization.

The application is a direct computational extension of the mixed-logic decoder design work (2-4HP and 4-16HP decoders, 32nm PTM LP), translating the physical intuition from circuit simulation into a learnable, data-driven optimization engine.

---

## Table of Contents

1. [Dataset & Feature Engineering](#1-dataset--feature-engineering)
2. [Machine Learning Pipeline](#2-machine-learning-pipeline)
3. [Model Architectures](#3-model-architectures)
4. [Explainability — SHAP Integration](#4-explainability--shap-integration)
5. [Multi-Objective Pareto Optimization](#5-multi-objective-pareto-optimization)
6. [Bayesian Optimization via scikit-optimize](#6-bayesian-optimization-via-scikit-optimize)
7. [Sensitivity Analysis](#7-sensitivity-analysis)
8. [Application Architecture](#8-application-architecture)
9. [Installation & Usage](#9-installation--usage)
10. [Dependencies](#10-dependencies)

---

## 1. Dataset & Feature Engineering

The application reads from a local CSV file: `decoder_power_delay_area_dataset.csv`.

### Required Columns

| Feature | Type | Physical Meaning |
|---------|------|-----------------|
| `decoder_size` | Integer (2–6) | Number of input bits $n$ in an $n$-to-$2^n$ decoder |
| `tech_node` | Categorical (nm) | Process node: {180, 130, 90, 65, 45, 32, 22} |
| `supply_voltage` | Continuous (V) | $V_{DD}$ operating voltage |
| `threshold_voltage` | Continuous (V) | $V_{th}$ of NMOS devices |
| `transistor_width` | Continuous (µm) | Gate width $W$ — primary sizing variable |
| `load_capacitance` | Continuous (fF) | Output node load $C_L$ |
| `pg_efficiency` | Continuous [0.5–0.95] | Power gating effectiveness $\eta$ |
| `switching_activity` | Continuous [0.1–0.8] | Activity factor $\alpha$ |
| `leakage_factor` | Continuous [0.01–0.1] | Subthreshold leakage scaling |
| `temperature` | Continuous (°C) | Operating junction temperature |
| `power` | Target (mW) | Total average power dissipation |
| `delay` | Target (ns) | Maximum propagation delay |
| `area` | Target (µm²) | Total transistor layout area |

### Physics Basis of the Feature Set

The features are grounded in standard CMOS power modeling:

**Dynamic Power:**
$$P_{dyn} = \alpha \cdot C_L \cdot V_{DD}^2 \cdot f$$

**Leakage Power:**
$$P_{leak} = V_{DD} \cdot I_{sub} = V_{DD} \cdot I_0 \cdot e^{\frac{V_{GS} - V_{th}}{nV_T}}$$

**Power-Gated Savings:**
$$P_{saved} = \eta \cdot P_{base}$$

where $\eta$ is the `pg_efficiency` feature, directly quantifying how effectively the sleep transistor isolates the decoder's logic cone during inactive periods. This parametric construction ensures the dataset captures physically meaningful variation rather than arbitrary random noise.

---

## 2. Machine Learning Pipeline

```
CSV Dataset
    │
    ▼
Validation (column check)
    │
    ▼
Configuration-based Filtering
(decoder_size, tech_node, supply_voltage ± 0.2V tolerance)
    │
    ▼
Train/Test Split
(80/20 for n ≥ 100 samples; 75/25 or 70/30 for smaller subsets)
    │
    ▼
StandardScaler (zero-mean, unit-variance normalization)
    │
    ▼
Parallel training on three targets: {power, delay, area}
    │
    ▼
Evaluation: R², RMSE, MAE per model per target
    │
    ▼
Surrogate-driven inference for:
  ├── Custom predictions (Tab 3)
  ├── Pareto optimization (Tab 4)
  ├── Sensitivity analysis (Tab 5)
  └── Bayesian optimization (Tab 6)
```

**Session state management:** All trained models, scalers, data splits, and prediction results are stored in `st.session_state`, ensuring models are only retrained when the configuration changes (tracked via a composite key of `decoder_bits`, `technology_node`, `supply_voltage`, `optimization_target`, and dataset size). The `@st.cache_resource` decorator is applied to the training function to prevent redundant recomputation across re-renders.

---

## 3. Model Architectures

Four regression algorithms are available, each with different inductive biases suited to different regions of the design space.

### 3.1 Random Forest Regressor

```
n_estimators = 100
max_depth    = 10
random_state = 42
n_jobs       = -1  (full parallelism)
```

An ensemble of 100 decision trees trained via bootstrap aggregation (bagging). Each tree is fit on a random subset of training samples with random feature subsets at each split. The final prediction is the arithmetic mean across all trees:

$$\hat{y} = \frac{1}{B} \sum_{b=1}^{100} T_b(\mathbf{x})$$

Random Forest provides native **feature importance** via mean decrease in impurity (MDI), used throughout Tab 2 to rank design parameters.

### 3.2 Gradient Boosting Regressor

```
n_estimators  = 100
max_depth     = 5
learning_rate = 0.1
random_state  = 42
```

A sequential additive ensemble where each tree $T_b$ corrects the residual error of its predecessor:

$$F_b(\mathbf{x}) = F_{b-1}(\mathbf{x}) + \nu \cdot T_b(\mathbf{x})$$

where $\nu = 0.1$ is the learning rate (shrinkage). Gradient Boosting minimizes a differentiable loss function (MSE for regression) in function space via gradient descent, making it highly accurate on structured tabular data. The shallower `max_depth=5` controls overfitting relative to the Random Forest.

### 3.3 Neural Network (MLPRegressor)

```
hidden_layer_sizes = (64, 32, 16)
max_iter           = 500
early_stopping     = True
random_state       = 42
```

A three-hidden-layer feedforward network with ReLU activations and Adam optimizer. The architecture progressively compresses the 10-dimensional input:

$$10 \xrightarrow{W_1 \in \mathbb{R}^{64 \times 10}} 64 \xrightarrow{W_2 \in \mathbb{R}^{32 \times 64}} 32 \xrightarrow{W_3 \in \mathbb{R}^{16 \times 32}} 16 \xrightarrow{W_{out} \in \mathbb{R}^{1 \times 16}} \hat{y}$$

Early stopping monitors a held-out validation loss to prevent overfitting. This model is best suited to capturing nonlinear interactions between features (e.g., the joint effect of `supply_voltage` and `threshold_voltage` on leakage).

### 3.4 Support Vector Regression (SVR)

```
kernel = 'rbf'
C      = 100
gamma  = 'scale'
```

SVR fits a hyperplane in the kernel-induced feature space such that most training points lie within an $\epsilon$-tube around the prediction. The RBF kernel computes:

$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right)$$

where `gamma='scale'` sets $\gamma = \frac{1}{n_{features} \cdot \text{Var}(X)}$. A high regularization parameter $C=100$ allows tight fit at the cost of wider margin, appropriate for this relatively low-noise physics-based dataset.

### Evaluation Metrics

All models are evaluated on the held-out test set using three metrics:

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| $R^2$ (coefficient of determination) | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | Fraction of variance explained; 1.0 is perfect |
| RMSE | $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ | Error in original units (mW, ns, µm²) |
| MAE | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Robust mean absolute deviation |

---

## 4. Explainability — SHAP Integration

The application uses **SHAP (SHapley Additive Explanations)** to explain individual model predictions, sourced from cooperative game theory.

For a prediction $\hat{y} = f(\mathbf{x})$, the SHAP value $\phi_j$ for feature $j$ is defined as:

$$\phi_j = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!\,(|F|-|S|-1)!}{|F|!} \left[ f(S \cup \{j\}) - f(S) \right]$$

where $F$ is the full feature set and the sum iterates over all subsets $S$ not containing feature $j$.

**Implementation branch:**
- For `RandomForestRegressor` and `GradientBoostingRegressor`: `shap.TreeExplainer` is used, which exploits the tree structure for exact, computationally efficient SHAP values in $O(TLD^2)$ time (T = trees, L = leaves, D = depth).
- For `MLPRegressor` and `SVR`: `shap.KernelExplainer` is used with a k-means summary (50 clusters) of the training set as background, computing model-agnostic approximate SHAP values via weighted linear regression on perturbations.

Force plots rendered via `matplotlib=True` show how each feature pushes the prediction above or below the expected model output, providing direct interpretability for design decisions.

---

## 5. Multi-Objective Pareto Optimization

Tab 4 implements a **Pareto frontier identification** over the three-dimensional objective space {Power, Delay, Area}.

### Dominance Definition

A solution $\mathbf{u}$ dominates solution $\mathbf{v}$ if:

$$u_k \leq v_k \quad \forall\, k \in \{\text{power, delay, area}\}$$
$$\exists\, k : u_k < v_k$$

The Pareto front $\mathcal{P}$ is the set of all non-dominated solutions.

### Implementation

The application samples `n_opt_samples` random configurations (user-configurable: 100–1000) from the design space, predicts their objectives using the trained surrogates, then runs a pairwise dominance check:

```python
for i in range(n):
    for j in range(n):
        if j dominates i:
            pareto_mask[i] = False
            break
```

This is an $O(n^2 \cdot d)$ algorithm (d = 3 objectives), appropriate for up to ~1000 samples in an interactive setting.

### Composite Scoring

Non-dominated solutions are ranked by a normalized composite score:

$$S = \hat{P}_{norm} + \hat{D}_{norm} + \hat{A}_{norm}$$

where each normalized objective is:

$$X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}} \in [0, 1]$$

The top-5 Pareto-optimal configurations by composite score are presented, with the best-scored configuration recommended as the operating point. A 3D interactive scatter plot (Plotly `Scatter3d`) renders the full Pareto front against the dominated design cloud.

---

## 6. Bayesian Optimization via scikit-optimize

Tab 6 uses `scikit-optimize` (`skopt`) to perform **model-based sequential optimization** over the decoder design space. This is fundamentally more sample-efficient than random search or grid search.

The search uses `gp_minimize` — Gaussian Process-based Bayesian optimization — which models the objective function as a GP prior and uses an acquisition function (Expected Improvement by default) to select the next query point:

$$\text{EI}(\mathbf{x}) = \mathbb{E}\left[\max(0,\, y^* - f(\mathbf{x}))\right]$$

where $y^*$ is the current best observed value. The GP posterior is updated after each evaluation, balancing **exploration** (high uncertainty regions) and **exploitation** (regions predicted to have low objective values).

The search space is defined using `skopt.space` primitives:

```python
space = [
    Integer(2, 6, name='decoder_size'),
    Categorical([180, 130, 90, 65, 45, 32, 22], name='tech_node'),
    Real(0.6, 1.8, name='supply_voltage'),
    Real(0.2, 0.5, name='threshold_voltage'),
    Real(0.5, 10.0, name='transistor_width'),
    Real(10.0, 200.0, name='load_capacitance'),
    Real(0.5, 0.95, name='pg_efficiency'),
    Real(0.1, 0.8, name='switching_activity'),
    Real(0.01, 0.1, name='leakage_factor'),
    Real(25.0, 85.0, name='temperature')
]
```

The objective passed to `gp_minimize` queries the trained ML surrogate rather than a real simulation, making each evaluation microseconds instead of minutes. This closed-loop architecture — **ML surrogate as objective function within a Bayesian optimizer** — is the core methodological contribution of the application.

---

## 7. Sensitivity Analysis

Tab 5 performs one-at-a-time (OAT) sensitivity analysis: one feature is varied across its full observed range while all others are held fixed at user-specified values. The trained surrogate predicts the target metric (power, delay, or area) across 100 evenly spaced values of the varied parameter (or across discrete categories if applicable).

This surfaces the **marginal response** of each design objective to individual parameters, directly informing designers about:
- Supply voltage scaling trade-offs ($V_{DD}$ vs. $P_{dyn} \propto V_{DD}^2$)
- The diminishing returns of increasing `pg_efficiency` near $\eta \to 1$
- The nonlinear delay sensitivity to `transistor_width` through $R_{on} \propto 1/W$

---

## 8. Application Architecture

```
app.py
├── Page configuration & CSS injection
├── Sidebar (global configuration panel)
│   ├── Algorithm selection (multiselect)
│   ├── Optimization target (selectbox)
│   └── Decoder specs (decoder_bits, tech_node, supply_voltage)
│
├── load_decoder_data()     [@st.cache_data]
│   └── Reads decoder_power_delay_area_dataset.csv
│
├── validate_dataset()
│   └── Enforces required column schema
│
├── train_ml_models()       [@st.cache_resource]
│   └── Fits selected models on scaled training data
│
└── Tab layout (7 tabs)
    ├── Tab 1 — Data Overview
    │   └── Distributions, correlation heatmap, statistics
    │
    ├── Tab 2 — ML Training
    │   ├── Filtered dataset preparation
    │   ├── Train/test split
    │   ├── Model training (all three targets)
    │   ├── Performance metrics table (R², RMSE, MAE)
    │   ├── Feature importance (Random Forest)
    │   └── Actual vs. Predicted scatter
    │
    ├── Tab 3 — Predictions & Explainability
    │   ├── Interactive parameter sliders
    │   ├── Per-algorithm predictions + ensemble average
    │   └── SHAP force plot (TreeExplainer / KernelExplainer)
    │
    ├── Tab 4 — Pareto Optimization
    │   ├── Random design space sampling
    │   ├── Surrogate-predicted objectives
    │   ├── Pareto dominance filtering
    │   ├── 3D Pareto front visualization (Plotly)
    │   └── Top-5 configurations + recommended optimum
    │
    ├── Tab 5 — Sensitivity Analysis
    │   └── OAT analysis with configurable fixed parameters
    │
    ├── Tab 6 — Bayesian Optimization
    │   └── gp_minimize over ML surrogate
    │
    └── Tab 7 — Report
        ├── Methodology summary
        ├── Downloadable CSV export
        └── Downloadable TXT report
```

**State management:** The application uses `st.session_state` to persist trained models (`all_trained_models`), data splits (`all_data_splits`), and prediction results across tab interactions. Configuration change detection is handled via a string key (`last_config`) that triggers retraining when any sidebar parameter changes.

---

## 9. Installation & Usage

### Prerequisites

Python 3.9 or higher is recommended.

### Setup

```bash
git clone https://github.com/Devanik21/<repo-name>.git
cd <repo-name>

pip install -r requirements.txt
```

### Required File

Place the dataset file in the same directory as `app.py`:

```
project/
├── app.py
├── decoder_power_delay_area_dataset.csv
└── requirements.txt
```

### Run

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

### Workflow

1. Select algorithms and decoder configuration in the sidebar.
2. Navigate to the **ML Training** tab and click **Train Models**.
3. Use **Predictions & Explainability** to query the trained surrogates for a specific design point and inspect SHAP attribution.
4. Use **Pareto Optimization** to explore the trade-off surface across a sampled design space.
5. Use **Bayesian Optimization** for a directed, sample-efficient search for the minimum of a single target.
6. Use **Sensitivity Analysis** to examine marginal parameter effects.
7. Export results from the **Report** tab.

---

## 10. Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web application framework |
| `scikit-learn` | ML models (RF, GB, MLP, SVR), preprocessing, metrics |
| `shap` | SHAP-based model explainability |
| `scikit-optimize` | Bayesian optimization (`gp_minimize`) |
| `plotly` | Interactive 2D/3D visualizations |
| `matplotlib` | SHAP force plot rendering |
| `pandas` | Data loading, manipulation, export |
| `numpy` | Numerical operations, array handling |

Install all dependencies with:

```bash
pip install streamlit scikit-learn shap scikit-optimize plotly matplotlib pandas numpy
```

---

## Author

**Devanik Debnath**  
B.Tech, Electronics & Communication Engineering  
National Institute of Technology Agartala

All machine learning design, surrogate model architecture, optimization pipeline, Streamlit interface, and application code in this repository were independently developed by the author.

| Platform | Link |
|----------|------|
| GitHub | [@Devanik21](https://github.com/Devanik21) |
| LinkedIn | [linkedin.com/in/devanik](https://www.linkedin.com/in/devanik/) |

---

> This application is part of a broader research effort on low-power mixed-logic decoder design. The companion hardware simulation work (LTspice, 32nm PTM LP, 2-4HP and 4-16HP decoders) is documented in a separate repository.

> © 2026 Devanik Debnath. All application code and design are original contributions by the author.
