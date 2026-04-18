# System Architecture Design

## Overview
The VLSI Decoder ML Optimizer is architected to decouple the physics-based SPICE simulation data generation from the surrogate model training and inference pipelines.

## Components
1. **Data Ingestion Layer**: Handles parsing of CSV/HDF5 datasets exported from HSPICE/LTspice.
2. **Feature Engineering Store**: Persists scaling parameters and normalization constants to ensure consistency between training and inference.
3. **Model Registry**: Stores serialized versions (e.g., joblib/ONNX) of trained Random Forest, Gradient Boosting, MLP, and SVR regressors.
4. **Optimization Engine**: Integrates `scikit-optimize` for Bayesian Optimization using the surrogate models as the objective function.
5. **Presentation Layer**: A Streamlit-based web interface for interactive design space exploration.
