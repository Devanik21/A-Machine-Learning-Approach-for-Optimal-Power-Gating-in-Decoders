from sklearn.neural_network import MLPRegressor

def create_custom_mlp():
    """Instantiates an MLPRegressor with custom architecture optimized for VLSI metrics."""
    return MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        max_iter=1000,
        early_stopping=True,
        random_state=42
    )
