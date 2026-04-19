from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def get_ensemble_models():
    """Returns a dictionary of uninitialized ensemble regressors."""
    return {
        'rf': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'gb': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    }
