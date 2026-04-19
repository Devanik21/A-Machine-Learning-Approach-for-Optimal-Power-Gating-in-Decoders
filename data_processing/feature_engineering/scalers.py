from sklearn.preprocessing import StandardScaler, MinMaxScaler

def get_scaler(method='standard'):
    """Returns the requested scikit-learn scaler."""
    if method == 'standard':
        return StandardScaler()
    elif method == 'minmax':
        return MinMaxScaler()
    else:
        raise ValueError("Unsupported scaling method")
