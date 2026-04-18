from sklearn.preprocessing import PolynomialFeatures

def apply_polynomial_features(X, degree=2):
    """Applies polynomial feature expansion to capture interaction terms."""
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    return poly.fit_transform(X)
