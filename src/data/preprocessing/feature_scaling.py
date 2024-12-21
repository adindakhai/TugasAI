import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Min-Max Scaling Function
def apply_minmax_scaling(X, feature_range=(0, 1)):
    scaler = MinMaxScaler(feature_range=feature_range)
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

# Standardization Function
def apply_standard_scaling(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

# Function to select and apply scaling method
def scale_features(X, method='minmax', feature_range=(0, 1)):
    """
    Scale features using the specified scaling method.

    Parameters:
        X (array-like): Input features.
        method (str): Scaling method ('minmax' or 'standard').
        feature_range (tuple): Range for Min-Max scaling.

    Returns:
        X_scaled (array-like): Scaled features.
        scaler (object): Fitted scaler object.
    """
    if method == 'minmax':
        return apply_minmax_scaling(X, feature_range)
    elif method == 'standard':
        return apply_standard_scaling(X)
    else:
        raise ValueError("Unsupported scaling method. Choose 'minmax' or 'standard'.")
