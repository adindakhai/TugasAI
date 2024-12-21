from src.data.preprocessing.feature_scaling import scale_features
import pandas as pd

# Example: Load dataset
df = pd.read_csv('/train.csv')

# Separate features and target
X = df.drop('target_column', axis=1)  # Replace 'target_column' with the actual name
y = df['target_column']

# Apply Min-Max Scaling
X_scaled, scaler = scale_features(X, method='minmax', feature_range=(0, 1))

# Alternatively, Apply Standard Scaling
# X_scaled, scaler = scale_features(X, method='standard')
