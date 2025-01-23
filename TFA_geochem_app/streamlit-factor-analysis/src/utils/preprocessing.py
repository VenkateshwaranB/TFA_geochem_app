# File: /streamlit-factor-analysis/src/utils/preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

def scale_features(data):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return pd.DataFrame(scaled_data, columns=data.columns)

def perform_bartlett_test(data):
    """Perform Bartlett's test for sphericity."""
    chi_square_value, p_value = calculate_bartlett_sphericity(data)
    return chi_square_value, p_value

def perform_kmo_test(data):
    """Perform KMO test for sampling adequacy."""
    kmo_all, kmo_model = calculate_kmo(data)
    return kmo_model

# This file is intentionally left blank.