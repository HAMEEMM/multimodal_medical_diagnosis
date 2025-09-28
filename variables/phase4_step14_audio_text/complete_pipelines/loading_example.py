
# Example: How to load and use the saved preprocessing models


# 1. Load the deployment pipeline
deployment_pipeline = joblib.load('complete_pipelines/deployment_pipeline.joblib')

# 2. Load individual components if needed
feature_selector = joblib.load('feature_selectors/best_feature_selector.joblib')
scaler = joblib.load('scalers/best_scaler.joblib')
normalizer = joblib.load('normalizers/best_normalizer.joblib')
label_encoder = joblib.load('encoders/label_encoder.joblib')

# 3. Transform new data
def preprocess_new_data(new_integrated_features):
    """
    Preprocess new integrated features using saved models
    """
    # Apply feature selection
    if deployment_pipeline['feature_selector'] is not None:
        selected_features = deployment_pipeline['feature_selector'].transform(new_integrated_features)
    else:
        selected_features = new_integrated_features

    # Apply scaling
    if deployment_pipeline['data_scaler'] is not None:
        scaled_features = deployment_pipeline['data_scaler'].transform(selected_features)
    else:
        scaled_features = selected_features

    # Apply normalization
    if deployment_pipeline['normalizer'] is not None:
        normalized_features = deployment_pipeline['normalizer'].transform(scaled_features)
    else:
        normalized_features = scaled_features

    return normalized_features

# 4. Example usage
# new_data = your_integrated_features  # Shape: (n_samples, n_original_features)
# processed_data = preprocess_new_data(new_data)
# predictions = your_trained_model.predict(processed_data)
