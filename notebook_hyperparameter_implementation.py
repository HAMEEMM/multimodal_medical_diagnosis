# Cell: Notebook Implementation - Expanded Hyperparameter Grids
# Copy this code directly into a new cell in your notebook

# Import required libraries
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import time
import warnings
warnings.filterwarnings('ignore')

# Expanded hyperparameter grids with broader parameter ranges
param_grids_expanded = {
    'RandomForest': {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [3, 5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy'],
        'class_weight': [None, 'balanced']
    },
    
    'SVM': {
        'C': [0.01, 0.1, 1, 10, 100, 1000],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10],
        'degree': [2, 3, 4, 5],  # for polynomial kernel
        'coef0': [0.0, 0.1, 0.5, 1.0],  # for poly and sigmoid kernels
        'probability': [True],  # enable probability estimates
        'class_weight': [None, 'balanced']
    },
    
    'LogisticRegression': {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'solver': ['liblinear', 'saga', 'lbfgs'],
        'max_iter': [1000, 2000, 3000],
        'class_weight': [None, 'balanced'],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]  # for elasticnet
    },
    
    'GradientBoosting': {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0],
        'max_features': ['sqrt', 'log2', None]
    }
}

# Function to perform enhanced grid search
def enhanced_grid_search(models_dict, X_train, y_train, use_randomized=True, n_iter=100):
    """
    Perform enhanced grid search with expanded parameter ranges
    """
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    best_models = {}
    best_params = {}
    cv_scores = {}
    
    print("Starting Enhanced Hyperparameter Tuning with Expanded Grids")
    print("=" * 60)
    
    for model_name, model in models_dict.items():
        print(f"\nOptimizing {model_name}...")
        start_time = time.time()
        
        param_grid = param_grids_expanded.get(model_name, {})
        
        if not param_grid:
            print(f"No expanded grid for {model_name}, using default parameters")
            continue
        
        try:
            if use_randomized and model_name in ['RandomForest', 'SVM', 'GradientBoosting']:
                # Use RandomizedSearchCV for complex models
                grid_search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grid,
                    n_iter=n_iter,
                    cv=cv_strategy,
                    scoring='accuracy',
                    n_jobs=-1,
                    random_state=42,
                    verbose=1
                )
            else:
                # Use GridSearchCV for simpler models
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=cv_strategy,
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=1
                )
            
            grid_search.fit(X_train, y_train)
            
            best_models[model_name] = grid_search.best_estimator_
            best_params[model_name] = grid_search.best_params_
            cv_scores[model_name] = grid_search.best_score_
            
            elapsed_time = time.time() - start_time
            
            print(f"✓ {model_name} completed in {elapsed_time:.2f} seconds")
            print(f"  Best CV Score: {grid_search.best_score_:.4f}")
            print(f"  Best Parameters: {grid_search.best_params_}")
            
        except Exception as e:
            print(f"✗ Error optimizing {model_name}: {str(e)}")
            best_models[model_name] = model
            best_params[model_name] = "Default (optimization failed)"
            cv_scores[model_name] = 0.0
    
    return best_models, best_params, cv_scores

# Example implementation in your notebook:
"""
# Define your models
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'LogisticRegression': LogisticRegression(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42)
}

# Perform enhanced grid search with expanded parameter ranges
optimized_models, best_parameters, cv_scores = enhanced_grid_search(
    models, X_train_scaled, y_train, use_randomized=True, n_iter=150
)

# Evaluate optimized models on test set
print("\nEvaluating Optimized Models on Test Set:")
print("=" * 50)

optimized_results = {}
for model_name, model in optimized_models.items():
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    optimized_results[model_name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'cv_score': cv_scores[model_name]
    }
    
    print(f"\n{model_name}:")
    print(f"  Test Accuracy: {accuracy:.4f}")
    print(f"  Test Precision: {precision:.4f}")
    print(f"  Test Recall: {recall:.4f}")
    print(f"  Test F1-Score: {f1:.4f}")
    print(f"  CV Score: {cv_scores[model_name]:.4f}")

# Create comparison DataFrame
optimized_df = pd.DataFrame(optimized_results).T
print("\nOptimized Model Performance Summary:")
print(optimized_df.round(4))

# Find best performing model
best_model_name = optimized_df['accuracy'].idxmax()
print(f"\nBest Performing Model: {best_model_name}")
print(f"Best Parameters: {best_parameters[best_model_name]}")
"""

print("Enhanced hyperparameter tuning implementation ready!")
print("Key improvements:")
print("- Expanded Random Forest grid: n_estimators, max_depth, min_samples_split/leaf, max_features, bootstrap, criterion")
print("- Comprehensive SVM grid: C, kernel types, gamma, degree, coef0, class_weight")
print("- RandomizedSearchCV for efficient large parameter space exploration")
print("- Stratified cross-validation for better performance estimation")
print("- Error handling and fallback to default parameters")