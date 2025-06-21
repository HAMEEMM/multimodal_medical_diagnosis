# Expanded Hyperparameter Grids for Comprehensive Model Optimization
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Expanded hyperparameter grids for comprehensive optimization
expanded_param_grids = {
    'RandomForest': {
        'n_estimators': [50, 100, 200, 300, 500, 800],
        'max_depth': [3, 5, 10, 15, 20, 25, None],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 6, 8, 10],
        'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7, 0.9],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy'],
        'class_weight': [None, 'balanced', 'balanced_subsample']
    },
    
    'SVM': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        'degree': [2, 3, 4, 5, 6],  # For polynomial kernel
        'coef0': [0.0, 0.1, 0.5, 1.0, 2.0],  # For poly and sigmoid kernels
        'shrinking': [True, False],
        'probability': [True],  # Enable probability estimates for ROC curves
        'class_weight': [None, 'balanced']
    },
    
    'GradientBoosting': {
        'n_estimators': [50, 100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        'max_depth': [3, 4, 5, 6, 7, 8, 10, 12],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 6, 8],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7],
        'validation_fraction': [0.1, 0.2, 0.3],
        'n_iter_no_change': [5, 10, 15]
    },
    
    'LogisticRegression': {
        'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'penalty': ['l1', 'l2', 'elasticnet', None],
        'solver': ['liblinear', 'saga', 'lbfgs', 'newton-cg', 'sag'],
        'max_iter': [100, 500, 1000, 2000, 3000, 5000],
        'class_weight': [None, 'balanced'],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],  # For elasticnet penalty
        'fit_intercept': [True, False]
    },
    
    'DecisionTree': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 5, 7, 10, 15, 20, 25, None],
        'min_samples_split': [2, 5, 10, 15, 20, 25],
        'min_samples_leaf': [1, 2, 4, 6, 8, 10, 12],
        'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7, 1.0],
        'splitter': ['best', 'random'],
        'class_weight': [None, 'balanced'],
        'max_leaf_nodes': [None, 10, 20, 50, 100, 200]
    },
    
    'KNeighbors': {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 25, 30],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [10, 20, 30, 40, 50, 60],
        'p': [1, 2, 3, 4],  # 1 for manhattan, 2 for euclidean
        'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
    },
    
    'GaussianNB': {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    }
}

def create_parameter_compatibility_svm(base_params):
    """
    Create SVM parameter combinations that are compatible with different kernels
    to avoid invalid parameter combinations during grid search.
    """
    compatible_params = []
    
    for kernel in base_params['kernel']:
        for C in base_params['C']:
            for shrinking in base_params['shrinking']:
                for probability in base_params['probability']:
                    for class_weight in base_params['class_weight']:
                        param_set = {
                            'C': C,
                            'kernel': kernel,
                            'shrinking': shrinking,
                            'probability': probability,
                            'class_weight': class_weight
                        }
                        
                        # Add gamma for non-linear kernels
                        if kernel in ['rbf', 'poly', 'sigmoid']:
                            for gamma in base_params['gamma']:
                                gamma_param_set = param_set.copy()
                                gamma_param_set['gamma'] = gamma
                                
                                # Add degree and coef0 for polynomial kernel
                                if kernel == 'poly':
                                    for degree in base_params['degree']:
                                        for coef0 in base_params['coef0']:
                                            poly_param_set = gamma_param_set.copy()
                                            poly_param_set['degree'] = degree
                                            poly_param_set['coef0'] = coef0
                                            compatible_params.append(poly_param_set)
                                # Add coef0 for sigmoid kernel
                                elif kernel == 'sigmoid':
                                    for coef0 in base_params['coef0']:
                                        sigmoid_param_set = gamma_param_set.copy()
                                        sigmoid_param_set['coef0'] = coef0
                                        compatible_params.append(sigmoid_param_set)
                                else:
                                    compatible_params.append(gamma_param_set)
                        else:
                            # Linear kernel doesn't need gamma, degree, or coef0
                            compatible_params.append(param_set)
    
    return compatible_params

def perform_comprehensive_hyperparameter_tuning(X_train, y_train, X_test, y_test, 
                                               models_dict, param_grids, 
                                               use_randomized_search=True, 
                                               n_iter=200, cv_folds=5, 
                                               scoring='accuracy', n_jobs=-1):
    """
    Perform comprehensive hyperparameter tuning with expanded parameter grids.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        models_dict: Dictionary of model instances
        param_grids: Dictionary of parameter grids
        use_randomized_search: Whether to use RandomizedSearchCV for large param spaces
        n_iter: Number of iterations for RandomizedSearchCV
        cv_folds: Number of cross-validation folds
        scoring: Scoring metric
        n_jobs: Number of parallel jobs
    
    Returns:
        Dictionary containing best models, parameters, and performance metrics
    """
    
    # Use stratified k-fold for better cross-validation
    cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    results = {
        'best_models': {},
        'best_params': {},
        'cv_scores': {},
        'test_scores': {},
        'training_times': {},
        'grid_search_objects': {}
    }
    
    print("Starting Comprehensive Hyperparameter Tuning")
    print("=" * 60)
    
    for model_name, model in models_dict.items():
        print(f"\nOptimizing {model_name}...")
        print("-" * 40)
        
        start_time = time.time()
        
        # Get parameter grid for current model
        param_grid = param_grids.get(model_name, {})
        
        if not param_grid:
            print(f"No parameter grid defined for {model_name}, using default parameters")
            model.fit(X_train, y_train)
            results['best_models'][model_name] = model
            results['best_params'][model_name] = "Default parameters"
            continue
        
        # Special handling for SVM to ensure parameter compatibility
        if model_name == 'SVM':
            print("Creating compatible SVM parameter combinations...")
            compatible_svm_params = create_parameter_compatibility_svm(param_grid)
            # Limit to reasonable number of combinations
            if len(compatible_svm_params) > 500:
                compatible_svm_params = compatible_svm_params[:500]
            param_grid = compatible_svm_params
        
        try:
            # Determine whether to use GridSearchCV or RandomizedSearchCV
            if use_randomized_search and model_name in ['RandomForest', 'GradientBoosting', 'SVM']:
                print(f"Using RandomizedSearchCV with {n_iter} iterations...")
                
                if model_name == 'SVM':
                    # For SVM with pre-generated compatible params, use a different approach
                    from sklearn.model_selection import ParameterSampler
                    sampled_params = list(ParameterSampler(
                        param_grid if not isinstance(param_grid, list) else {'dummy': [1]}, 
                        n_iter=min(n_iter, len(param_grid) if isinstance(param_grid, list) else n_iter),
                        random_state=42
                    ))
                    
                    if isinstance(param_grid, list):
                        sampled_params = param_grid[:n_iter]
                    
                    best_score = -1
                    best_params = None
                    best_model = None
                    
                    for i, params in enumerate(sampled_params):
                        try:
                            temp_model = SVC(**params, random_state=42)
                            scores = cross_val_score(temp_model, X_train, y_train, 
                                                   cv=cv_strategy, scoring=scoring)
                            avg_score = scores.mean()
                            
                            if avg_score > best_score:
                                best_score = avg_score
                                best_params = params
                                best_model = temp_model
                            
                            if (i + 1) % 20 == 0:
                                print(f"Evaluated {i + 1}/{len(sampled_params)} combinations, "
                                     f"best score: {best_score:.4f}")
                                
                        except Exception as e:
                            continue
                    
                    # Fit the best model
                    if best_model is not None:
                        best_model.fit(X_train, y_train)
                        grid_search = type('GridSearch', (), {
                            'best_estimator_': best_model,
                            'best_params_': best_params,
                            'best_score_': best_score
                        })()
                    else:
                        raise ValueError("No valid SVM parameters found")
                        
                else:
                    grid_search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=param_grid,
                        n_iter=n_iter,
                        cv=cv_strategy,
                        scoring=scoring,
                        n_jobs=n_jobs,
                        random_state=42,
                        verbose=1
                    )
                    grid_search.fit(X_train, y_train)
            else:
                print("Using GridSearchCV...")
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=cv_strategy,
                    scoring=scoring,
                    n_jobs=n_jobs,
                    verbose=1
                )
                grid_search.fit(X_train, y_train)
            
            # Store results
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            cv_score = grid_search.best_score_
            
            # Evaluate on test set
            y_pred = best_model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            test_precision = precision_score(y_test, y_pred, average='weighted')
            test_recall = recall_score(y_test, y_pred, average='weighted')
            test_f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Store all results
            results['best_models'][model_name] = best_model
            results['best_params'][model_name] = best_params
            results['cv_scores'][model_name] = cv_score
            results['test_scores'][model_name] = {
                'accuracy': test_accuracy,
                'precision': test_precision,
                'recall': test_recall,
                'f1_score': test_f1
            }
            results['training_times'][model_name] = time.time() - start_time
            results['grid_search_objects'][model_name] = grid_search
            
            # Print results
            print(f"Best parameters: {best_params}")
            print(f"Best CV score: {cv_score:.4f}")
            print(f"Test accuracy: {test_accuracy:.4f}")
            print(f"Test F1-score: {test_f1:.4f}")
            print(f"Training time: {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error optimizing {model_name}: {str(e)}")
            # Fall back to default model
            model.fit(X_train, y_train)
            results['best_models'][model_name] = model
            results['best_params'][model_name] = f"Default (error: {str(e)})"
            continue
    
    return results

def visualize_hyperparameter_results(results):
    """
    Create comprehensive visualizations of hyperparameter tuning results.
    """
    # Extract data for visualization
    model_names = list(results['test_scores'].keys())
    accuracies = [results['test_scores'][name]['accuracy'] for name in model_names]
    f1_scores = [results['test_scores'][name]['f1_score'] for name in model_names]
    cv_scores = [results['cv_scores'][name] for name in model_names]
    training_times = [results['training_times'][name] for name in model_names]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comprehensive Hyperparameter Tuning Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Test Accuracy Comparison
    axes[0, 0].bar(model_names, accuracies, color='skyblue', alpha=0.8)
    axes[0, 0].set_title('Test Accuracy by Model')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: CV Score vs Test Accuracy
    axes[0, 1].scatter(cv_scores, accuracies, s=100, alpha=0.7, color='red')
    for i, name in enumerate(model_names):
        axes[0, 1].annotate(name, (cv_scores[i], accuracies[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
    axes[0, 1].set_xlabel('CV Score')
    axes[0, 1].set_ylabel('Test Accuracy')
    axes[0, 1].set_title('CV Score vs Test Accuracy')
    axes[0, 1].grid(alpha=0.3)
    
    # Plot 3: F1-Score Comparison
    axes[1, 0].bar(model_names, f1_scores, color='lightgreen', alpha=0.8)
    axes[1, 0].set_title('F1-Score by Model')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(f1_scores):
        axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Training Time Comparison
    axes[1, 1].barh(model_names, training_times, color='orange', alpha=0.8)
    axes[1, 1].set_title('Training Time by Model')
    axes[1, 1].set_xlabel('Time (seconds)')
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(training_times):
        axes[1, 1].text(v + max(training_times) * 0.01, i, f'{v:.1f}s', 
                       va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Create results summary DataFrame
    summary_data = []
    for name in model_names:
        summary_data.append({
            'Model': name,
            'Test_Accuracy': results['test_scores'][name]['accuracy'],
            'Test_Precision': results['test_scores'][name]['precision'],
            'Test_Recall': results['test_scores'][name]['recall'],
            'Test_F1': results['test_scores'][name]['f1_score'],
            'CV_Score': results['cv_scores'][name],
            'Training_Time': results['training_times'][name]
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Test_Accuracy', ascending=False)
    
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING RESULTS SUMMARY")
    print("="*80)
    print(summary_df.round(4).to_string(index=False))
    
    return summary_df

# Example usage:
"""
# Initialize models
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(random_state=42),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'KNeighbors': KNeighborsClassifier(),
    'GaussianNB': GaussianNB()
}

# Perform comprehensive hyperparameter tuning
results = perform_comprehensive_hyperparameter_tuning(
    X_train_scaled, y_train, X_test_scaled, y_test,
    models, expanded_param_grids,
    use_randomized_search=True,
    n_iter=150,  # Adjust based on computational resources
    cv_folds=5,
    scoring='accuracy'
)

# Visualize results
summary_df = visualize_hyperparameter_results(results)

# Print best parameters for top performing models
print("\nBEST PARAMETERS FOR TOP 3 MODELS:")
print("="*50)
top_3_models = summary_df.head(3)['Model'].tolist()

for model_name in top_3_models:
    print(f"\n{model_name}:")
    print("-" * 30)
    best_params = results['best_params'][model_name]
    if isinstance(best_params, dict):
        for param, value in best_params.items():
            print(f"  {param}: {value}")
    else:
        print(f"  {best_params}")
"""

print("Expanded hyperparameter tuning framework created successfully!")
print("This framework includes:")
print("- Comprehensive parameter grids for Random Forest and SVM")
print("- Smart parameter compatibility handling for SVM")
print("- Efficient RandomizedSearchCV for large parameter spaces")
print("- Comprehensive visualization and reporting")
print("- Support for multiple models and metrics")