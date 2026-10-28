import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report

def generate_synthetic_data(num_samples=8000):
    """
    Simulates a robust civic dataset mapping the 10 core feature boundaries 
    into an optimized target mathematical constraint logic.
    Provides necessary multi-dimensional data arrays since no CSV was supplied natively.
    """
    np.random.seed(42)  # Secure pseudo-random generation limit for evaluation consistency
    categories = ['Roads', 'Sanitation', 'Water', 'Electricity', 'Other']
    severities = ['Low', 'Medium', 'High']
    
    # Generate structural columns
    df = pd.DataFrame({
        'complaint_category': np.random.choice(categories, num_samples),
        'severity_score': np.random.choice(severities, num_samples, p=[0.4, 0.4, 0.2]),
        'report_count': np.random.randint(1, 50, num_samples),
        'population_density': np.random.uniform(500, 25000, num_samples), # physical people per sq km metrics
        'traffic_density': np.random.uniform(0, 100, num_samples),        # percentage impact map
        'ward_infrastructure_age': np.random.randint(5, 100, num_samples),# historical years
        'distance_to_hospital': np.random.uniform(50, 5000, num_samples), # mapped limits natively in meters
        'distance_to_school': np.random.uniform(50, 3000, num_samples),   # mapped limits natively in meters
        'is_main_road': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
        'time_since_last_repair': np.random.randint(1, 120, num_samples)  # tracking degradation over months
    })
    
    # Compound Underlying Base Target Priority Logics directly into a function
    # Allows the ML RF Algorithm to reverse-engineer and smooth edges cleanly over large evaluation paths
    def calculate_priority(row):
        score = 0
        # Categorical ordinal weight assignments mapping visual hazard definitions
        if row['severity_score'] == 'High': score += 40
        elif row['severity_score'] == 'Medium': score += 15
        
        # Geodistancing structural bounds triggering high urgency
        score += row['report_count'] * 1.5
        if row['is_main_road'] == 1: score += 20
        if row['distance_to_hospital'] < 500: score += 15
        if row['distance_to_school'] < 500: score += 10
        if row['traffic_density'] > 80: score += 15
        if row['ward_infrastructure_age'] > 50: score += 10
        
        # P1 = Critical, P2 = High, P3 = Medium, P4 = Low
        if score > 100: return 'P1'
        elif score > 70: return 'P2'
        elif score > 40: return 'P3'
        else: return 'P4'
        
    df['priority_class'] = df.apply(calculate_priority, axis=1)
    
    # Inject 5% systemic noise strictly to ensure the RandomForest handles actual classification 
    # instead of behaving like a perfect deterministic calculator structure inherently bypassing accuracy metric proofs
    noise_indices = np.random.choice(df.index, size=int(num_samples * 0.05), replace=False)
    df.loc[noise_indices, 'priority_class'] = np.random.choice(['P1', 'P2', 'P3', 'P4'], size=len(noise_indices))
    
    return df

def build_and_train_model():
    # 1. Pipeline Data Matrix Generation
    print("[1] Building massive 10-Variable feature subset limits securely to target vectors...")
    df = generate_synthetic_data(num_samples=8000)
    
    X = df.drop('priority_class', axis=1)
    y = df['priority_class']

    # 2. Scikit-Learn Processing Segregation Constraints
    numeric_features = [
        'report_count', 'population_density', 'traffic_density', 
        'ward_infrastructure_age', 'distance_to_hospital', 
        'distance_to_school', 'time_since_last_repair'
    ]
    categorical_features = ['complaint_category']
    # Explicitly enforce hierarchical bounds against the ML evaluation tree natively mapped
    ordinal_features = ['severity_score'] 
    bool_features = ['is_main_road'] 

    # Standardization pipeline transformations
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    ordinal_transformer = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])

    # Unified integration wrapper natively pushing variable types mapping to identical sequence tensors 
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('ord', ordinal_transformer, ordinal_features),
            ('bool', 'passthrough', bool_features) 
        ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"[2] Mapped Training Chunk: {X_train.shape[0]} | Validation Chunk: {X_test.shape[0]}")

    # 3. Model Architecture Instantiation mapping identical inputs via Sklearn Pipeline limit optimizations
    # Automatically triggers 100 Ensembled Base Classifiers inherently bound to optimal depth
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    # Dynamic Hyperparameter Evaluation boundaries natively executed via 3-Fold CV
    param_grid = {
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }
    
    print("[3] Launching robust parallel GridSearchCV optimizations targeting >=90% classification Accuracy...")
    grid_search = GridSearchCV(rf_pipeline, param_grid, cv=3, n_jobs=-1, scoring='accuracy', verbose=1)
    
    # Backpropogate RF nodes over structured splits
    grid_search.fit(X_train, y_train)
    
    print(f"\n[Validation] Selected Output Topology Structure Map: {grid_search.best_params_}")

    # 4. Extract Analytical Final Evaluation Constraints
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print("\n--- Final ML Evaluation Outputs (Target Limit: >= 0.9000) ---")
    report = classification_report(y_test, y_pred, digits=4)
    print(report)

    # 5. Extract Complex Internal Node Importance Thresholds Mapping Feature Contributions inherently
    # Access the FITTED preprocessor from inside the best pipeline (not the unfitted standalone one)
    fitted_preprocessor = best_model.named_steps['preprocessor']
    try:
        ohe_categories = fitted_preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    except AttributeError:
        ohe_categories = [f"cat_{c}" for c in range(fitted_preprocessor.named_transformers_['cat'].categories_[0].shape[0])]
    feature_names = numeric_features + list(ohe_categories) + ordinal_features + bool_features
    
    # Track the native Gini Importance parameters directly calculated inside the classification nodes
    importances = best_model.named_steps['classifier'].feature_importances_
    
    # Generate visualization using explicit dynamic matplotlib limits natively tracking the values mathematically
    indices = np.argsort(importances)
    plt.figure(figsize=(10, 8))
    plt.title("Master Ticket AI Priority Architecture -> Analyzed Internal Feature Map Weights")
    plt.barh(range(len(importances)), importances[indices], color='indigo', align='center')
    plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
    plt.xlabel('Gini Structural Significance Matrix')
    plt.tight_layout()
    
    plot_path = "feature_importance.png"
    plt.savefig(plot_path, dpi=300)
    print(f"\n[Artifact Computed] Safely compiled algorithmic structure matrix boundaries to -> '{plot_path}'")

    # 6. Final Model Export
    model_path = "rf_priority_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"[Success] Written binary architecture state natively to -> '{model_path}'")

if __name__ == "__main__":
    build_and_train_model()
