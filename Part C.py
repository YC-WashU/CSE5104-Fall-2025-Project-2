import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ----- 1. Read Data & Dimension Reduction & Data Split & Save Files -----

def prepare_reduced_datasets():
    print("Start data preprocessing and dimension reduction (using PCA)...")
    
    # Read Files Created in Part A
    try:
        df_binary = pd.read_csv('wine_binary.csv')
        df_multi = pd.read_csv('wine_multiclass.csv')
    except FileNotFoundError:
        print("No File Found")
        return None

    # Difine Dimension Reduction Function
    def process_and_reduce(df, target_col, filename_prefix):
        # Separate features and labels
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # (1) Split (90% Train, 10% Validation)
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            X, y, test_size=0.10, random_state=1, stratify=y
        )
        
        # (2) Standardization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_val_scaled = scaler.transform(X_val_raw)
        
        # (3) Apply PCA
        pca = PCA(n_components=5, random_state=1)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_val_pca = pca.transform(X_val_scaled)
        
        print(f"[{filename_prefix}] Explained Variance Ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
        
        # (4) Reorganize and Save
        cols = [f'PC{i+1}' for i in range(5)]  # Create new column names PC1, PC2, PC3, PC4, PC5
        
        # Training Set
        df_train_reduced = pd.DataFrame(X_train_pca, columns=cols)
        df_train_reduced[target_col] = y_train.values
        train_file = f'{filename_prefix}_reduced_train.csv'
        df_train_reduced.to_csv(train_file, index=False)
        
        # Validation Set
        df_val_reduced = pd.DataFrame(X_val_pca, columns=cols)
        df_val_reduced[target_col] = y_val.values
        val_file = f'{filename_prefix}_reduced_val.csv'
        df_val_reduced.to_csv(val_file, index=False)
        
        print(f"Datasets saved successfully: {train_file}, {val_file}")
        
        return train_file, val_file

    # Binary Dataset
    bin_train, bin_val = process_and_reduce(df_binary, 'High_Quality', 'wine_binary')
    
    # Multi-class Dataset
    multi_train, multi_val = process_and_reduce(df_multi, 'Quality_Tier', 'wine_multi')
    
    return bin_train, bin_val, multi_train, multi_val

# Read Data & Dimension Reduction & Data Split & Save Files
files = prepare_reduced_datasets()
if files[0] is None:
    exit()

# ----- 2. Model Configurations -----
models_config = {
    'kNN': {
        'model': KNeighborsClassifier(),
        'param_name': 'n_neighbors',
        'param_grid': {'n_neighbors': [1, 2, 3, 5, 8, 11, 15, 21]},
        'type': 'numerical'
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=1),
        'param_name': 'max_depth',
        'param_grid': {'max_depth': [2, 5, 8, 12, 16, 20, 30, 40]}, 
        'type': 'numerical'
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=1),
        'param_name': 'n_estimators',
        'param_grid': {'n_estimators': [5, 25, 50, 75, 100, 150, 200, 300]},
        'type': 'numerical'
    },
    'SVM': {
        'model': SVC(random_state=1),
        'param_name': 'kernel',
        'param_grid': {'kernel': ['linear', 'poly', 'rbf']},
        'type': 'categorical'
    }
}

# -----3. Training & Evaluation -----

def run_part_c_task(train_file, val_file, target_col, task_name):
    print(f"\n{'='*20} Part C: {task_name} (Reduced Data) {'='*20}")
    
    # Read data after PCA
    df_train = pd.read_csv(train_file)
    df_val = pd.read_csv(val_file)
    
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]
    X_val = df_val.drop(columns=[target_col])
    y_val = df_val[target_col]
    
    # Plot Preparations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'CV Results (PCA Reduced) - {task_name}(k=10)', fontsize=16)
    axes = axes.flatten()
    
    results_list = []
    
    for i, (name, config) in enumerate(models_config.items()):
        print(f"\n--- Model: {name} ---")
        
        # A. Cross Validation
        grid = GridSearchCV(config['model'], config['param_grid'], cv=10, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train) # Apply 10-fold CV
        
        best_param = grid.best_params_[config['param_name']]
        best_cv_score = grid.best_score_
        
        # B. Visualization
        cv_results = grid.cv_results_
        params = cv_results['param_' + config['param_name']].data
        scores = cv_results['mean_test_score']
        
        ax = axes[i]
        if config['type'] == 'numerical':
            params = params.astype(int)
            sorted_idx = np.argsort(params)
            ax.plot(params[sorted_idx], scores[sorted_idx], marker='o', linestyle='-', color='g') 
        else:
            ax.bar(params, scores, color='lightgreen', edgecolor='black')

        ax.annotate(f'{best_cv_score:.4f}', 
                    xy=(best_param, best_cv_score), 
                    xytext=(0, 1),             
                    textcoords='offset points',
                    ha='center',               
                    va='bottom',               
                    fontsize=9,                
                    color='red',               
                    fontweight='bold')
        
        ax.set_title(f'{name} (Best: {best_param})')
        ax.set_xlabel(config['param_name'])
        ax.set_ylabel('Mean CV Accuracy (k=10)')
        ax.grid(True, alpha=0.3)
        
        # C. Final Training on full Train Set
        best_model = grid.best_estimator_
        
        start_train = time.time()
        best_model.fit(X_train, y_train) # Refit using optimal parameters
        train_time = time.time() - start_train
        
        # D. Calculate Training Performance
        y_train_pred = best_model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        
        # E. Calculate Validation Performance
        start_pred = time.time()
        y_pred = best_model.predict(X_val)
        pred_time = time.time() - start_pred
        val_acc = accuracy_score(y_val, y_pred)
        
        # Show the result, including tables
        print(f"Chosen Hyperparameter: {best_param}")
        print(f"Training Accuracy:   {train_acc:.4f} (Time: {train_time:.6f}s)")
        print(f"Validation Accuracy: {val_acc:.4f} (Time: {pred_time:.6f}s)")
        
        results_list.append({
            'Model': name,
            'Best Param': best_param,
            'Train Acc': train_acc,
            'Val Acc': val_acc,
            'Train Time': train_time,
            'Pred Time': pred_time
        })

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    return pd.DataFrame(results_list)

# Run
res_binary_c = run_part_c_task(files[0], files[1], 'High_Quality', 'Binary Classification')
res_multi_c = run_part_c_task(files[2], files[3], 'Quality_Tier', 'Multi-class Classification')

print("\n=== Part C Summary: Binary Classification ===")
print(res_binary_c)
print("\n=== Part C Summary: Multi-class Classification ===")
print(res_multi_c)
