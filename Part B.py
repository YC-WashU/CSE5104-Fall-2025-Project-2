import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ----- 1. Data Splitting & Saving -----

def prepare_and_save_splits():
    # Read Files Created in Part A
    try:
        df_binary = pd.read_csv('winequality_binary.csv')
        df_multi = pd.read_csv('winequality_multiclass.csv')
    except FileNotFoundError:
        print("No file found.")
        return

    # 1. Split Binary Data
    train_b, val_b = train_test_split(
        df_binary, test_size=0.10, random_state=1, stratify=df_binary['High_Quality']
    )
    # Save
    train_b.to_csv('wine_binary_train.csv', index=False)
    val_b.to_csv('wine_binary_val.csv', index=False)
    print("Splitted dataset successfully saved: wine_binary_train.csv, wine_binary_val.csv")

    # 2. Split Multi-class Data
    train_m, val_m = train_test_split(
        df_multi, test_size=0.10, random_state=1, stratify=df_multi['Quality_Tier']
    )
    # Save
    train_m.to_csv('wine_multi_train.csv', index=False)
    val_m.to_csv('wine_multi_val.csv', index=False)
    print("Splitted dataset successfully saved: wine_multi_train.csv, wine_multi_val.csv")

# Run the split process:
prepare_and_save_splits()


# ----- 2. Model Configuration -----
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
        'param_grid': {'kernel': ['linear', 'poly', 'rbf'],},
        'type': 'categorical'
    }
}


# ----- 3. Training & Evaluation -----

def Train_Eval(train_file, val_file, target_col, task_name):
    print(f"\n{'='*20} Part B: Processing {task_name} {'='*20}")
    
    # (1) Read data
    df_train = pd.read_csv(train_file)
    df_val = pd.read_csv(val_file)
    
    # Separate features and labels 
    X_train_raw = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]
    X_val_raw = df_val.drop(columns=[target_col])
    y_val = df_val[target_col]
    
    # (2) Standardized (Scaling)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    
    # Plot Preparations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Cross-Validation Results: {task_name} (k=10)', fontsize=16)
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
            ax.plot(params[sorted_idx], scores[sorted_idx], marker='o', linestyle='-', color='b')
        else:
            ax.bar(params, scores, color='skyblue', edgecolor='black')

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
        ax.set_ylabel('Mean CV Accuracy')
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
        
        # F. Print Result
        print(f"Chosen Hyperparameter: {best_param}")
        print(f"Training Accuracy:   {train_acc:.4f} (Time: {train_time:.6f}s)")
        print(f"Validation Accuracy: {val_acc:.4f} (Time: {pred_time:.6f}s)")
        
        # G. Create a table 
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
res_binary = Train_Eval('wine_binary_train.csv', 'wine_binary_val.csv', 'High_Quality', 'Binary Classification')
res_multi = Train_Eval('wine_multi_train.csv', 'wine_multi_val.csv', 'Quality_Tier', 'Multi-class Classification')

print("\n=== Part B Summary: Binary Classification ===")
print(res_binary)
print("\n=== Part B Summary: Multi-class Classification===")
print(res_multi)
