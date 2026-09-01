import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from tabpfn import TabPFNClassifier
from xgboost import XGBClassifier

# ---------------------------
# 0. DYNAMIC PATH SETUP
# ---------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'bank-additional-full.csv')

# ---------------------------
# 1. Load and Prepare Data
# ---------------------------
if not os.path.exists(DATA_PATH):
    print(f"Error: File not found at {DATA_PATH}")
    exit()

df = pd.read_csv(DATA_PATH, sep=';')

X = df.drop('y', axis=1)
y = df['y']

categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(exclude=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# 2. Define Models
# ---------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
}

# ---------------------------
# 3. Train and Evaluate (Standard Models)
# ---------------------------
results = []

for name, model in models.items():
    print(f"Training {name}...")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    
    # Calculate probabilities for ROC-AUC
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label='yes')
    rec = recall_score(y_test, y_pred, pos_label='yes')
    f1 = f1_score(y_test, y_pred, pos_label='yes')
    
    # Updated order: F1 first, then Recall, ROC-AUC, Accuracy, Precision
    results.append({
        "Model": name, 
        "F1 Score": f1, 
        "Recall": rec, 
        "ROC-AUC": auc, 
        "Accuracy": acc, 
        "Precision": prec
    })
    print(f"  {name} F1: {f1:.4f}, Recall: {rec:.4f}, AUC: {auc:.4f}")

# ---------------------------
# 4. Train and Evaluate TabPFN (Separate Process)
# ---------------------------
print("Training TabPFN...")
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

try:
    tabpfn = TabPFNClassifier(device='cpu', random_state=42)
    tabpfn.fit(X_train_transformed, y_train)
    
    os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
    joblib.dump(tabpfn, os.path.join(BASE_DIR, 'models', 'tabpfn_model.pkl'))
    
    y_pred_tabpfn = tabpfn.predict(X_test_transformed)
    y_pred_proba_tabpfn = tabpfn.predict_proba(X_test_transformed)[:, 1]
    
    acc_tabpfn = accuracy_score(y_test, y_pred_tabpfn)
    prec_tabpfn = precision_score(y_test, y_pred_tabpfn, pos_label='yes')
    rec_tabpfn = recall_score(y_test, y_pred_tabpfn, pos_label='yes')
    f1_tabpfn = f1_score(y_test, y_pred_tabpfn, pos_label='yes')
    auc_tabpfn = roc_auc_score(y_test, y_pred_proba_tabpfn)
    
    results.append({
        "Model": "TabPFN", 
        "F1 Score": f1_tabpfn, 
        "Recall": rec_tabpfn, 
        "ROC-AUC": auc_tabpfn, 
        "Accuracy": acc_tabpfn, 
        "Precision": prec_tabpfn
    })
    print(f"  TabPFN F1: {f1_tabpfn:.4f}, Recall: {rec_tabpfn:.4f}, AUC: {auc_tabpfn:.4f}")
except Exception as e:
    print(f"TabPFN training failed (likely due to dataset size): {e}")
    results.append({
        "Model": "TabPFN", "F1 Score": 0.0, "Recall": 0.0, "ROC-AUC": 0.0, "Accuracy": 0.0, "Precision": 0.0
    })

# ---------------------------
# 5. Save Results
# ---------------------------
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='F1 Score', ascending=False)

os.makedirs(os.path.join(BASE_DIR, 'reports'), exist_ok=True)
results_df.to_csv(os.path.join(BASE_DIR, 'reports', 'performance_table.csv'), index=False)

training_history = pd.DataFrame({
    "Model": results_df["Model"],
    "Status": ["Trained"] * len(results_df)
})
training_history.to_csv(os.path.join(BASE_DIR, 'reports', 'training_results.csv'), index=False)

print("\nAll models evaluated!")
print("Results saved to 'reports/performance_table.csv' (Sorted by F1 Score)")