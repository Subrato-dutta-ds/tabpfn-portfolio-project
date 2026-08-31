import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from tabpfn import TabPFNClassifier
from xgboost import XGBClassifier

# ---------------------------
# 1. Load and Prepare Data
# ---------------------------
data_path = r'C:\Users\subrato dutta\tabpfn-portfolio-project\data\raw\bank-additional-full.csv'
if not os.path.exists(data_path):
    print(f"Error: File not found at {data_path}")
    exit()

# Bank dataset uses semicolon separator
df = pd.read_csv(data_path, sep=';')

# Target is 'y'
X = df.drop('y', axis=1)
y = df['y']

# Define column types
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(exclude=['object']).columns

# Create Preprocessor (Same as training)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# 2. Define Models
# ---------------------------
# Make sure you have xgboost installed: pip install xgboost
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# TabPFN is added to the list, but we handle it separately because it is strict
# about fitting on transformed (scaled/encoded) data only.
model_names = list(models.keys()) + ["TabPFN"]

# ---------------------------
# 3. Train and Evaluate
# ---------------------------
results = []

for name, model in models.items():
    print(f"Training {name}...")
    # Create Pipeline for sklearn models
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Predict
    y_pred = pipeline.predict(X_test)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label='yes')
    rec = recall_score(y_test, y_pred, pos_label='yes')
    f1 = f1_score(y_test, y_pred, pos_label='yes')
    
    results.append({
        "Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1
    })
    print(f"  {name} Accuracy: {acc:.4f}")

# ---------------------------
# 4. Train and Evaluate TabPFN (Separate Process)
# ---------------------------
print("Training TabPFN...")
# TabPFN requires the data to be numeric and scaled. 
# We fit the preprocessor on training data and transform both train and test.
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# TabPFN expects numpy arrays
# Note: TabPFN is designed for small datasets (<10k samples). If your dataset is large, this might take a while.
try:
    tabpfn = TabPFNClassifier(device='cpu', random_state=42)
    tabpfn.fit(X_train_transformed, y_train)
    y_pred_tabpfn = tabpfn.predict(X_test_transformed)
    
    acc_tabpfn = accuracy_score(y_test, y_pred_tabpfn)
    prec_tabpfn = precision_score(y_test, y_pred_tabpfn, pos_label='yes')
    rec_tabpfn = recall_score(y_test, y_pred_tabpfn, pos_label='yes')
    f1_tabpfn = f1_score(y_test, y_pred_tabpfn, pos_label='yes')
    
    results.append({
        "Model": "TabPFN", "Accuracy": acc_tabpfn, "Precision": prec_tabpfn, "Recall": rec_tabpfn, "F1 Score": f1_tabpfn
    })
    print(f"  TabPFN Accuracy: {acc_tabpfn:.4f}")
except Exception as e:
    print(f"TabPFN training failed (likely due to dataset size): {e}")
    # Create a placeholder row so the app doesn't crash
    results.append({
        "Model": "TabPFN", "Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "F1 Score": 0.0
    })

# ---------------------------
# 5. Save Results
# ---------------------------
results_df = pd.DataFrame(results)
os.makedirs('reports', exist_ok=True)
results_df.to_csv('reports/performance_table.csv', index=False)

# Save Training History (placeholder for your Streamlit UI)
training_history = pd.DataFrame({
    "Model": results_df["Model"],
    "Status": ["Trained"] * len(results_df)
})
training_history.to_csv('reports/training_results.csv', index=False)

print("\nAll models evaluated!")
print("Results saved to 'reports/performance_table.csv'")