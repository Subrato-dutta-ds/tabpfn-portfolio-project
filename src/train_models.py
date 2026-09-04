import os, joblib, pandas as pd, numpy as np, json
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, average_precision_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from src.data_loader import load_data

RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'reports'), exist_ok=True)

# Load Data and Map Target
df = load_data()
X = df.drop('y', axis=1)
y = df['y'].map({'yes': 1, 'no': 0})

# Single Split for Training/Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# Preprocessing
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(exclude=['object']).columns
preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numerical_cols), ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)])

# Models and Grids
pipelines = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE)
}
param_grids = {
    "Random Forest": {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [None, 10, 20]},
    "XGBoost": {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [3, 5, 7], 'classifier__learning_rate': [0.01, 0.1]}
}

trained_pipelines = {}
print("--- Training & Tuning (5-Fold CV) ---")

for name, model in pipelines.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    if name in param_grids:
        # Fix #5: StratifiedKFold with shuffle and random_state
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        search = RandomizedSearchCV(pipeline, param_grids[name], n_iter=3, cv=cv, scoring='f1', n_jobs=-1, random_state=RANDOM_STATE)
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_
    else:
        pipeline.fit(X_train, y_train)
    
    trained_pipelines[name] = pipeline
    print(f"Trained {name}")

# Fix #4: Determine threshold on TRAINING data (OOF probabilities) to prevent test leakage
print("--- Optimizing Threshold on Training Data (OOF) ---")
best_f1 = 0
best_thresh = 0.5
for name, pipeline in trained_pipelines.items():
    # Generate OOF predictions on training set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = cross_val_predict(pipeline, X_train, y_train, cv=cv, method='predict_proba')[:, 1]
    
    for thresh in np.arange(0.3, 0.7, 0.05):
        f1 = f1_score(y_train, (oof_proba >= thresh).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_model_name = name

print(f"Best Model: {best_model_name}, Best Threshold: {best_thresh:.2f}, Train OOF F1: {best_f1:.4f}")

# Save the TUNED pipeline (Fix #1)
best_pipeline = trained_pipelines[best_model_name]
joblib.dump(best_pipeline, os.path.join(BASE_DIR, 'models', 'model_pipeline.pkl'))

# Save metadata
with open(os.path.join(BASE_DIR, 'models', 'model_metadata.json'), 'w') as f:
    json.dump({"model": best_model_name, "threshold": best_thresh, "primary_metric": "F1"}, f, indent=4)

print("Production model saved.")
