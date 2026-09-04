import os, joblib, pandas as pd, numpy as np, json
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from src.data_loader import load_data

RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'reports'), exist_ok=True)

df = load_data()
X = df.drop('y', axis=1)
y = df['y'].map({'yes': 1, 'no': 0})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(exclude=['object']).columns
preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numerical_cols), ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)])

# Fix #8: Separate hyperparameter grids
param_grids = {
    "Random Forest": {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_leaf': [1, 2, 4]  # This is correct for RF
    },
    "XGBoost": {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__subsample': [0.8, 1.0]
    }
}

results = []
pipelines = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(eval_metric='logloss', class_weight='balanced', random_state=RANDOM_STATE)
}

for name, model in pipelines.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    
    # Tune models
    if name in param_grids:
        search = RandomizedSearchCV(pipeline, param_grids[name], n_iter=3, cv=StratifiedKFold(n_splits=3), scoring='f1', n_jobs=-1, random_state=RANDOM_STATE)
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_
    else:
        pipeline.fit(X_train, y_train)

    # Evaluate and optimize threshold
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    best_thresh = 0.5
    best_f1 = 0
    for thresh in np.arange(0.3, 0.7, 0.05):
        y_pred_thresh = (y_proba >= thresh).astype(int)
        f1 = f1_score(y_test, y_pred_thresh)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    y_pred = (y_proba >= best_thresh).astype(int)
    f1 = f1_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    
    k = int(len(y_test) * 0.20)
    top_indices = np.argsort(y_proba)[::-1][:k]
    precision_at_20 = y_test.iloc[top_indices].mean()
    
    results.append({'Model': name, 'F1': f1, 'ROC-AUC': auc, 'PR-AUC': pr_auc, 'Precision@20%': precision_at_20, 'Optimal Threshold': round(best_thresh, 2)})
    print(f"{name} -> F1: {f1:.4f}, P@20: {precision_at_20:.4f}, Thresh: {best_thresh:.2f}")

# Save report
results_df = pd.DataFrame(results).sort_values(by='F1', ascending=False)
results_df.to_csv(os.path.join(BASE_DIR, 'reports', 'performance_table.csv'), index=False)

# Fix #1, #2, #9: Select best model, save it as the production pipeline
best_model_name = results_df.iloc[0]['Model']
best_model = pipelines[best_model_name]
production_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', best_model)])
production_pipeline.fit(X_train, y_train)

joblib.dump(production_pipeline, os.path.join(BASE_DIR, 'models', 'model_pipeline.pkl'))

# Save metadata
metadata = {
    "model": best_model_name,
    "threshold": float(results_df.iloc[0]['Optimal Threshold']),
    "primary_metric": "F1",
    "f1_score": float(results_df.iloc[0]['F1'])
}
with open(os.path.join(BASE_DIR, 'models', 'model_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=4)

print(f"Production model: {best_model_name} saved with threshold {metadata['threshold']}")
