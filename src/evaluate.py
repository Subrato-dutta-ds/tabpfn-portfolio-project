import os, joblib, pandas as pd, numpy as np, json
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from src.data_loader import load_data

RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(BASE_DIR, 'reports'), exist_ok=True)

df = load_data()
X = df.drop('y', axis=1)
y = df['y'].map({'yes': 1, 'no': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# Save exact X_test for SHAP consistency
X_test.to_csv(os.path.join(BASE_DIR, 'reports', 'X_test.csv'), index=False)
pd.DataFrame({'y': y_test}).to_csv(os.path.join(BASE_DIR, 'reports', 'y_test.csv'), index=False)

categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(exclude=['object']).columns
preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numerical_cols), ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)])

pipelines = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE)
}
param_grids = {
    "Random Forest": {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [None, 10, 20]},
    "XGBoost": {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [3, 5, 7], 'classifier__learning_rate': [0.01, 0.1]}
}

results = []
trained_pipelines = {}

for name, model in pipelines.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    if name in param_grids:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        search = RandomizedSearchCV(pipeline, param_grids[name], n_iter=3, cv=cv, scoring='f1', n_jobs=-1, random_state=RANDOM_STATE)
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_
    else:
        pipeline.fit(X_train, y_train)
    
    trained_pipelines[name] = pipeline
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    f1 = f1_score(y_test, (y_proba >= 0.5).astype(int))
    pr_auc = average_precision_score(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    k = int(len(y_test) * 0.20)
    top_indices = np.argsort(y_proba)[::-1][:k]
    p_at_20 = y_test.iloc[top_indices].mean()
    
    results.append({'Model': name, 'F1': f1, 'PR-AUC': pr_auc, 'ROC-AUC': auc, 'Precision@20%': p_at_20})

# Threshold optimization on OOF predictions
best_f1 = 0
best_thresh = 0.5
best_model_name = "Random Forest"
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
for name, pipeline in trained_pipelines.items():
    oof_proba = cross_val_predict(pipeline, X_train, y_train, cv=cv, method='predict_proba')[:, 1]
    for thresh in np.arange(0.3, 0.7, 0.05):
        f1 = f1_score(y_train, (oof_proba >= thresh).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_model_name = name

# Save results CSV
pd.DataFrame(results).sort_values(by='F1', ascending=False).to_csv(os.path.join(BASE_DIR, 'reports', 'performance_table.csv'), index=False)

# Save metadata with best model and threshold
metadata = {"model": best_model_name, "threshold": best_thresh, "primary_metric": "F1"}
with open(os.path.join(BASE_DIR, 'models', 'model_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=4)

print(f"Final results: {results}")
print(f"Best model: {best_model_name}, Best threshold: {best_thresh}")
