import os, time, joblib, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'bank-additional-full.csv')

df = pd.read_csv(DATA_PATH, sep=';').drop(columns=['duration'], errors='ignore')
X = df.drop('y', axis=1)
y = df['y'].map({'yes': 1, 'no': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(exclude=['object']).columns
preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numerical_cols), ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)])

results = []
models = {'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'), 'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'), 'XGBoost': XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE)}

for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    f1 = f1_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_proba)
    k = int(len(y_test) * 0.20)
    top_indices = np.argsort(y_proba)[::-1][:k]
    precision_at_20 = y_test.iloc[top_indices].mean()
    results.append({'Model': name, 'F1': f1, 'ROC-AUC': roc_auc_score(y_test, y_proba), 'PR-AUC': pr_auc, 'Precision@20%': precision_at_20})
    print(f"Trained {name} | F1: {f1:.4f}")

pd.DataFrame(results).to_csv(os.path.join(BASE_DIR, 'reports', 'performance_table.csv'), index=False)
print('Evaluation complete (TabPFN skipped due to license).')
