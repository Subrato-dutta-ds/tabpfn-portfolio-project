import os, joblib, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from src.data_loader import load_data

RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)

df = load_data()
X = df.drop('y', axis=1)

# CRITICAL FIX: Map 'yes' to 1 and 'no' to 0 for XGBoost
y = df['y'].map({'yes': 1, 'no': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(exclude=['object']).columns
preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numerical_cols), ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(eval_metric='logloss', class_weight='balanced', random_state=RANDOM_STATE)
}

for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    pipeline.fit(X_train, y_train)
    joblib.dump(model, os.path.join(BASE_DIR, 'models', f'{name.lower().replace(" ", "_")}.pkl'))
    print(f"Saved {name}")

full_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', models["Random Forest"])])
full_pipeline.fit(X_train, y_train)
joblib.dump(full_pipeline, os.path.join(BASE_DIR, 'models', 'model_pipeline.pkl'))
print("Full pipeline saved.")
