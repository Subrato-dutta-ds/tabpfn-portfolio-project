import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'bank-additional-full.csv')

df = pd.read_csv(DATA_PATH, sep=';')
df = df.drop(columns=['duration'], errors='ignore')
X = df.drop('y', axis=1)
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(exclude=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE))
])

pipeline.fit(X_train, y_train)
print(f"Model trained. Accuracy: {pipeline.score(X_test, y_test):.4f}")

os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
joblib.dump(pipeline, os.path.join(BASE_DIR, 'models', 'model_pipeline.pkl'))
print('Pipeline saved to models/model_pipeline.pkl')
