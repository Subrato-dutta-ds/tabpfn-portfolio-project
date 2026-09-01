import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- 1. DYNAMIC PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'bank-additional-full.csv')

# --- 2. LOAD AND PREPROCESS DATA (LEAKAGE FIX) ---
df = pd.read_csv(DATA_PATH, sep=';')

# CRITICAL FIX: Drop 'duration' to prevent data leakage!
# (You did this in data_loader.py, now we do it here too)
df = df.drop(columns=['duration'], errors='ignore')

# Target column is 'y'
X = df.drop('y', axis=1)
y = df['y']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define column types
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(exclude=['object']).columns

# Create the model pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Train and Save
pipeline.fit(X_train, y_train)
print(f"Model trained. Accuracy: {pipeline.score(X_test, y_test):.4f}")

# Make sure the 'models' folder exists
os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)

# Save the ENTIRE pipeline
joblib.dump(pipeline, os.path.join(BASE_DIR, 'models', 'model_pipeline.pkl'))
print("Pipeline saved to models/model_pipeline.pkl")