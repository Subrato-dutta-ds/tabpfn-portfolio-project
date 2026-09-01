import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- 1. DYNAMIC PATH SETUP (Fixes reproducibility issue) ---
# Get the absolute path to the project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'bank-additional-full.csv')

# Load the Bank Marketing dataset (Uses semicolon separator)
df = pd.read_csv(DATA_PATH, sep=';')

# Target column is 'y'
X = df.drop('y', axis=1)
y = df['y']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define column types
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(exclude=['object']).columns

# --- 2. PREPROCESSING PIPELINE ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Create the model pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# --- 3. TRAIN AND SAVE ---
pipeline.fit(X_train, y_train)

# Print validation score
print(f"Model trained. Accuracy on test set: {pipeline.score(X_test, y_test):.4f}")

# Make sure the 'models' folder exists
os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)

# Save the ENTIRE pipeline (scaler + encoder + model)
joblib.dump(pipeline, os.path.join(BASE_DIR, 'models', 'model_pipeline.pkl'))
print("Pipeline saved to models/model_pipeline.pkl")