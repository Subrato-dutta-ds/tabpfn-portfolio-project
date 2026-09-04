import os, joblib, pandas as pd, shap, matplotlib.pyplot as plt
from src.data_loader import load_data
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pipeline = joblib.load(os.path.join(BASE_DIR, 'models', 'model_pipeline.pkl'))
df = load_data()
X = df.drop('y', axis=1)

# Use raw data and transform it through the canonical pipeline
_, X_test = train_test_split(X, test_size=0.2, random_state=42)
X_transformed = pipeline.named_steps['preprocessor'].transform(X_test)

feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
explainer = shap.Explainer(pipeline.named_steps['classifier'], X_transformed)
shap_values = explainer(X_transformed)

# Save image
plt.figure()
shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
plt.savefig(os.path.join(BASE_DIR, 'reports', 'shap_summary.png'), bbox_inches='tight')
print("SHAP summary saved.")
