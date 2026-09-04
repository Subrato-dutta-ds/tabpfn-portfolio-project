import os, joblib, shap, matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pipeline = joblib.load(os.path.join(BASE_DIR, 'models', 'model_pipeline.pkl'))

X_test = pd.read_csv(os.path.join(BASE_DIR, 'reports', 'X_test.csv'))

X_transformed = pipeline.named_steps['preprocessor'].transform(X_test)
feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
model = pipeline.named_steps['classifier']

if hasattr(model, 'feature_importances_'):
    explainer = shap.TreeExplainer(model)
else:
    explainer = shap.LinearExplainer(model, X_transformed)

shap_values = explainer.shap_values(X_transformed)
plt.figure()
shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
plt.savefig(os.path.join(BASE_DIR, 'reports', 'shap_summary.png'), bbox_inches='tight')
print("SHAP summary generated on exact test split.")
