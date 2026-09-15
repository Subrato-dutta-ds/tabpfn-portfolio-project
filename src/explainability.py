import os, joblib, shap, matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.config import BASE_DIR

# Load the PRODUCTION model (calibrated) - not model_pipeline.pkl
production = joblib.load(os.path.join(BASE_DIR, 'models', 'production_model.pkl'))

# Extract the underlying pipeline from the calibrated wrapper
if hasattr(production, 'calibrated_classifiers_'):
    base_pipeline = production.calibrated_classifiers_[0].estimator
    print('Detected CalibratedClassifierCV — using first fold estimator for SHAP')
else:
    base_pipeline = production

preprocessor = base_pipeline.named_steps['preprocessor']
model = base_pipeline.named_steps['classifier']

X_test = pd.read_csv(os.path.join(BASE_DIR, 'reports', 'X_test.csv'))
X_transformed = preprocessor.transform(X_test)
feature_names = preprocessor.get_feature_names_out()

print(f'Explaining: {type(model).__name__}')
print(f'Features: {len(feature_names)}')

if hasattr(model, 'feature_importances_'):
    explainer = shap.TreeExplainer(model)
else:
    explainer = shap.LinearExplainer(model, X_transformed)

shap_values = explainer.shap_values(X_transformed)

# Summary plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
plt.savefig(os.path.join(BASE_DIR, 'reports', 'shap_summary.png'), bbox_inches='tight', dpi=100)
plt.close()

# Feature importance CSV
mean_abs = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    'feature': feature_names,
    'mean_abs_shap': mean_abs
}).sort_values('mean_abs_shap', ascending=False)
importance_df.to_csv(os.path.join(BASE_DIR, 'reports', 'feature_importance.csv'), index=False)

print('SHAP summary saved to reports/shap_summary.png')
print('Feature importance saved to reports/feature_importance.csv')
print(f'Top 5: {importance_df.head(5)["feature"].tolist()}')
print()
print('NOTE: SHAP explains the underlying base estimator.')
print('      Displayed probabilities come from the calibrated production model.')
