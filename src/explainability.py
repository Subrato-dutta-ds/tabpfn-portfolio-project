"""
Explainability script using SHAP.
Generates global feature importance and individual explanations.
Caches results for use in the API later.
"""
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

MODELS_DIR = "models"
PROCESSED_DIR = "data/processed"
REPORTS_DIR = "reports"

def load_data():
    """Load test data and feature names."""
    X_test = np.load(f"{PROCESSED_DIR}/X_test.npy")
    y_test = np.load(f"{PROCESSED_DIR}/y_test.npy")
    with open(f"{PROCESSED_DIR}/feature_names.txt", 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    return X_test, y_test, feature_names

def load_best_model():
    """Load the best performing model (XGBoost by default)."""
    # Try XGBoost first, fallback to Random Forest, then Logistic Regression
    model_paths = [
        ("XGBoost", f"{MODELS_DIR}/xgboost.pkl"),
        ("Random Forest", f"{MODELS_DIR}/random_forest.pkl"),
        ("Logistic Regression", f"{MODELS_DIR}/logistic_regression.pkl")
    ]
    
    for name, path in model_paths:
        if os.path.exists(path):
            print(f"✅ Loading model: {name}")
            return joblib.load(path), name
    
    raise FileNotFoundError("No models found!")

def compute_shap():
    """Compute SHAP values and generate plots."""
    print("=" * 60)
    print("STEP 4: SHAP Explainability Analysis")
    print("=" * 60)
    
    # Load data
    X_test, y_test, feature_names = load_data()
    print(f"Test data: {X_test.shape[0]} rows, {X_test.shape[1]} features")
    
    # Load best model
    model, model_name = load_best_model()
    
    # Create SHAP explainer based on model type
    print(f"\nCreating SHAP explainer for {model_name}...")
    
    if model_name == "XGBoost":
        # XGBoost uses TreeExplainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    elif model_name == "Random Forest":
        # Random Forest uses TreeExplainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    elif model_name == "Logistic Regression":
        # Logistic Regression uses LinearExplainer
        explainer = shap.LinearExplainer(model, X_test)
        shap_values = explainer.shap_values(X_test)
    else:
        print(f"⚠️ Unsupported model type: {model_name}")
        return
    
    print("✅ SHAP values computed successfully!")
    
    # Create reports directory
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # --- 1. Global Feature Importance (Summary Plot) ---
    print("\n📊 Generating global feature importance...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/shap_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {REPORTS_DIR}/shap_summary.png")
    
    # --- 2. Feature Importance Bar Chart (Mean |SHAP|) ---
    print("\n📊 Generating feature importance bar chart...")
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean |SHAP|": mean_abs_shap
    }).sort_values("Mean |SHAP|", ascending=False)
    
    # Save to CSV
    importance_df.to_csv(f"{REPORTS_DIR}/feature_importance.csv", index=False)
    
    # Plot top 15 features
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(15)
    plt.barh(top_features["Feature"], top_features["Mean |SHAP|"])
    plt.xlabel("Mean |SHAP Value|")
    plt.title(f"Top 15 Feature Importances ({model_name})")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {REPORTS_DIR}/feature_importance.png")
    
    # --- 3. Individual Explanation (Waterfall Plot for one sample) ---
    print("\n📊 Generating individual explanation (waterfall plot)...")
    # Pick a random sample (index 42)
    sample_idx = 42
    sample_idx = min(sample_idx, len(X_test) - 1)
    
    plt.figure(figsize=(12, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[sample_idx],
            base_values=explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
            data=X_test[sample_idx],
            feature_names=feature_names
        ),
        show=False
    )
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/shap_waterfall_sample.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {REPORTS_DIR}/shap_waterfall_sample.png")
    
    # --- 4. Cache SHAP values for API ---
    print("\n💾 Caching SHAP values for API...")
    joblib.dump(explainer, f"{MODELS_DIR}/shap_explainer.pkl")
    joblib.dump(shap_values, f"{MODELS_DIR}/shap_values.pkl")
    joblib.dump(feature_names, f"{MODELS_DIR}/feature_names.pkl")
    print(f"  ✅ Cached to: {MODELS_DIR}/shap_explainer.pkl")
    
    # --- Summary ---
    print("\n" + "=" * 60)
    print("✅ SHAP Analysis Complete!")
    print("=" * 60)
    print(f"\n📁 Reports saved to: {REPORTS_DIR}/")
    print("  - shap_summary.png (global feature importance)")
    print("  - feature_importance.png (bar chart)")
    print("  - shap_waterfall_sample.png (individual explanation)")
    print("  - feature_importance.csv (data)")
    print(f"\n📁 Cached SHAP values saved to: {MODELS_DIR}/")

if __name__ == "__main__":
    compute_shap()
