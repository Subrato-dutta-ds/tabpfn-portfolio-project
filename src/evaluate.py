"""
Evaluation script with latency benchmarking.
Measures accuracy, F1, ROC-AUC, and inference time per 1000 rows.
Saves results to reports/performance_table.csv
"""
import pandas as pd
import numpy as np
import joblib
import time
import os
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

MODELS_DIR = "models"
PROCESSED_DIR = "data/processed"
REPORTS_DIR = "reports"

def evaluate():
    print("=" * 60)
    print("STEP 3: Model Evaluation with Latency Benchmarking")
    print("=" * 60)
    
    # Load test data
    X_test = np.load(f"{PROCESSED_DIR}/X_test.npy")
    y_test = np.load(f"{PROCESSED_DIR}/y_test.npy")
    print(f"Test set: {len(X_test)} rows, {X_test.shape[1]} features")
    
    # Load models
    models = {}
    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "Random Forest": "random_forest.pkl",
        "XGBoost": "xgboost.pkl",
        # "TabPFN": "tabpfn_model.pkl"  # Add later
    }
    
    for name, filename in model_files.items():
        path = f"{MODELS_DIR}/{filename}"
        if os.path.exists(path):
            models[name] = joblib.load(path)
            print(f"✅ Loaded: {name}")
        else:
            print(f"⚠️ Skipping: {name} (file not found)")
    
    if not models:
        print("❌ No models found!")
        return
    
    # Use subset for latency measurement (1000 rows)
    latency_sample = X_test[:1000]
    if len(latency_sample) < 1000:
        latency_sample = X_test
        print(f"⚠️ Test set has only {len(latency_sample)} rows; using full set for latency.")
    
    results = []
    print("\n" + "-" * 60)
    print("Evaluating models...")
    print("-" * 60)
    
    for name, model in models.items():
        print(f"\n📊 {name}")
        
        # --- Accuracy metrics ---
        try:
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            # ROC-AUC
            try:
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test)[:, 1]
                else:
                    y_proba = y_pred
                roc = roc_auc_score(y_test, y_proba)
            except:
                roc = 0.0
            
            print(f"  Accuracy:  {acc:.4f}")
            print(f"  F1 Score:  {f1:.4f}")
            print(f"  ROC-AUC:   {roc:.4f}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
        
        # --- Inference latency benchmark ---
        try:
            # Warm-up
            _ = model.predict(latency_sample[:1])
            # Measure
            start_time = time.perf_counter()
            _ = model.predict(latency_sample)
            elapsed = time.perf_counter() - start_time
            latency_per_1000 = elapsed * (1000 / len(latency_sample))
            print(f"  Latency:   {latency_per_1000:.4f} s per 1000 rows")
            print(f"  Throughput:{1000/latency_per_1000:.1f} rows/s")
        except Exception as e:
            print(f"  ❌ Latency measurement failed: {e}")
            latency_per_1000 = -1
        
        results.append([name, acc, f1, roc, latency_per_1000])
    
    # Save results
    os.makedirs(REPORTS_DIR, exist_ok=True)
    df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1", "ROC-AUC", "Latency (s per 1000 rows)"])
    df.to_csv(f"{REPORTS_DIR}/performance_table.csv", index=False)
    
    print("\n" + "=" * 60)
    print("📊 FINAL PERFORMANCE COMPARISON")
    print("=" * 60)
    print(df.to_string(index=False))
    
    if len(df) > 0:
        best_acc = df.loc[df['Accuracy'].idxmax()]
        best_f1 = df.loc[df['F1'].idxmax()]
        fastest = df.loc[df['Latency (s per 1000 rows)'].idxmin()]
        print("\n🏆 Summary:")
        print(f"  Best Accuracy:  {best_acc['Model']} ({best_acc['Accuracy']:.4f})")
        print(f"  Best F1 Score:  {best_f1['Model']} ({best_f1['F1']:.4f})")
        print(f"  Fastest:        {fastest['Model']} ({fastest['Latency (s per 1000 rows)']:.4f} s/1000)")
    
    print(f"\n📁 Results saved to: {REPORTS_DIR}/performance_table.csv")

if __name__ == "__main__":
    evaluate()
