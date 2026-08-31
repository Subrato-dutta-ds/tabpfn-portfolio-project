"""
Train all 4 models: Logistic Regression, Random Forest, XGBoost, and TabPFN.
Saves trained models to the models/ directory.
"""
import pandas as pd
import numpy as np
import joblib
import os
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Try importing TabPFN with version handling
TABPFN_AVAILABLE = False
try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
    print("✅ TabPFN imported successfully!")
except ImportError as e:
    print(f"⚠️ TabPFN import error: {e}")
    print("   Will skip TabPFN training.")

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
REPORTS_DIR = "reports"

def load_processed_data():
    """Load the preprocessed data from disk."""
    X_train = np.load(f"{PROCESSED_DIR}/X_train.npy")
    X_test = np.load(f"{PROCESSED_DIR}/X_test.npy")
    y_train = np.load(f"{PROCESSED_DIR}/y_train.npy")
    y_test = np.load(f"{PROCESSED_DIR}/y_test.npy")
    return X_train, X_test, y_train, y_test

def train_and_save_models():
    """Train all models and save them."""
    print("=" * 60)
    print("STEP 2: Training All Models")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = load_processed_data()
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Class distribution in training: {np.bincount(y_train)}")
    
    results = []
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # --- 1. Logistic Regression ---
    print("\n" + "-" * 40)
    print("Training Logistic Regression...")
    start = time.time()
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    train_time = time.time() - start
    joblib.dump(lr, f"{MODELS_DIR}/logistic_regression.pkl")
    
    y_pred = lr.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    try:
        roc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
    except:
        roc = 0.0
    results.append(["Logistic Regression", acc, f1, roc, train_time])
    print(f"  ✅ Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc:.4f}")
    print(f"  ⏱️  Training time: {train_time:.2f}s")
    
    # --- 2. Random Forest ---
    print("\n" + "-" * 40)
    print("Training Random Forest...")
    start = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    train_time = time.time() - start
    joblib.dump(rf, f"{MODELS_DIR}/random_forest.pkl")
    
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    try:
        roc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    except:
        roc = 0.0
    results.append(["Random Forest", acc, f1, roc, train_time])
    print(f"  ✅ Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc:.4f}")
    print(f"  ⏱️  Training time: {train_time:.2f}s")
    
    # --- 3. XGBoost ---
    print("\n" + "-" * 40)
    print("Training XGBoost...")
    start = time.time()
    xgb = XGBClassifier(
        n_estimators=100,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0
    )
    xgb.fit(X_train, y_train)
    train_time = time.time() - start
    joblib.dump(xgb, f"{MODELS_DIR}/xgboost.pkl")
    
    y_pred = xgb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    try:
        roc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
    except:
        roc = 0.0
    results.append(["XGBoost", acc, f1, roc, train_time])
    print(f"  ✅ Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc:.4f}")
    print(f"  ⏱️  Training time: {train_time:.2f}s")
    
    # --- 4. TabPFN (with better error handling) ---
    if TABPFN_AVAILABLE:
        print("\n" + "-" * 40)
        print("Training TabPFN...")
        print("  (This may take 3-8 minutes depending on your CPU)")
        try:
            start = time.time()
            tabpfn = TabPFNClassifier(
                device='cpu',
                n_estimators=4,
                random_state=42
            )
            tabpfn.fit(X_train, y_train)
            train_time = time.time() - start
            joblib.dump(tabpfn, f"{MODELS_DIR}/tabpfn_model.pkl")
            
            y_pred = tabpfn.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            try:
                roc = roc_auc_score(y_test, tabpfn.predict_proba(X_test)[:, 1])
            except:
                roc = 0.0
            results.append(["TabPFN", acc, f1, roc, train_time])
            print(f"  ✅ Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc:.4f}")
            print(f"  ⏱️  Training time: {train_time:.2f}s")
        except Exception as e:
            print(f"  ❌ TabPFN training failed: {e}")
            print("  ⚠️ TabPFN will be skipped.")
    else:
        print("\n" + "-" * 40)
        print("⚠️ Skipping TabPFN (not installed)")
    
    # Save results
    results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1", "ROC-AUC", "Training Time (s)"])
    results_df.to_csv(f"{REPORTS_DIR}/training_results.csv", index=False)
    
    print("\n" + "=" * 60)
    print("✅ All models trained and saved!")
    print("=" * 60)
    print("\n📊 Training Results Summary:")
    print(results_df.to_string(index=False))
    print("\n📁 Models saved to: models/")
    print("📁 Results saved to: reports/training_results.csv")

if __name__ == "__main__":
    train_and_save_models()
