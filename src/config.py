import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'bank-additional-full.csv')

def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE)
    }

def get_param_grids():
    return {
        "Random Forest": {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [None, 10, 20]},
        "XGBoost": {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [3, 5, 7], 'classifier__learning_rate': [0.01, 0.1]}
    }
