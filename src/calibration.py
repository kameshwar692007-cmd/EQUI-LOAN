"""
calibration.py — EquiLoan Research Framework
Phase 9 / Step 10: Probability Calibration & Confidence Reliability.
Evaluates Brier Score, Log Loss, and Expected Calibration Error (ECE)
comparing Uncalibrated vs. Platt Scaled (Sigmoidal) vs. Isotonic Calibrated models.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

try:
    from data_validation import DataValidator, get_dataset_path
    from preprocessing import create_preprocessing_pipeline
    from baseline_models import get_candidate_models
except ImportError:
    from src.data_validation import DataValidator, get_dataset_path
    from src.preprocessing import create_preprocessing_pipeline
    from src.baseline_models import get_candidate_models

def calculate_ece(y_true, y_prob, n_bins=10):
    """Calculates Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
            
    return ece

def benchmark_calibration(df: pd.DataFrame, model_name: str = 'XGBoost', drop_leakage: bool = True, random_state: int = 42):
    """
    Evaluates probability calibration on out-of-fold predictions using 5-fold CV.
    Compares:
    - Uncalibrated Base Model
    - Platt Scaling (Sigmoid)
    - Isotonic Regression
    """
    X = df.drop(columns=['status'], errors='ignore')
    y = df['status'].values
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    methods = ['Uncalibrated', 'Platt Scaling (Sigmoid)', 'Isotonic Regression']
    results = []
    
    for method in methods:
        fold_brier, fold_logloss, fold_ece, fold_auc = [], [], [], []
        
        for train_idx, val_idx in skf.split(X, y):
            X_train_raw, X_val_raw = df.iloc[train_idx], df.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            preprocessor, _, _ = create_preprocessing_pipeline(X_train_raw, drop_leakage=drop_leakage, apply_iqr=True)
            X_train_proc = preprocessor.fit_transform(X_train_raw)
            X_val_proc = preprocessor.transform(X_val_raw)
            
            base_model = get_candidate_models(random_state=random_state)[model_name]
            
            if method == 'Uncalibrated':
                cal_model = base_model
                cal_model.fit(X_train_proc, y_train)
            elif method == 'Platt Scaling (Sigmoid)':
                cal_model = CalibratedClassifierCV(estimator=base_model, method='sigmoid', cv=3)
                cal_model.fit(X_train_proc, y_train)
            elif method == 'Isotonic Regression':
                cal_model = CalibratedClassifierCV(estimator=base_model, method='isotonic', cv=3)
                cal_model.fit(X_train_proc, y_train)
                
            probs = cal_model.predict_proba(X_val_proc)[:, 1] # Probability of status 1 (Denied)
            
            fold_brier.append(brier_score_loss(y_val, probs))
            fold_logloss.append(log_loss(y_val, probs))
            fold_ece.append(calculate_ece(y_val, probs))
            fold_auc.append(roc_auc_score(y_val, probs))
            
        results.append({
            'Model Architecture': model_name,
            'Calibration Method': method,
            'Brier Score (lower=better)': f"{np.mean(fold_brier):.4f} +/- {np.std(fold_brier):.4f}",
            'Log Loss (lower=better)': f"{np.mean(fold_logloss):.4f} +/- {np.std(fold_logloss):.4f}",
            'ECE (lower=better)': f"{np.mean(fold_ece):.4f} +/- {np.std(fold_ece):.4f}",
            'ROC AUC': f"{np.mean(fold_auc):.4f} +/- {np.std(fold_auc):.4f}"
        })
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    filepath = get_dataset_path()
    validator = DataValidator(filepath)
    df_nat, df_bal = validator.prepare_experimental_conditions()
    
    print("\n=======================================================")
    print("PROBABILITY CALIBRATION BENCHMARK — NATURAL DISTRIBUTION")
    print("=======================================================")
    cal_nat_df = benchmark_calibration(df_nat, model_name='XGBoost')
    print(cal_nat_df.to_string(index=False))
    
    print("\n=======================================================")
    print("PROBABILITY CALIBRATION BENCHMARK — BALANCED CONDITION")
    print("=======================================================")
    cal_bal_df = benchmark_calibration(df_bal, model_name='XGBoost')
    print(cal_bal_df.to_string(index=False))
    
    os.makedirs("results", exist_ok=True)
    cal_nat_df.to_csv("results/calibration_natural.csv", index=False)
    cal_bal_df.to_csv("results/calibration_balanced.csv", index=False)
