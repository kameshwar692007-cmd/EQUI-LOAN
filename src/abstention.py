"""
abstention.py — EquiLoan Research Framework
Phase 10 / Step 11: Tri-State Decision-Confidence / Abstention Layer.
Instead of forcing binary decisions, classifies applicants as:
- High-confidence Approval (e.g. prob_deny < lower_thresh) -> APPROVE
- High-confidence Rejection (e.g. prob_deny > upper_thresh) -> REJECT
- Borderline / High Uncertainty -> HUMAN REVIEW
Analyzes Coverage, Accuracy at Coverage, Error Rate, and Review Percentage across threshold grids.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
try:
    from data_validation import DataValidator, get_dataset_path
    from preprocessing import create_preprocessing_pipeline
    from baseline_models import get_candidate_models
except ImportError:
    from src.data_validation import DataValidator, get_dataset_path
    from src.preprocessing import create_preprocessing_pipeline
    from src.baseline_models import get_candidate_models

def evaluate_abstention_layer(df: pd.DataFrame, model_name: str = 'XGBoost', lower_thresholds: list = [0.20, 0.30, 0.35, 0.40], upper_thresholds: list = [0.80, 0.70, 0.65, 0.60], drop_leakage: bool = True, random_state: int = 42):
    """
    Evaluates tri-state decisions on out-of-fold predictions.
    Computes Coverage, Accuracy@Coverage, Error Rate@Coverage, and Review %.
    """
    X = df.drop(columns=['status'], errors='ignore')
    y = df['status'].values
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    # Generate calibrated out-of-fold probabilities
    oof_probs = np.zeros(len(df))
    
    for train_idx, val_idx in skf.split(X, y):
        X_train_raw, X_val_raw = df.iloc[train_idx], df.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        preprocessor, _, _ = create_preprocessing_pipeline(X_train_raw, drop_leakage=drop_leakage, apply_iqr=True)
        X_train_proc = preprocessor.fit_transform(X_train_raw)
        X_val_proc = preprocessor.transform(X_val_raw)
        
        base_model = get_candidate_models(random_state=random_state)[model_name]
        cal_model = CalibratedClassifierCV(estimator=base_model, method='sigmoid', cv=3)
        cal_model.fit(X_train_proc, y_train)
        
        oof_probs[val_idx] = cal_model.predict_proba(X_val_proc)[:, 1] # P(status = 1 / Denied)
        
    results = []
    
    # Baseline 100% Coverage (Forced Binary at 0.5 threshold)
    forced_preds = (oof_probs >= 0.5).astype(int)
    forced_acc = np.mean(forced_preds == y)
    forced_err = 1.0 - forced_acc
    
    results.append({
        'Threshold Config': 'Forced Binary (0.50 cutoff)',
        'Lower Thresh (Approve)': 0.50,
        'Upper Thresh (Reject)': 0.50,
        'Automated Coverage %': 100.0,
        'Human Review %': 0.0,
        'Accuracy @ Coverage': round(forced_acc, 4),
        'Error Rate @ Coverage': round(forced_err, 4),
        'False Approval Rate': round(np.mean((forced_preds == 0) & (y == 1)), 4),
        'False Rejection Rate': round(np.mean((forced_preds == 1) & (y == 0)), 4)
    })
    
    for low, high in zip(lower_thresholds, upper_thresholds):
        # Tri-state decision logic
        decisions = []
        for p in oof_probs:
            if p <= low:
                decisions.append('APPROVE')
            elif p >= high:
                decisions.append('REJECT')
            else:
                decisions.append('HUMAN REVIEW')
                
        decisions = np.array(decisions)
        
        auto_mask = decisions != 'HUMAN REVIEW'
        coverage = np.mean(auto_mask) * 100
        review_pct = 100.0 - coverage
        
        if np.sum(auto_mask) > 0:
            auto_preds = np.where(decisions[auto_mask] == 'REJECT', 1, 0)
            auto_actuals = y[auto_mask]
            
            acc_at_cov = np.mean(auto_preds == auto_actuals)
            err_at_cov = 1.0 - acc_at_cov
            fa_rate = np.mean((auto_preds == 0) & (auto_actuals == 1))
            fr_rate = np.mean((auto_preds == 1) & (auto_actuals == 0))
        else:
            acc_at_cov, err_at_cov, fa_rate, fr_rate = np.nan, np.nan, np.nan, np.nan
            
        results.append({
            'Threshold Config': f"Band [{low:.2f}, {high:.2f}]",
            'Lower Thresh (Approve)': low,
            'Upper Thresh (Reject)': high,
            'Automated Coverage %': round(coverage, 2),
            'Human Review %': round(review_pct, 2),
            'Accuracy @ Coverage': round(acc_at_cov, 4),
            'Error Rate @ Coverage': round(err_at_cov, 4),
            'False Approval Rate': round(fa_rate, 4),
            'False Rejection Rate': round(fr_rate, 4)
        })
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    filepath = get_dataset_path()
    validator = DataValidator(filepath)
    df_nat, df_bal = validator.prepare_experimental_conditions()
    
    print("\n=======================================================")
    print("TRI-STATE ABSTENTION LAYER — NATURAL DISTRIBUTION")
    print("=======================================================")
    abs_nat_df = evaluate_abstention_layer(df_nat, model_name='XGBoost')
    print(abs_nat_df.to_string(index=False))
    
    print("\n=======================================================")
    print("TRI-STATE ABSTENTION LAYER — BALANCED CONDITION")
    print("=======================================================")
    abs_bal_df = evaluate_abstention_layer(df_bal, model_name='XGBoost')
    print(abs_bal_df.to_string(index=False))
    
    os.makedirs("results", exist_ok=True)
    abs_nat_df.to_csv("results/abstention_natural.csv", index=False)
    abs_bal_df.to_csv("results/abstention_balanced.csv", index=False)
