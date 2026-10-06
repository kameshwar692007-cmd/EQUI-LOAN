"""
fairness.py — EquiLoan Research Framework
Phase 8 / Step 8: Demographic Bias Auditing & Fairness Analysis.
Evaluates Statistical Parity Difference (SPD) and Disparate Impact Ratio (DIR)
across sensitive / proxy demographic groups (`gender`, `region`, `age`).
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from data_validation import DataValidator
from preprocessing import create_preprocessing_pipeline
from baseline_models import get_candidate_models

def audit_demographic_fairness(df: pd.DataFrame, model_name: str = 'XGBoost', sensitive_cols: list = ['gender', 'region', 'age'], drop_leakage: bool = True, random_state: int = 42):
    """
    Audits model predictions for demographic disparities across sensitive columns.
    Target status: 0 = Approved, 1 = Rejected.
    Fitted via Stratified 5-Fold Cross-Validation.
    """
    X = df.drop(columns=['status'], errors='ignore')
    y = df['status'].values
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    # Store predictions along with sensitive attributes
    df_results = df.copy()
    df_results['predicted_status'] = np.nan
    df_results['predicted_prob_approved'] = np.nan
    
    for train_idx, val_idx in skf.split(X, y):
        X_train_raw, X_val_raw = df.iloc[train_idx], df.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        preprocessor, _, _ = create_preprocessing_pipeline(X_train_raw, drop_leakage=drop_leakage, apply_iqr=True)
        X_train_proc = preprocessor.fit_transform(X_train_raw)
        X_val_proc = preprocessor.transform(X_val_raw)
        
        model = get_candidate_models(random_state=random_state)[model_name]
        model.fit(X_train_proc, y_train)
        
        preds = model.predict(X_val_proc)
        probs = model.predict_proba(X_val_proc)[:, 0] # probability of status 0 (Approved)
        
        df_results.iloc[val_idx, df_results.columns.get_loc('predicted_status')] = preds
        df_results.iloc[val_idx, df_results.columns.get_loc('predicted_prob_approved')] = probs
        
    fairness_metrics = []
    
    for col in sensitive_cols:
        if col not in df_results.columns:
            continue
            
        group_rates = df_results.groupby(col)['predicted_status'].apply(lambda s: (s == 0).mean())
        
        if len(group_rates) >= 2:
            max_rate = group_rates.max()
            min_rate = group_rates.min()
            max_group = group_rates.idxmax()
            min_group = group_rates.idxmin()
            
            spd = round(max_rate - min_rate, 4)
            dir_ratio = round(min_rate / max_rate, 4) if max_rate > 0 else np.nan
            
            adverse_impact_flag = "ALERT (DIR < 0.8)" if (not np.isnan(dir_ratio) and dir_ratio < 0.8) else "PASS (DIR >= 0.8)"
            
            fairness_metrics.append({
                'Demographic Attribute': col,
                'Privileged Group (Max Rate)': f"{max_group} ({max_rate:.1%})",
                'Unprivileged Group (Min Rate)': f"{min_group} ({min_rate:.1%})",
                'Statistical Parity Diff (SPD)': spd,
                'Disparate Impact Ratio (DIR)': dir_ratio,
                'Adverse Impact Screening': adverse_impact_flag
            })
            
    return pd.DataFrame(fairness_metrics), df_results

if __name__ == "__main__":
    filepath = os.path.join("dataset which u should use for training and testing", "loan_v2 (real distribution).csv")
    validator = DataValidator(filepath)
    df_nat, df_bal = validator.prepare_experimental_conditions()
    
    print("\n=======================================================")
    print("FAIRNESS AUDIT — NATURAL DISTRIBUTION (XGBoost Baseline)")
    print("=======================================================")
    fair_nat_df, _ = audit_demographic_fairness(df_nat, model_name='XGBoost', sensitive_cols=['gender', 'region', 'age'])
    print(fair_nat_df.to_string(index=False))
    
    print("\n=======================================================")
    print("FAIRNESS AUDIT — BALANCED CONDITION (XGBoost Baseline)")
    print("=======================================================")
    fair_bal_df, _ = audit_demographic_fairness(df_bal, model_name='XGBoost', sensitive_cols=['gender', 'region', 'age'])
    print(fair_bal_df.to_string(index=False))
    
    os.makedirs("results", exist_ok=True)
    fair_nat_df.to_csv("results/fairness_audit_natural.csv", index=False)
    fair_bal_df.to_csv("results/fairness_audit_balanced.csv", index=False)
