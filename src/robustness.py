"""
robustness.py — EquiLoan Research Framework
Phase 14 / Step 16: Feature Stability & Robustness Analysis across Folds.
Computes cross-fold rank correlation (Spearman rho) and Top-10 feature overlap Jaccard index
to verify feature importance stability across Stratified K-Fold CV.
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold

from data_validation import DataValidator
from preprocessing import create_preprocessing_pipeline
from baseline_models import get_candidate_models

def calculate_feature_stability(df: pd.DataFrame, model_name: str = 'XGBoost', n_splits: int = 5, drop_leakage: bool = True, random_state: int = 42):
    """
    Computes feature importance rankings across K folds and measures stability via:
    1. Spearman Rank Correlation (mean pairwise correlation across folds)
    2. Top-10 Jaccard Overlap Index
    """
    X = df.drop(columns=['status'], errors='ignore')
    y = df['status'].values
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_importances = []
    feature_names = None
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_raw = df.iloc[train_idx]
        y_train = y[train_idx]
        
        preprocessor, num_cols, cat_cols = create_preprocessing_pipeline(X_train_raw, drop_leakage=drop_leakage, apply_iqr=True)
        X_train_proc = preprocessor.fit_transform(X_train_raw)
        
        # Get feature names after one-hot encoding
        if feature_names is None:
            cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
            cat_feature_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
            feature_names = num_cols + cat_feature_names
            
        model = get_candidate_models(random_state=random_state)[model_name]
        model.fit(X_train_proc, y_train)
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            continue
            
        fold_importances.append(pd.Series(importances, index=feature_names))
        
    df_imp = pd.DataFrame(fold_importances)
    
    # 1. Spearman Rank Correlation matrix across folds
    spearman_corrs = []
    top10_jaccards = []
    
    n_folds = len(fold_importances)
    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            # Spearman
            rho, _ = spearmanr(df_imp.iloc[i], df_imp.iloc[j])
            spearman_corrs.append(rho)
            
            # Top-10 Jaccard
            top_i = set(df_imp.iloc[i].nlargest(10).index)
            top_j = set(df_imp.iloc[j].nlargest(10).index)
            jaccard = len(top_i.intersection(top_j)) / len(top_i.union(top_j))
            top10_jaccards.append(jaccard)
            
    mean_imp = df_imp.mean(axis=0).sort_values(ascending=False)
    std_imp = df_imp.std(axis=0)
    
    stability_summary = pd.DataFrame({
        'Feature': mean_imp.index,
        'Mean Importance': mean_imp.values.round(5),
        'Std Importance': std_imp[mean_imp.index].values.round(5),
        'Coefficient of Variation (CV)': (std_imp[mean_imp.index] / mean_imp).values.round(4)
    })
    
    metrics = {
        'Model Architecture': model_name,
        'Mean Spearman Rank Correlation across Folds': round(np.mean(spearman_corrs), 4),
        'Mean Top-10 Jaccard Overlap across Folds': round(np.mean(top10_jaccards), 4),
        'Top 5 Consistently Important Features': ", ".join(mean_imp.head(5).index.tolist())
    }
    
    return metrics, stability_summary

if __name__ == "__main__":
    filepath = os.path.join("dataset which u should use for training and testing", "loan_v2 (real distribution).csv")
    validator = DataValidator(filepath)
    df_nat, df_bal = validator.prepare_experimental_conditions()
    
    print("\n=======================================================")
    print("FEATURE STABILITY ANALYSIS — NATURAL DISTRIBUTION")
    print("=======================================================")
    stab_nat_metrics, stab_nat_df = calculate_feature_stability(df_nat, model_name='XGBoost')
    print("Stability Metrics:", stab_nat_metrics)
    print("\nTop 10 Feature Stability:")
    print(stab_nat_df.head(10).to_string(index=False))
    
    print("\n=======================================================")
    print("FEATURE STABILITY ANALYSIS — BALANCED CONDITION")
    print("=======================================================")
    stab_bal_metrics, stab_bal_df = calculate_feature_stability(df_bal, model_name='XGBoost')
    print("Stability Metrics:", stab_bal_metrics)
    print("\nTop 10 Feature Stability:")
    print(stab_bal_df.head(10).to_string(index=False))
    
    os.makedirs("results", exist_ok=True)
    stab_nat_df.to_csv("results/feature_stability_natural.csv", index=False)
    stab_bal_df.to_csv("results/feature_stability_balanced.csv", index=False)
