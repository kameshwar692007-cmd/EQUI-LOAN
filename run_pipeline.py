"""
run_pipeline.py — EquiLoan Master Execution Pipeline
======================================================
Executes the complete end-to-end EquiLoan framework across:
1. Data Integrity & Validation
2. Feature Preprocessing & Scaling
3. Baseline Model CV Benchmarks (Logistic Regression, Decision Tree, Random Forest, XGBoost, Naive Bayes)
4. Intersectional Demographic Fairness Audits
5. Probability Calibration (Platt & Isotonic)
6. Confidence-Gated Tri-State Abstention
7. Feature Stability & Robustness Analysis
"""

import sys
import os
import pandas as pd
import numpy as np

# Ensure project root and src/ are in Python path
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('src'))

from src.data_validation import DataValidator, get_dataset_path
from src.preprocessing import create_preprocessing_pipeline
from src.baseline_models import evaluate_baseline_cv
from src.fairness import audit_demographic_fairness
from src.calibration import benchmark_calibration
from src.abstention import evaluate_abstention_layer
from src.robustness import calculate_feature_stability

def main():
    print("=" * 70)
    print("      EQUI-LOAN: MASTER RESEARCH & PRODUCTION PIPELINE")
    print("=" * 70)
    
    # 1. Load and Validate Data
    dataset_path = get_dataset_path()
    print(f"\n[STEP 1/6] Loading & Validating Dataset: {dataset_path}")
    validator = DataValidator(dataset_path)
    health = validator.inspect_health()
    print(f"  - Total Applicants: {health['total_rows']:,}")
    print(f"  - Target Breakdown: {health['target_counts']}")
    print(f"  - Duplicates Found: {health['duplicates']}")
    
    df_nat, df_bal = validator.prepare_experimental_conditions()
    
    # 2. Baseline Models Cross-Validation
    print("\n[STEP 2/6] Benchmarking Baseline Classifiers (5-Fold Stratified CV)...")
    res_nat_clean = evaluate_baseline_cv(df_nat, drop_leakage=True, n_splits=5)
    res_bal_clean = evaluate_baseline_cv(df_bal, drop_leakage=True, n_splits=5)
    
    os.makedirs("results", exist_ok=True)
    res_nat_clean.to_csv("results/baseline_natural_clean.csv", index=False)
    res_bal_clean.to_csv("results/baseline_balanced_clean.csv", index=False)
    print("  ✓ Baseline results saved to results/baseline_natural_clean.csv & baseline_balanced_clean.csv")
    
    # 3. Demographic Fairness Audit
    print("\n[STEP 3/6] Auditing Demographic Fairness across Sensitive Groups...")
    fair_nat_df, _ = audit_demographic_fairness(df_nat, model_name='XGBoost', sensitive_cols=['gender', 'region', 'age'])
    fair_bal_df, _ = audit_demographic_fairness(df_bal, model_name='XGBoost', sensitive_cols=['gender', 'region', 'age'])
    
    fair_nat_df.to_csv("results/fairness_audit_natural.csv", index=False)
    fair_bal_df.to_csv("results/fairness_audit_balanced.csv", index=False)
    print("  ✓ Fairness audit saved to results/fairness_audit_natural.csv & fairness_audit_balanced.csv")
    
    # 4. Probability Calibration
    print("\n[STEP 4/6] Benchmarking Probability Calibration (Platt & Isotonic)...")
    cal_nat_df = benchmark_calibration(df_nat, model_name='XGBoost')
    cal_bal_df = benchmark_calibration(df_bal, model_name='XGBoost')
    
    cal_nat_df.to_csv("results/calibration_natural.csv", index=False)
    cal_bal_df.to_csv("results/calibration_balanced.csv", index=False)
    print("  ✓ Calibration results saved to results/calibration_natural.csv & calibration_balanced.csv")
    
    # 5. Tri-State Abstention Layer
    print("\n[STEP 5/6] Evaluating Confidence-Gated Tri-State Abstention Router...")
    abs_nat_df = evaluate_abstention_layer(df_nat, model_name='XGBoost')
    abs_bal_df = evaluate_abstention_layer(df_bal, model_name='XGBoost')
    
    abs_nat_df.to_csv("results/abstention_natural.csv", index=False)
    abs_bal_df.to_csv("results/abstention_balanced.csv", index=False)
    print("  ✓ Abstention metrics saved to results/abstention_natural.csv & abstention_balanced.csv")
    
    # 6. Feature Stability & Robustness
    print("\n[STEP 6/6] Analyzing Feature Importance Stability across Folds...")
    stab_nat_metrics, stab_nat_df = calculate_feature_stability(df_nat, model_name='XGBoost')
    stab_bal_metrics, stab_bal_df = calculate_feature_stability(df_bal, model_name='XGBoost')
    
    stab_nat_df.to_csv("results/feature_stability_natural.csv", index=False)
    stab_bal_df.to_csv("results/feature_stability_balanced.csv", index=False)
    print("  ✓ Feature stability saved to results/feature_stability_natural.csv & feature_stability_balanced.csv")
    
    print("\n" + "=" * 70)
    print("  ✓ EQUI-LOAN MASTER PIPELINE EXECUTION COMPLETE!")
    print("  All benchmarks, scorecards, and audit logs are ready in results/")
    print("=" * 70)

if __name__ == "__main__":
    main()
