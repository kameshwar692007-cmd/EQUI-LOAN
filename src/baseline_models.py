"""
baseline_models.py — EquiLoan Research Framework
Phase 6: Baseline Model Benchmark across:
- Dual Experimental Conditions (Natural Distribution vs. Balanced Condition)
- Dual Leakage Settings (With Post-Underwriting Rate Leakage vs. Leakage Pruned)
- Stratified 5-Fold Cross-Validation
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss

try:
    from data_validation import DataValidator, get_dataset_path
    from preprocessing import create_preprocessing_pipeline
except ImportError:
    from src.data_validation import DataValidator, get_dataset_path
    from src.preprocessing import create_preprocessing_pipeline

def get_candidate_models(random_state=42):
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state),
        'Decision Tree': DecisionTreeClassifier(random_state=random_state, max_depth=10),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=random_state, eval_metric='logloss', verbosity=0, n_jobs=-1),
        'Naive Bayes': GaussianNB()
    }

def evaluate_baseline_cv(df: pd.DataFrame, drop_leakage: bool = True, n_splits: int = 5, random_state: int = 42):
    """
    Evaluates baseline models using strict leakage-free Stratified K-Fold CV.
    Preprocessing pipeline is fitted inside each fold.
    """
    X = df.drop(columns=['status'], errors='ignore')
    y = df['status'].values
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    models = get_candidate_models(random_state=random_state)
    
    results = []
    
    for name, model in models.items():
        print(f"  [CV BENCHMARK] Evaluating {name} (drop_leakage={drop_leakage})...")
        fold_accs, fold_prec, fold_rec, fold_f1, fold_auc, fold_brier = [], [], [], [], [], []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train_raw, X_val_raw = df.iloc[train_idx], df.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Build & fit preprocessing pipeline on TRAIN fold only
            preprocessor, _, _ = create_preprocessing_pipeline(X_train_raw, drop_leakage=drop_leakage, apply_iqr=True)
            
            X_train_proc = preprocessor.fit_transform(X_train_raw)
            X_val_proc = preprocessor.transform(X_val_raw)
            
            # Fit model on transformed train fold
            model.fit(X_train_proc, y_train)
            
            # Predict on val fold
            preds = model.predict(X_val_proc)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_val_proc)[:, 1]
            else:
                probs = preds
                
            fold_accs.append(accuracy_score(y_val, preds))
            fold_prec.append(precision_score(y_val, preds, zero_division=0))
            fold_rec.append(recall_score(y_val, preds, zero_division=0))
            fold_f1.append(f1_score(y_val, preds, zero_division=0))
            fold_auc.append(roc_auc_score(y_val, probs))
            fold_brier.append(brier_score_loss(y_val, probs))
            
        results.append({
            'Model': name,
            'Drop Leakage': drop_leakage,
            'Accuracy': f"{np.mean(fold_accs):.4f} +/- {np.std(fold_accs):.4f}",
            'Precision': f"{np.mean(fold_prec):.4f} +/- {np.std(fold_prec):.4f}",
            'Recall': f"{np.mean(fold_rec):.4f} +/- {np.std(fold_rec):.4f}",
            'F1 Score': f"{np.mean(fold_f1):.4f} +/- {np.std(fold_f1):.4f}",
            'ROC AUC': f"{np.mean(fold_auc):.4f} +/- {np.std(fold_auc):.4f}",
            'Brier Score': f"{np.mean(fold_brier):.4f} +/- {np.std(fold_brier):.4f}",
            'Mean F1 Raw': np.mean(fold_f1),
            'Mean AUC Raw': np.mean(fold_auc)
        })
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    filepath = get_dataset_path()
    validator = DataValidator(filepath)
    df_nat, df_bal = validator.prepare_experimental_conditions()
    
    print("\n=======================================================")
    print("EXPERIMENT A: NATURAL DISTRIBUTION (CLEAN / LEAKAGE-PRUNED)")
    print("=======================================================")
    res_nat_clean = evaluate_baseline_cv(df_nat, drop_leakage=True, n_splits=5)
    print(res_nat_clean[['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'Brier Score']].to_string(index=False))
    
    print("\n=======================================================")
    print("EXPERIMENT B: BALANCED CONDITION (CLEAN / LEAKAGE-PRUNED)")
    print("=======================================================")
    res_bal_clean = evaluate_baseline_cv(df_bal, drop_leakage=True, n_splits=5)
    print(res_bal_clean[['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'Brier Score']].to_string(index=False))
    
    # Save baseline benchmark results to results/
    os.makedirs("results", exist_ok=True)
    res_nat_clean.to_csv("results/baseline_natural_clean.csv", index=False)
    res_bal_clean.to_csv("results/baseline_balanced_clean.csv", index=False)
    print("\n[BASELINE BENCHMARK COMPLETE] Saved to results/ directory.")
