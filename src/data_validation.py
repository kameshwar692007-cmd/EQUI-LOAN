"""
data_validation.py — EquiLoan Research Framework
Validates raw datasets, performs leakage auditing, checks data integrity,
and prepares dual experimental conditions (Natural Distribution vs. Balanced Experimental Condition).
"""

import os
import pandas as pd
import numpy as np

class DataValidator:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df_raw = None
        self.df_natural = None
        self.df_balanced = None
        
    def load_data(self) -> pd.DataFrame:
        """Loads raw dataset and logs basic dimensions."""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Dataset path not found: {self.filepath}")
        
        print(f"[DATA VALIDATION] Loading dataset from {self.filepath}...")
        self.df_raw = pd.read_csv(self.filepath)
        print(f"[DATA VALIDATION] Raw Dataset Shape: {self.df_raw.shape[0]:,} rows x {self.df_raw.shape[1]} columns")
        return self.df_raw

    def inspect_health(self) -> dict:
        """Performs initial health check: duplicates, missing values, dtypes, target dist."""
        if self.df_raw is None:
            self.load_data()
            
        df = self.df_raw.copy()
        
        # Check target column
        if 'status' not in df.columns:
            raise KeyError("Target column 'status' not found in dataset.")
            
        target_dist = df['status'].value_counts(dropna=False).to_dict()
        target_pct = (df['status'].value_counts(normalize=True, dropna=False) * 100).round(2).to_dict()
        
        # Missing values
        missing = df.isnull().sum()
        missing_cols = missing[missing > 0].sort_values(ascending=False).to_dict()
        missing_pct = ((missing[missing > 0] / len(df)) * 100).round(2).to_dict()
        
        # Duplicates
        duplicates = df.duplicated().sum()
        
        health_report = {
            'total_rows': len(df),
            'total_cols': len(df.columns),
            'duplicates': duplicates,
            'target_counts': target_dist,
            'target_percentages': target_pct,
            'missing_counts': missing_cols,
            'missing_percentages': missing_pct
        }
        
        return health_report

    def audit_target_leakage(self) -> pd.DataFrame:
        """
        Audits rate columns for potential target leakage.
        Examines missingness by target status and correlation with status.
        """
        if self.df_raw is None:
            self.load_data()
            
        df = self.df_raw
        rate_cols = [c for c in ['rate_of_interest', 'interest_rate_spread', 'upfront_charges', 'high_interest_rate'] if c in df.columns]
        
        leakage_metrics = []
        for col in rate_cols:
            miss_status_0 = df[df['status'] == 0][col].isnull().mean() * 100
            miss_status_1 = df[df['status'] == 1][col].isnull().mean() * 100
            
            # Numeric correlation if numeric
            if np.issubdtype(df[col].dtype, np.number):
                corr = df[col].corr(df['status'])
            else:
                corr = np.nan
                
            leakage_metrics.append({
                'Feature': col,
                'Missing % in Approved (0)': round(miss_status_0, 2),
                'Missing % in Rejected (1)': round(miss_status_1, 2),
                'Correlation with Status': round(corr, 4) if not np.isnan(corr) else 'Categorical'
            })
            
        return pd.DataFrame(leakage_metrics)

    def prepare_experimental_conditions(self, random_state: int = 42) -> tuple:
        """
        Creates:
        1. Condition A: Natural Distribution (full 148,670 rows)
        2. Condition B: Balanced Experimental Condition (37,397 per class = 74,794 rows)
        """
        if self.df_raw is None:
            self.load_data()
            
        # Condition A: Natural Distribution
        self.df_natural = self.df_raw.copy()
        
        # Condition B: Exact Balanced Experimental Condition
        df_status_0 = self.df_raw[self.df_raw['status'] == 0]
        df_status_1 = self.df_raw[self.df_raw['status'] == 1]
        
        min_class_size = min(len(df_status_0), len(df_status_1))
        
        df_status_0_sampled = df_status_0.sample(n=min_class_size, random_state=random_state)
        df_status_1_sampled = df_status_1.sample(n=min_class_size, random_state=random_state)
        
        self.df_balanced = pd.concat([df_status_0_sampled, df_status_1_sampled], axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        
        print(f"[DATA VALIDATION] Condition A (Natural Distribution): {len(self.df_natural):,} rows | Target Dist: {self.df_natural['status'].value_counts().to_dict()}")
        print(f"[DATA VALIDATION] Condition B (Balanced Experimental): {len(self.df_balanced):,} rows | Target Dist: {self.df_balanced['status'].value_counts().to_dict()}")
        
        return self.df_natural, self.df_balanced

def get_dataset_path():
    candidates = [
        os.path.join("data", "raw", "loan_v2_real_distribution.csv"),
        os.path.join("..", "data", "raw", "loan_v2_real_distribution.csv"),
        os.path.join("dataset which u should use for training and testing", "loan_v2 (real distribution).csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]

if __name__ == "__main__":
    filepath = get_dataset_path()
    validator = DataValidator(filepath)
    health = validator.inspect_health()
    print("\n--- DATA HEALTH SUMMARY ---")
    print(f"Total Rows: {health['total_rows']:,}")
    print(f"Target Breakdown: {health['target_counts']}")
    print(f"Duplicates: {health['duplicates']}")
    
    print("\n--- TARGET LEAKAGE AUDIT ---")
    leak_df = validator.audit_target_leakage()
    print(leak_df.to_string(index=False))
    
    validator.prepare_experimental_conditions()

