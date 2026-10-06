"""
preprocessing.py — EquiLoan Research Framework
Leakage-Free Modular Preprocessing Pipelines.
All fit operations (imputation medians, scaling parameters, outlier fences, SMOTE)
are strictly performed on Training Folds ONLY.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

DROP_ID_COLS = ['Unnamed: 0', 'id', 'year']
POST_UNDERWRITING_LEAKAGE_COLS = ['interest_rate_spread', 'upfront_charges', 'rate_of_interest', 'high_interest_rate']

class IQROutlierClipper(BaseEstimator, TransformerMixin):
    """
    Fits IQR fences (Q1 - 1.5*IQR, Q3 + 1.5*IQR) on TRAIN fold only,
    and clips TRAIN and TEST folds accordingly to prevent data leakage.
    """
    def __init__(self, target_cols=None, factor=1.5):
        self.target_cols = target_cols
        self.factor = factor
        self.fences_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        cols_to_clip = self.target_cols if self.target_cols else X_df.select_dtypes(include=[np.number]).columns
        
        for col in cols_to_clip:
            if col in X_df.columns:
                q1 = X_df[col].quantile(0.25)
                q3 = X_df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - self.factor * iqr
                upper = q3 + self.factor * iqr
                self.fences_[col] = (lower, upper)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        for col, (lower, upper) in self.fences_.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].clip(lower, upper)
        return X_df

def create_preprocessing_pipeline(df: pd.DataFrame, drop_leakage: bool = True, apply_iqr: bool = True):
    """
    Builds a leakage-free Scikit-Learn Pipeline / ColumnTransformer.
    - drop_leakage: If True, removes post-decision rate features.
    - apply_iqr: If True, applies IQROutlierClipper fitted on train.
    """
    # Copy and drop identifier columns
    cols_to_drop = [c for c in DROP_ID_COLS if c in df.columns]
    if drop_leakage:
        cols_to_drop.extend([c for c in POST_UNDERWRITING_LEAKAGE_COLS if c in df.columns])
        
    feature_df = df.drop(columns=cols_to_drop + ['status'], errors='ignore')
    
    num_cols = feature_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = feature_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Numeric Pipeline
    num_steps = [('imputer', SimpleImputer(strategy='median'))]
    if apply_iqr:
        num_steps.append(('iqr_clipper', IQROutlierClipper(target_cols=num_cols)))
    num_steps.append(('scaler', StandardScaler()))
    
    num_pipeline = Pipeline(num_steps)
    
    # Categorical Pipeline
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ],
        remainder='drop'
    )
    
    return preprocessor, num_cols, cat_cols

if __name__ == "__main__":
    from data_validation import DataValidator
    import os
    filepath = os.path.join("dataset which u should use for training and testing", "loan_v2 (real distribution).csv")
    validator = DataValidator(filepath)
    df_nat, _ = validator.prepare_experimental_conditions()
    
    preprocessor, num_c, cat_c = create_preprocessing_pipeline(df_nat, drop_leakage=True, apply_iqr=True)
    print(f"[PREPROCESSING] Pipeline instantiated. Numeric features: {len(num_c)}, Categorical features: {len(cat_c)}")
