# EquiLoan: Explainable, Fair, Gated, & Calibrated Machine Learning Framework for Loan Approval Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: Scikit-Learn / LightGBM / XGBoost](https://img.shields.io/badge/Framework-Scikit--Learn%20%7C%20LightGBM%20%7C%20XGBoost-orange.svg)](https://scikit-learn.org/)
[![Fairness & XAI](https://img.shields.io/badge/XAI-SHAP%20%26%20Demographic%20Parity-green.svg)](docs/report/EquiLoan_FINAL__UG_REPORT.pdf)

---

## 📌 Executive Summary

**EquiLoan** is an end-to-end Machine Learning research framework built for automated, fair, and reliable credit risk assessment. Traditional algorithmic credit scoring models often suffer from bias against demographic subgroups, uncalibrated probability estimates, and overconfident predictions on edge cases.

EquiLoan addresses these issues through a multi-tiered pipeline:
1. **Data Validation & Preprocessing**: Automated schema checks, outlier mitigation, and missing value imputation on real-world loan distributions.
2. **Gradient-Boosted & Ensemble Baselines**: XGBoost, LightGBM, Random Forest, and Logistic Regression baseline benchmarking.
3. **Probability Calibration**: Platt Scaling and Isotonic Regression to map raw output scores to empirical posterior probabilities.
4. **Demographic & Intersectional Fairness Audit**: Quantitative evaluation of Demographic Parity, Equal Opportunity, and Equalized Odds across demographic subgroups.
5. **Confidence-Gated Abstention (Reject Option)**: Confidence thresholding mechanism that routes ambiguous loan applications to human underwriters.
6. **Robustness & Noise Sensitivity Analysis**: Stress-testing feature stability under continuous Gaussian noise perturbations.

---

## 📁 Repository Structure

```
pbl-ds/ (EQUI-LOAN)
│
├── README.md                            # Comprehensive Project Guide & Architecture
├── requirements.txt                      # Project Dependencies
├── setup.py                              # Modular Package Installation Script
├── LICENSE                               # MIT License
├── .gitignore                           # Configured Git Exclusions
│
├── data/                                # Dataset Storage
│   └── raw/
│       └── loan_v2_real_distribution.csv# Primary Dataset (Real Distribution)
│
├── notebooks/                           # Ordered Machine Learning Notebooks
│   ├── 01_data_exploration.ipynb        # Exploratory Data Analysis (EDA)
│   ├── 02_model_experimentation.ipynb   # Baseline Model Experiments & Tuning
│   ├── 03_EquiLoan_Master_Notebook.ipynb# Production Master Notebook
│   └── archive/                         # Experimental Drafts & Historical Notebooks
│
├── src/                                 # Modular Python Source Package (`equiloan`)
│   ├── __init__.py                      # Package Initialization
│   ├── data_validation.py               # Data Integrity & Schema Validation
│   ├── preprocessing.py                 # Feature Scaling, Encoding & Imputation
│   ├── baseline_models.py               # Model Definitions & Training
│   ├── evaluation.py                    # Metric Calculation (ROC-AUC, F1, G-Mean)
│   ├── fairness.py                      # Intersectional Fairness Audit Engine
│   ├── calibration.py                   # Platt & Isotonic Probability Calibration
│   ├── abstention.py                    # Confidence-Gated Reject-Option Router
│   └── robustness.py                    # Noise Sensitivity & Feature Perturbation
│
├── dashboards/                          # Interactive Web Interfaces
│   ├── EquiLoan_Dashboard.html          # Web Decision-Support Dashboard
│   └── EquiLoan_EvoForge_Terminal.html  # Interactive Terminal Dashboard
│
├── figures/                             # Visual Artifacts & Graphics
│   ├── plots/                           # Generated Benchmark & Stability Plots
│   ├── screenshots/                     # Interface & Audit Screenshots
│   └── archive/                         # Visual Archives
│
├── docs/                                # Documentation & Academic References
│   ├── report/                          # Final Undergraduate Project Reports
│   ├── references/                      # Peer-Reviewed Academic Literature
│   └── workflow/                        # Step-by-Step Execution Walkthroughs
│
└── results/                             # Quantitative Scorecards & Audit Logs (CSVs)
    ├── baseline_balanced_clean.csv      # Baseline Performance Metrics
    ├── fairness_audit_balanced.csv      # Fairness Audit Metrics
    ├── calibration_balanced.csv         # Calibration Error Metrics (ECE)
    ├── abstention_balanced.csv          # Abstention Trade-off Metrics
    ├── feature_stability_balanced.csv   # Perturbation Sensitivity Metrics
    └── final_research_scorecard.csv     # Combined Summary Scorecard
```

---

## ⚙️ Key Data Science Components

### 1. Data Validation (`src/data_validation.py`)
Ensures incoming loan applicant records satisfy structural and domain constraints prior to inference:
- Numerical bounds verification (e.g. `Income >= 0`, `Credit_Score ∈ [300, 850]`).
- Detection of non-null constraints and unexpected data type conversions.

### 2. Preprocessing & Feature Engineering (`src/preprocessing.py`)
- Standardized scaling (`StandardScaler` / `RobustScaler`) for continuous variables.
- One-Hot / Target encoding for categorical features (Employment Type, Education Level, Loan Intent).
- Class-imbalance adjustment using Synthetic Minority Over-sampling (SMOTE) and cost-sensitive reweighting.

### 3. Model Benchmark & Evaluation (`src/baseline_models.py`, `src/evaluation.py`)
Evaluates classifiers across multiple statistical metrics:
- **Primary Metrics**: ROC-AUC, PR-AUC, G-Mean (Geometric Mean of Sensitivity and Specificity), F1-Score, Brier Score.
- **Algorithms Benchmark**: LightGBM, XGBoost, Random Forest, Logistic Regression, Gradient Boosting.

### 4. Intersectional Fairness Audit (`src/fairness.py`)
Quantifies algorithmic bias across sensitive attributes (Gender, Age Group, Marital Status):
- **Demographic Parity Difference ($\Delta DP$)**: $|P(\hat{Y}=1 | A=0) - P(\hat{Y}=1 | A=1)|$
- **Equal Opportunity Difference ($\Delta EO$)**: $|P(\hat{Y}=1 | Y=1, A=0) - P(\hat{Y}=1 | Y=1, A=1)|$

### 5. Probability Calibration (`src/calibration.py`)
Fixes miscalibrated model confidence probabilities using:
- **Platt Scaling (Sigmoidal Regression)**
- **Isotonic Regression (Non-parametric)**
- Measures **Expected Calibration Error (ECE)** before and after calibration.

### 6. Confidence-Gated Abstention Router (`src/abstention.py`)
Implements a safety mechanism for automated decision-making:
- High-confidence predictions ($\hat{P} \ge \tau_{\text{high}}$ or $\hat{P} \le \tau_{\text{low}}$) are automatically approved/rejected.
- Low-confidence predictions ($\tau_{\text{low}} < \hat{P} < \tau_{\text{high}}$) trigger **Abstention**, routing the applicant for manual review.

---

## 🚀 Getting Started & How to Run

### Prerequisites
- Python 3.8+ installed on your environment.

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/kameshwar692007-cmd/EQUI-LOAN.git
cd EQUI-LOAN

# Install required Python dependencies
pip install -r requirements.txt

# Install equiloan package in editable mode
pip install -e .
```

### 2. Running Jupyter Notebooks
Open the notebooks in sequential order:
```bash
jupyter notebook notebooks/
```
1. **`notebooks/01_data_exploration.ipynb`**: Conduct exploratory analysis on applicant distributions.
2. **`notebooks/02_model_experimentation.ipynb`**: Run baseline model comparison and hyperparameter tuning.
3. **`notebooks/03_EquiLoan_Master_Notebook.ipynb`**: Run the complete end-to-end production pipeline (Validation → Training → Calibration → Fairness Audit → Abstention).

### 3. Running Core Python Modules Directly
You can run individual modules from the command line or import them into custom scripts:
```python
from src.data_validation import validate_loan_data
from src.preprocessing import preprocess_data
from src.fairness import audit_demographic_fairness

# Example Usage
print("EquiLoan Framework initialized successfully.")
```

### 4. Viewing Interactive Dashboards
Open the HTML dashboards in any web browser:
- Double-click `dashboards/EquiLoan_Dashboard.html` or `dashboards/EquiLoan_EvoForge_Terminal.html` to inspect metrics, feature importance, and interactive decision trees visually.

---

## 📊 Benchmark Summary Results

| Model | ROC-AUC | PR-AUC | G-Mean | F1-Score | ECE (Calibrated) |
|---|---|---|---|---|---|
| **LightGBM (Calibrated)** | **0.892** | **0.865** | **0.841** | **0.828** | **0.014** |
| **XGBoost** | 0.884 | 0.851 | 0.830 | 0.817 | 0.028 |
| **Random Forest** | 0.867 | 0.832 | 0.815 | 0.801 | 0.035 |
| **Logistic Regression** | 0.795 | 0.748 | 0.732 | 0.710 | 0.049 |

---

## 📄 License & Attribution

This project is licensed under the [MIT License](LICENSE). 
Developed as part of Data Science & Machine Learning Research on Fair Credit Scoring.
