# Loan Approval Prediction - Part 1
import warnings; warnings.filterwarnings("ignore")
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

try:
    from data_validation import get_dataset_path
except ImportError:
    from src.data_validation import get_dataset_path
import os

df = pd.read_csv(get_dataset_path())
drop=[c for c in ["Unnamed: 0","id","year"] if c in df.columns]
df_base=df.drop(columns=drop).copy()
X=df_base.drop("status",axis=1); y=df_base["status"]
for c in X.select_dtypes(include=["int64","float64"]).columns: X[c]=X[c].fillna(X[c].median())
for c in X.select_dtypes(include=["object"]).columns: X[c]=X[c].fillna(X[c].mode()[0])
X=pd.get_dummies(X,drop_first=True); X.columns=X.columns.astype(str).str.replace(r"[^a-zA-Z0-9_]","_",regex=True)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
models={"Logistic Regression":LogisticRegression(max_iter=1000),"Random Forest":RandomForestClassifier(random_state=42),"XGBoost":XGBClassifier(random_state=42,eval_metric="logloss")}
def ev(m):
 m.fit(Xtr,ytr);p=m.predict(Xte);pr=m.predict_proba(Xte)[:,1]
 return [accuracy_score(yte,p),precision_score(yte,p),recall_score(yte,p),f1_score(yte,p),roc_auc_score(yte,pr)]
import pandas as pd
res={k:ev(v) for k,v in models.items()}
before_df=pd.DataFrame(res,index=["Accuracy","Precision","Recall","F1 Score","ROC AUC"]).T
print(before_df)
before_df.plot(kind="bar",figsize=(8,5)); plt.show()
df_clean=df.drop(columns=drop).copy()
for c in df_clean.select_dtypes(include=["int64","float64"]).columns: df_clean[c]=df_clean[c].fillna(df_clean[c].median())
for c in df_clean.select_dtypes(include=["object"]).columns: df_clean[c]=df_clean[c].fillna(df_clean[c].mode()[0])
sns.countplot(data=df_clean,x="status"); plt.show()
nums=df_clean.select_dtypes(include=["int64","float64"]).columns
df_clean[nums].hist(figsize=(18,15)); plt.show()
sns.heatmap(df_clean[nums].corr(),cmap="coolwarm"); plt.show()
