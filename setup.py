from setuptools import setup, find_packages

setup(
    name="equiloan",
    version="1.0.0",
    description="An Explainable, Fair, Gated, & Calibrated Machine Learning Framework for Loan Approval Prediction",
    author="Kameshwar M",
    author_email="kameshwar692007@gmail.com",
    url="https://github.com/kameshwar692007-cmd/EQUI-LOAN",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "shap",
        "matplotlib",
        "seaborn",
        "joblib",
        "scipy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
