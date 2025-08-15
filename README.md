# Loan-to-Cap Ratio (LTC) 
## Loan Default Prediction
Project Overview
This project builds a machine learning pipeline to predict loan default risk using not only traditional borrower information (like income and credit history) but also a novel financial resilience metric called the Loan-to-Cap (LTC) Ratio. The LTC ratio measures a borrower’s total loan amount relative to their insurance coverage limits, adding a critical dimension of financial safety net analysis to credit risk modeling.

Why This Matters
Traditional credit scoring models often overlook whether a borrower has significant insurance coverage that can prevent default during crises. LTC provides insight into how well a borrower’s debt is protected by insurance, which can improve prediction accuracy and help lenders better manage portfolio risk.

Better risk assessment: Incorporates insurance coverage into loan risk models.

Fairer lending: Recognizes resilient borrowers who may have thin credit but strong insurance.

Regulatory compliance: Builds explainable and transparent AI models using modern interpretability tools.

Features
Data loading with fallback to Kaggle dataset, manual upload, or synthetic data generation.

Robust data cleaning and missing value imputation.

Feature engineering including generation of synthetic insurance-related variables and LTC ratio.

Encoding of categorical variables into machine learning-friendly formats.

Scaled and stratified train-test data split.

Model training across multiple algorithms:

Logistic Regression

Random Forest

XGBoost

CatBoost

LightGBM

Gradient Boosting

HistGradientBoosting

ExtraTrees

Model evaluation with metrics: Accuracy, F1-score, ROC-AUC, Precision-Recall AUC.

Visualizations: Accuracy trends, ROC and PR curves, confusion matrices.

Optional SHAP-based model explainability for top boosting models.

Installation
Clone this repository:

bash
git clone https://github.com/Nuthan10/loan-to-cap-ratio.git
cd loan-to-cap-ratio
Create and activate a Python environment (recommended):

bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install the required packages:

bash
pip install -r requirements.txt
(Alternatively, direct install of dependencies like xgboost, catboost, lightgbm, shap, kagglehub, scikit-learn, pandas, matplotlib, numpy)

Usage
Run the main notebook/script:

It attempts to load data automatically from Kaggle.

If unavailable, it asks for manual upload or generates synthetic data.

The notebook performs:

Data preprocessing and feature engineering.

Model training and evaluation.

Plots of metrics and error analysis.

To enable SHAP explainability, set DO_SHAP = True in the notebook.

File Structure
loan_to_cap_ratio.ipynb — Main Jupyter notebook with full pipeline

README.md — This file

requirements.txt — Python dependencies

data/ — (Optional) directory for datasets if manually stored

Results Interpretation
The Accuracy plot shows stability and comparative performance across model families.

The ROC & PR curves give insight into sensitivity and precision trade-offs.

Confusion matrices help assess error types and model reliability.

SHAP plots visualize feature importance and aid explainability.

Contributing
Contributions, issues, and feature requests are welcome!
Feel free to fork the repo and submit pull requests.

License
Distributed under the MIT License. See LICENSE for more information.

Acknowledgments
Kaggle: Original Loan Prediction Dataset

SHAP library for model explainability

Scikit-learn and other ML libraries for algorithms and utilities
