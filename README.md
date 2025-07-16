# Finance Protector – AI-Powered Bankruptcy Risk Prediction for Stakeholders

## Overview
Finance Protector is a predictive analytics platform designed to tackle the rising risk of corporate bankruptcies across global financial sectors. With insolvency filings increasing by over 33.5% worldwide, investors, lenders, and financial institutions require smarter early-warning systems. Our platform combines three advanced models, MARS, XGBoost, and Random Forest, into a unified web-based tool that offers real-time predictions, explainability, and strategic financial insights.

## Objectives
- **Investors**: Empower investors to detect distressed firms and divest early
- **Lenders**: Help lenders assess credit risk and triage loan applicants
- **Financial Institutions**: Equip financial institutions with sector-wide risk monitoring tools

## Dataset
Bankruptcy data from the Taiwan Economic Journal (1999–2009)
[https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction/data](url)

## Key Insights & Features
- **Aggressive Bankruptcy Detection**: Tuned Random Forest captures 86.36% of bankruptcies with minimal false negatives  
- **Model Performance**: Random Forest achieves the highest sensitivity (86.36%) and AUC-ROC (0.9324)
- **Risk Explainability**: SHAP & MARS thresholds reveal high-risk features like borrowing dependency, ROA, and net worth
- **Web Dashboard**: Role-based tools for investors, lenders, and institutions featuring visual alerts, stakeholder-specific views, and predictive insights
- **Early Warning System**: Combines model predictions with real-time alerts and sector flags

## Stakeholder Solutions
### For Investors
- Bankruptcy heatmaps, risk dashboards and SHAP-driven feature insights  
- ESG-compliant thresholds (MARS) for ethical investing  
- "What-if" scenario testing using MARS interaction logic

### For Lenders
- Real-time loan applicant screening with XGBoost  
- Risk tiering and loan segmentation with Random Forest
- Transparent decision logs for credit justifications (MARS thresholds)
  
### For Financial Institutions
- Cross-sector monitoring with Random Forest clustering  
- Systemic risk alerting based on feature drift  
- Board-ready dashboards for policy and governance use
  
## Visual Highlights
- ROC curves comparing tuned and default models  
- SHAP summary plots for top bankruptcy drivers  
- Confusion matrices showcasing detection performance  
- Threshold tables with decision rules (MARS)  
- Cluster maps of bankruptcy-prone firms (Random Forest)

## Impact & Scalability
- **For Hedge Funds** (eg. BlackRock): Pre-emptive divestment strategies  
- **For SME Lenders** (eg. Kabbage): Smarter, fairer credit decisions  
- **For Banks/Regulators** (eg. JPMorgan, MAS): Sector-wide risk intelligence and compliance readiness

## Technologies
- **Languages & Frameworks**: Python & R for data analytics, machine learning, and visualisation
- **Machine Learning Libraries**: `scikit-learn`, `xgboost`, `py-earth` (MARS), `shap`, `imbalanced-learn`
- **Data Preprocessing**: `pandas`, `numpy`, `KNNImputer`, `SMOTE` for cleaning, imputation, and class balancing
- **Visualisation Tools**: `matplotlib`, `seaborn`, `plotly` for exploratory data analysis and model performance plots
- **Web Interface & Dashboard**: Tableau for interactive dashboards tailored to stakeholders (investors, lenders, institutions)
- **Model Explainability**: SHAP values (XGBoost & Random Forest) and interpretable MARS threshold rules for actionable insights

