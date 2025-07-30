# ML-Project
# Market Regime Prediction with Machine Learning

This project applies machine learning techniques to forecast monthly U.S. equity market regimes (bullish or bearish) using daily S&P 500 excess returns and interest rate data. The work was completed as part of the **FIN-407 Machine Learning in Finance** course at EPFL.

## Problem Overview

We frame regime prediction as a **binary classification task**:
- **Bullish**: Positive excess return over the last 10 trading days of the month
- **Bearish**: Negative excess return over the same period

Only data available **at the start of the month** is used for feature generation to avoid lookahead bias.

---

## Features

- **Return-based**: Mean, std, min, max, and cumulative return from the first 10 trading days
- **Macro**: Interest rate level and monthly change

---

## Models and Approaches

- **Baseline models**: Logistic Regression, Random Forest
- **Tree-based model**: XGBoost (with tuning and SHAP explainability)
- **Neural networks**: Multiple PyTorch MLP architectures
- **Ensembles**:
  - **Soft-labeled ensemble** (XGBoost + MLP)
  - **Stacked model**: XGBoost probabilities used as features in an MLP

### Confidence-Aware Classification
We implement an abstention mechanism to avoid overconfident misclassifications:
- **Bullish**: P > θ_high  
- **Bearish**: P < θ_low  
- **Neutral**: Abstain if θ_low ≤ P ≤ θ_high


---

## Key Results

- **Best performance**: Stacked XGBoost + MLP
- **Accuracy** (confident predictions): up to **76.9%**
- **F1-Score**: Strong performance on bullish signals; bearish detection improved via custom loss functions and ensemble modeling
- **Model robustness**: Low prediction variance across random seeds
- **Interpretability**: SHAP values and gain-based feature importances

---

## Files

- `Report_ML.pdf`: Full project report with detailed methodology, results, and evaluation
- `model_code.ipynb` (or similar): Notebook or scripts implementing the models and analysis

---

## Team

Victor Nahoul · Salim Ameziane · Charaf Eddine Dahbi · Rami Mjalli · Jeffrey Rached  
EPFL · June 2025
