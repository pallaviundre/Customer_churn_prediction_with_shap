# Customer Churn Prediction with Model Interpretation

This project builds an end-to-end machine learning pipeline to predict customer churn using structured customer data. The objective is not only to achieve strong predictive performance but also to **interpret model decisions** using explainability techniques.

## Problem Statement
Customer churn is a critical business problem where companies aim to identify customers who are likely to leave their service. Early identification helps businesses take proactive retention actions and reduce revenue loss.

## Approach
- Performed data preprocessing and feature engineering on customer demographic and behavioral features
- Trained multiple machine learning models, including Logistic Regression and XGBoost
- Evaluated model performance using appropriate classification metrics with a focus on precision–recall trade-offs
- Applied cross-validation to ensure robust model performance
- Used SHAP (SHapley Additive exPlanations) to interpret both global feature importance and individual predictions

## Key Highlights
- End-to-end supervised machine learning workflow
- Use of tree-based models to capture non-linear relationships
- Model validation using cross-validation
- Explainable AI using SHAP to identify key churn drivers
- Business-oriented interpretation of model outputs

## Tools & Technologies
- Python
- NumPy, Pandas
- scikit-learn
- XGBoost
- SHAP
- Matplotlib / Seaborn
- Jupyter Notebook

## Results
The final model achieved strong classification performance while maintaining interpretability. SHAP analysis revealed the most influential features contributing to customer churn, enabling actionable insights for retention strategies.

## Future Improvements
- Hyperparameter tuning using Bayesian optimization
- Incorporating temporal behavior features
- Cost-sensitive learning to better handle churn-related business trade-offs


