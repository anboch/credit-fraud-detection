# Credit Card Fraud Detection: An End-to-End Machine Learning Solution

This repository presents a comprehensive, production-ready solution for detecting fraudulent credit card transactions. Moving beyond a simple classification task, this project emphasizes a business-centric approach, from a strategic exploratory data analysis (EDA) to cost-sensitive threshold optimization and robustness testing, culminating in a live deployment via Streamlit.

## https://credit-fraud-detection-2eezbh5cuipaatpsqqdpkg.streamlit.app/

![Streamlit App Demo GIF](https://github.com/anboch/credit-fraud-detection/blob/2353641cabcf3d1beff0b5e82aecb43cbd4e5a58/readme_assets/gif_demo.gif)

## Strategic summary

In a landscape of extreme class imbalance (only 0.173% of transactions are fraudulent), accuracy is a deceptive metric. This project re-frames the challenge from a simple classification problem to a business optimization task. The core objective is not just to predict fraud, but to build a robust, interpretable, and financially sound detection system.

The final solution is a tuned XGBoost classifier that delivers a high AUPRC of 0.8845 and a business-optimized decision threshold that maximizes net financial savings.

## Key project highlights

This project stands out by focusing on the practical and strategic aspects of the machine learning lifecycle:

### Business threshold optimization:
 Instead of relying on default metrics like F1-Score, I developed a cost model to find the probability threshold that maximizes net financial savings. This analysis directly translates model performance into business impact, demonstrating a potential saving of over $8,650 on the test set compared to a no-model baseline.

 ### Model robustness analysis: 
 The model's heavy reliance on the top feature (V14) was identified as a production risk. A robustness check was conducted by removing V14 and retraining the model, proving its ability to adapt and maintain high performance (AUPRC 0.8714) by leveraging secondary features.

 ### Advanced hyperparameter tuning: 
 Employed Optuna for efficient Bayesian optimization of the XGBoost model, using stratified cross-validation to ensure the AUPRC metric was robust and generalizable.

 ### Live interactive deployment: 
 The final model, scaler, and feature pipeline were serialized and deployed as an interactive web application using Streamlit, providing a tangible proof-of-concept.

### Strategic EDA: 
The analysis went beyond surface-level plots to uncover actionable insights:

`-` Fraudulent transactions are decoupled from human circadian rhythms, peaking when legitimate activity is lowest.

`-` Fraud exhibits a dual monetary tactic: a high volume of low-value "card testing" transactions and a smaller number of high-value "cash-out" events.

`-` The feature space contains non-linear, separable clusters of fraud, justifying the choice of a tree-based model over linear alternatives.

## Technical workflow

The project follows a structured, end-to-end machine learning workflow:

### Exploratory Data Analysis (01_EDA.ipynb):

 Deep dive into feature distributions, class imbalance, and temporal/monetary patterns to form a data-driven modeling strategy.

### Modeling & Optimization (02_MODEL.ipynb):

`-` Establish a robust baseline using Logistic Regression.

`-` Develop and tune an XGBoost model using Optuna, focusing on AUPRC.

`-` Conduct a cost-benefit analysis to determine the optimal business threshold.

`-` Perform a robustness check by removing the most critical feature.

### Deployment (app.py):

Serialize the final model, pre-processing objects, and feature lists using joblib for a seamless deployment on Streamlit.

## Key Findings & Analysis

### Insights from EDA:

The exploratory analysis confirmed that fraud is not random noise but a pattern that can be learned. The most powerful signals were found in the anonymized PCA features, particularly V14, V4, and V12. The scatter plot below illustrates an example of how fraudulent transactions form a distinct, separable cluster with certain features, validating the choice of a non-linear model.

![Fraud footprint](https://github.com/anboch/credit-fraud-detection/blob/2353641cabcf3d1beff0b5e82aecb43cbd4e5a58/readme_assets/fraud_footprint.png)

### From metric to impact: Cost-Sensitive thresholding

A model's value is in its decisions. While the F1-score is a balanced metric, it treats false positives and false negatives equally. In fraud detection, a missed fraud (False Negative) is often far more costly than a false alarm (False Positive).
By assigning a hypothetical administrative cost of €10 to each FP and using the actual transaction amount for each FN, we can calculate the net savings at every possible probability threshold. The analysis revealed that a threshold of 0.12 maximizes savings, a stark contrast to the F1-optimal threshold (near 1). This insight is critical for deploying the model in a real-world financial context.

![Compare threshold](https://github.com/anboch/credit-fraud-detection/blob/2353641cabcf3d1beff0b5e82aecb43cbd4e5a58/readme_assets/compare_thresholds.png)

## Live demo usage

The final model is deployed as a live Streamlit application (URL showed before)

The interface allows you to input transaction details for the most impactful features. The model then predicts the probability of the transaction being fraudulent, using the business-optimized threshold of 12% to issue an alert.

## How to run locally:

To run the Streamlit application on your local machine, follow these steps:

### 1. Clone the repository:

```bash
git clone https://github.com/anboch/credit-fraud-detection.git
cd credit-fraud-detection
```

### 2. Set up a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 3. Run the Streamlit app:

```bash
streamlit run app.py
```

The application will open in your web browser.