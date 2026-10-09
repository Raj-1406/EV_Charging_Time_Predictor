# Smart EV Charging Time Predictor

## Overview
A classical Machine Learning project that predicts the expected duration of an electric vehicle charging session using charging-session details available at the beginning of the session.

## Problem Statement
EV charging duration varies across chargers, locations, and session-start conditions. This project estimates session duration in minutes using historical charging transactions.

## Dataset
The project uses the [EV Charging Transactions dataset on Figshare](https://figshare.com/articles/dataset/A_dataset_for_multi-faceted_analysis_of_electric_vehicle_charging_transactions/22495141).

The supplied CSV contains 72,856 original records and 13 columns. After removing exact duplicates and non-positive duration records, 72,108 records remained.

## Input Features
- ChargerCompany
- Location
- ChargerType
- StartHour
- DayOfWeek
- Month
- IsWeekend

**Target variable:** Duration, measured in minutes.

## Algorithms Compared
1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

## Model Evaluation

Gradient Boosting was selected as the best-performing model among the three tested algorithms.

| Metric | Result |
|---|---:|
| MAE | 83.80 minutes |
| RMSE | 125.94 minutes |
| R² | 0.2666 |

These results indicate that the model has limited predictive performance, and actual session duration may differ considerably from the predicted value.

## Technologies
Python, Pandas, NumPy, Matplotlib, Scikit-learn, Joblib, Streamlit, GitHub.

## Run Locally

1. Download or clone the repository.
2. Install dependencies:

   `python -m pip install -r requirements.txt`

3. Start the application:

   `python -m streamlit run app.py`

## Application Features
- Predict EV charging-session duration.
- Preview the dataset.
- Compare regression models.
- View project workflow and limitations.

## Deployment
Hosted using Streamlit Community Cloud.

**Live application:** Add your deployed URL here.

## Future Scope
Evaluate additional prediction-time features, if available, and investigate methods to improve generalization and prediction errors.