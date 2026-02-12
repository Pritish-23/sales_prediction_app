Product Sales Prediction using Review Analytics

An AI-powered decision-support system that analyzes customer reviews to estimate product sales performance and generate actionable business insights.

Project Overview

This project presents a Machine Learning-based framework for predicting the sales performance of a product using customer review data. Instead of relying solely on historical sales figures, the system leverages review sentiment, rating distribution, textual patterns, and engagement metrics to infer sales strength.

The application is implemented using Streamlit and deployed as an interactive web dashboard.

Objectives

Predict whether a product belongs to a High Sales or Low Sales category.

Compute a normalized Sales Strength Index (0–1).

Analyze customer sentiment distribution.

Identify key business-relevant themes from textual reviews.

Generate actionable business insights.

Visualize review activity trends over time.

Machine Learning Approach

The system uses two supervised learning models:

1. Classification Model

Predicts whether a product belongs to a High Sales or Low Sales category.

2. Regression Model

Predicts a normalized Sales Strength Index ranging between 0 and 1.

Feature Engineering Includes:

Average rating

Review count

Sentiment proportions

Text-based token features

Bigram patterns

Temporal engagement features

Application Features

The Streamlit dashboard provides:

CSV upload for a single product review dataset

Data preview section

Customizable analysis controls

Executive summary dashboard

Sales performance evaluation

Review activity trend visualization

Business interpretation section

Key theme detection

Review highlights (positive and negative)

Automated actionable insights

Project Structure
project-root/
│
├── app.py
├── requirements.txt
├── models/
│   ├── sales_classifier.pkl
│   ├── sales_regressor.pkl
│   ├── feature_order.pkl
│   ├── reg_feature_order.pkl
│   ├── sales_score_p05.pkl
│   └── sales_score_p95.pkl
│
├── utils/
│   ├── preprocess.py
│   └── __init__.py
│
└── README.md
Installation and Setup
1. Clone the Repository
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
2. Install Dependencies
pip install -r requirements.txt
3. Run the Application Locally
streamlit run app.py
Deployment

The application is deployed using Streamlit Community Cloud.

To deploy:

Push the code to GitHub.

Log in to Streamlit Cloud.

Select the repository.

Choose app.py as the main file.

Deploy the application.

Input Dataset Requirements

The CSV file must contain:

Review text column

Rating column

Time column (in UNIX timestamp format)

Each dataset should represent reviews of a single product.

Sales Strength Index

The Sales Strength Index:

Is a normalized indicator between 0 and 1

Reflects relative sales performance

Is not a future sales forecast

Is not a probability score

Business Value

This system assists:

E-commerce managers

Product strategists

Marketing teams

Business analysts

It enables them to:

Detect early sales signals

Understand customer sentiment

Identify operational issues

Optimize product positioning

Technologies Used

Python

Streamlit

Pandas

Scikit-learn

XGBoost

Joblib

NumPy

Academic Context

This project was developed as part of an academic Machine Learning and Data Analytics initiative focusing on real-world business intelligence applications using Natural Language Processing and predictive modeling.

Author

Your Name
Bachelor of Engineering / Computer Science
LinkedIn: [Your LinkedIn Profile]
GitHub: [Your GitHub Profile]

Disclaimer

This model predicts relative sales performance based on review-derived features. It does not guarantee actual sales outcomes and should be used as a decision-support tool.
