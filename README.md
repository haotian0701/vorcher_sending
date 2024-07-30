### vorcher_sending

# Introduction
In the e-commerce sector, a significant number of customers make a single purchase and then never return. To combat this, many platforms use strategies like discount vouchers to encourage repeat purchases. However, indiscriminately giving out these vouchers is not cost-effective, as some customers would have made repeat purchases without any incentives. Our goal is to develop a predictive model to determine whether a €5.00 voucher should be issued to a customer based on their initial purchase characteristics.

# Objective
The primary objective of this project is to predict if a customer will place a subsequent order within a 90-day period following their initial purchase (represented by the target90 variable in the dataset). The decision to issue a voucher is based on this prediction, with the aim of maximizing the expected revenue.

# Dataset
The dataset (train.csv) contains various features associated with a customer’s initial order, along with the target variable target90. A detailed description of these features is provided in the data dictionary.pdf file.

# Methodology
- Data Preprocessing:
Load and inspect the dataset.
Handle missing values and perform necessary data cleaning.
Feature engineering to create or transform variables that can enhance model performance.

- Model Development:
Split the data into training and validation sets.
Train multiple machine learning models to predict the target90 variable.
Evaluate model performance using metrics like accuracy, precision, recall, and F1-score.
Select the best-performing model based on these metrics.

- Revenue Optimization:
Calculate the expected revenue for each customer based on model predictions.
Develop a strategy to issue vouchers only to those customers whose predicted behavior maximizes overall revenue.
Validate the strategy by comparing the expected revenue against actual outcomes.
