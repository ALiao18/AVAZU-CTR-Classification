# AVAZU Clickthrough Rate Prediction

## Project Overview

This project focuses on predicting click-through rates (CTR) for mobile advertisements using the AVAZU dataset. The goal is to build and evaluate machine learning models that can accurately predict whether a user will click on an ad based on various features. The dataset presents an imbalanced classification problem with a 17% click rate.

## Dataset Description

The dataset contains various features related to mobile advertisements including:
- Temporal features (hour)
- Device information (device_type, device_conn_type) 
- App/site information (app_id, site_id, site_domain, site_category)
- Device details (device_id, device_ip, device_model)
- Anonymous categorical variables (C1-C21)

## Feature Engineering

### 1. High Cardinality Feature Treatment (site_id)
- Applied Empirical Bayes Estimation to handle sites with few entries
- Used method of moments to estimate β distribution parameters
- Grouped sites by quantiles (12 groups) 
- Validated site CTR stability across different time frames
- Identified 1,322 instances with high temporal CTR variation (>0.01 difference)

### 2. Statistical Significance Based Category Merging
- Calculated minimum required sample size for reliable CTR estimation
- Merged site categories below threshold with closest CTR categories
- Reduced from 22 to 5 site categories while preserving 99% of data representation

### 3. Temporal Feature Engineering
- Implemented cyclical hour encoding using sine and cosine transformations
- Created time blocks (morning, workday, evening, night)
- Generated site_id * time block interaction features
- Developed 1-hour CTR window features:
  - Previous hour click count
  - Previous hour mean click rate
  - Handled missing values with appropriate defaults

## Model Development & Results

| Model                          | AUC      | Negative Log Loss |
|-------------------------------|----------|------------------|
| Base Logistic Regression      | 0.611000 | 0.442000        |
| Logistic Regression           | 0.697078 | 0.420385        |
| Neural Network                | 0.767106 | 0.389971        |
| Decision Tree                 | 0.740012 | 0.400956        |
| Decision Tree (XGBoost)       | 0.740012 | 0.400956        |

### Neural Network Success Factors
1. Superior learning of complex temporal dependencies
2. Better handling of high dimensional data with appropriate embedding sizes
3. Ability to learn higher-level feature representations through multiple layers
4. Better performance with high cardinality features compared to tree-based models that rely on one-hot encoding

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- tensorflow
- xgboost
- matplotlib
- seaborn

## Future Work

1. **Model Improvements**
   - Experiment with different tree depths in decision tree models
   - Fine-tune neural network architecture
   - Investigate better handling of high cardinality features

2. **Feature Engineering**
   - Explore additional temporal patterns
   - Develop more sophisticated site grouping methods
   - Create additional interaction features

3. **Model Blending**
   - Implement ensemble techniques
   - Test different blending strategies
   - Optimize model weights

## Acknowledgments

- Professor Pascal Bianchi for providing the dataset
- Kaggle community for insights 
