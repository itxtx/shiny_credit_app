# Shiny Credit Card Fraud Detection Dashboard

This is a Shiny R application that provides an interactive dashboard for evaluating and comparing machine learning models for credit card fraud detection. This project extends the comprehensive analysis from the [Credit Card Fraud Prediction](https://github.com/itxtx/credit_card_fraud_prediction) repository.

## Overview

The original project compared supervised models (Logistic Regression, Random Forest, XGBoost) and unsupervised models (Isolation Forest, Local Outlier Factor, Autoencoder) for credit card fraud detection. This Shiny application provides an interactive interface to explore the model predictions and performance metrics from that analysis.

## Features

### Dashboard
- **Model Selection**: Choose between XGBoost, Random Forest, Logistic Regression, and unsupervised models
- **Dynamic Threshold Control**: Adjust classification thresholds with optimal defaults
- **Real-time Metrics**: Precision, Recall, F1-Score, Matthews Correlation Coefficient
- **Interactive Plots**: Precision-Recall curves, ROC curves, score distributions
- **Advanced Analytics**: Comprehensive suite of plots for supervised models including:
  - Threshold analysis
  - Cumulative gain charts
  - Class-wise performance metrics
  - Calibration plots
  - Kolmogorov-Smirnov statistics
  - Score distribution by class

### Model Comparison
- Side-by-side performance comparison across all available models
- Bar charts and detailed metrics tables
- Optimal threshold-based comparisons

### Documentation
- Model overview and usage instructions
- Educational content on imbalanced classification best practices

## Data Requirements

The application expects CSV files with the following structure:
- `true_label`: Binary labels (0 for normal, 1 for fraud)
- `predicted_score` or `predicted_probability`: Model prediction scores
- Optional: `predicted_label` for binary predictions

Supported files:
- `xgb_predictions.csv` - XGBoost predictions
- `rf_predictions.csv` - Random Forest predictions  
- `lr_predictions.csv` - Logistic Regression predictions
- `if_predictions.csv` - Isolation Forest predictions
- `lof_predictions.csv` - Local Outlier Factor predictions
- `ae_predictions.csv` - Autoencoder predictions

## Installation and Usage

### Prerequisites
- R (version 4.0 or higher)
- Required R packages: shiny, shinydashboard, ggplot2, dplyr, DT, RColorBrewer, PRROC, markdown

### Installation
```r
# Install required packages
install.packages(c("shiny", "shinydashboard", "ggplot2", "dplyr", "DT", "RColorBrewer", "PRROC", "markdown"))
```

### Running the Application
```r
# Run the Shiny app
shiny::runApp()
```

The application will start on `http://127.0.0.1:XXXX` where XXXX is the assigned port.

## Model Performance Context

Based on the original analysis from the [Credit Card Fraud Prediction](https://github.com/itxtx/credit_card_fraud_prediction) project:

### Supervised Models
- **XGBoost**: Achieved the best overall performance with optimized hyperparameters
- **Random Forest**: Strong performance with good balance of precision and recall
- **Logistic Regression**: Baseline supervised model with reasonable performance

### Unsupervised Models
- **Isolation Forest**: Moderate performance after hyperparameter tuning
- **Local Outlier Factor**: Poor performance for this specific fraud detection task
- **Autoencoder**: Highest recall among unsupervised methods but low precision

## Key Insights from Original Analysis

The original project revealed several important findings:
- Supervised models significantly outperformed unsupervised approaches
- Class imbalance handling was crucial for meaningful evaluation
- Hyperparameter tuning with Optuna provided substantial performance improvements
- Traditional accuracy metrics were misleading due to extreme class imbalance
- Precision-Recall curves were more informative than ROC curves for this domain

## Technical Details

### Architecture
- Built with Shiny and shinydashboard for responsive UI
- Reactive programming for real-time updates
- Modular design with separate UI and server components
- Comprehensive error handling and data validation

### Performance Metrics
- Standard metrics: Precision, Recall, F1-Score, MCC
- Advanced metrics: KS Statistic, Calibration, Cumulative Gain
- Threshold analysis for optimal model tuning
- Class-wise performance breakdown

