import pandas as pd
from Models.moving_avg import moving_average
from Models.exp_smoothing import simple_exponential_smoothing, double_exponential_smoothing, triple_exponential_smoothing
from Models.arima import arima_model
from Models.linear_regression import linear_regression
from Models.error_analysis import calculate_error_metrics
from Models.tracking_signal import calculate_tracking_signal, plot_control_chart

import warnings
warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found.")

# Load dataset
df = pd.read_csv('Data/sales_data.csv')

# Run all models
moving_average(df, n=4)
simple_exponential_smoothing(df, alpha=0.3)
double_exponential_smoothing(df, alpha=0.3, beta=0.1)
triple_exponential_smoothing(df, alpha=0.3, beta=0.1, gamma=0.1, seasonality_period=12)
arima_model(df)
linear_regression(df)

# Perform error analysis
calculate_error_metrics(df)

# Calculate tracking signals and plot control charts
for model in ['MA', 'SES', 'DES', 'TES', 'ARIMA', 'LR']:
    calculate_tracking_signal(df, model)
    plot_control_chart(df, model)
