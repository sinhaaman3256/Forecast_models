import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing

# Simple Exponential Smoothing (SES)
def simple_exponential_smoothing(df, alpha=0.3):
    model = SimpleExpSmoothing(df['Sales'])
    fit = model.fit(smoothing_level=alpha, optimized=False)
    df['SES'] = fit.fittedvalues

    plt.figure(figsize=(10,6))
    plt.plot(df['Week'], df['Sales'], label='Actual Sales')
    plt.plot(df['Week'], df['SES'], label=f'SES (alpha={alpha})', color='red')
    plt.title('Simple Exponential Smoothing')
    plt.xlabel('Week')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    plt.show()

# Double Exponential Smoothing (DES)
def double_exponential_smoothing(df, alpha=0.3, beta=0.1):
    model = ExponentialSmoothing(df['Sales'], trend='add')
    fit = model.fit(smoothing_level=alpha, smoothing_trend=beta, optimized=False)
    df['DES'] = fit.fittedvalues

    plt.figure(figsize=(10,6))
    plt.plot(df['Week'], df['Sales'], label='Actual Sales')
    plt.plot(df['Week'], df['DES'], label=f'DES (alpha={alpha}, beta={beta})', color='green')
    plt.title('Double Exponential Smoothing')
    plt.xlabel('Week')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    plt.show()

# Triple Exponential Smoothing (TES)
def triple_exponential_smoothing(df, alpha=0.3, beta=0.1, gamma=0.1, seasonality_period=12):
    model = ExponentialSmoothing(df['Sales'], trend='add', seasonal='add', seasonal_periods=seasonality_period)
    fit = model.fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma, optimized=True)
    df['TES'] = fit.fittedvalues

    plt.figure(figsize=(10,6))
    plt.plot(df['Week'], df['Sales'], label='Actual Sales')
    plt.plot(df['Week'], df['TES'], label=f'TES (alpha={alpha}, beta={beta}, gamma={gamma})', color='blue')
    plt.title('Triple Exponential Smoothing')
    plt.xlabel('Week')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    plt.show()
