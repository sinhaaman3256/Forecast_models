import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

def arima_model(df, order=(1,1,1)):
    model = ARIMA(df['Sales'], order=order)
    fit = model.fit()

    df['ARIMA'] = fit.fittedvalues

    plt.figure(figsize=(10,6))
    plt.plot(df['Week'], df['Sales'], label='Actual Sales')
    plt.plot(df['Week'], df['ARIMA'], label=f'ARIMA {order}', color='blue')
    plt.title('ARIMA Model')
    plt.xlabel('Week')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    plt.show()
