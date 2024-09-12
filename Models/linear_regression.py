import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

def linear_regression(df):
    X = np.array(df['Week']).reshape(-1, 1)
    y = df['Sales']

    model = LinearRegression()
    model.fit(X, y)
    df['LR'] = model.predict(X)

    plt.figure(figsize=(10,6))
    plt.plot(df['Week'], df['Sales'], label='Actual Sales')
    plt.plot(df['Week'], df['LR'], label='Linear Regression', color='cyan')
    plt.title('Linear Regression Model')
    plt.xlabel('Week')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    plt.show()
