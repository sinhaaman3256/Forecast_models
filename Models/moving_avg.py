import pandas as pd
import matplotlib.pyplot as plt

def moving_average(df, n=4):
    df['MA'] = df['Sales'].rolling(window=n).mean()

    plt.figure(figsize=(10,6))
    plt.plot(df['Week'], df['Sales'], label='Actual Sales')
    plt.plot(df['Week'], df['MA'], label=f'Moving Average (n={n})', color='orange')
    plt.title('Moving Average Model')
    plt.xlabel('Week')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    plt.show()
