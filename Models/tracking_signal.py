import numpy as np
import matplotlib.pyplot as plt


def calculate_tracking_signal(df, model):
    if model in df.columns:
        # Calculate error
        df[f'{model}_Error'] = df['Sales'] - df[model]

        # Calculate MAD (Mean Absolute Deviation)
        mad = np.mean(np.abs(df[f'{model}_Error']))

        # Calculate cumulative sum of errors
        cumsum_error = np.cumsum(df[f'{model}_Error'])

        # Calculate tracking signal
        df[f'{model}_Tracking_Signal'] = cumsum_error / mad

        print(f"Tracking Signal for {model} calculated.")

        return df[f'{model}_Tracking_Signal']
    else:
        print(f"Model {model} does not exist in the dataframe.")
        return None


def plot_control_chart(df, model):
    if f'{model}_Error' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df['Week'], df[f'{model}_Error'], label=f'{model} Error', color='orange')
        plt.axhline(0, color='black', linewidth=1)
        plt.axhline(np.mean(df[f'{model}_Error']) + 3 * np.std(df[f'{model}_Error']), color='red', linestyle='--',
                    label='Upper Control Limit')
        plt.axhline(np.mean(df[f'{model}_Error']) - 3 * np.std(df[f'{model}_Error']), color='red', linestyle='--',
                    label='Lower Control Limit')
        plt.title(f'{model} Control Chart')
        plt.xlabel('Week')
        plt.ylabel('Error')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print(f"No error column found for {model}.")
