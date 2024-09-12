import numpy as np

def calculate_error_metrics(df):
    models = ['MA', 'SES', 'DES', 'TES', 'ARIMA', 'LR']
    errors = {}

    for model in models:
        if model in df.columns:
            try:
                # Ensure forecasted values are 1D
                df[model] = df[model].values.flatten() if df[model].ndim > 1 else df[model]

                # Calculate the errors
                df[f'{model}_Error'] = df['Sales'] - df[model]

                # Calculate Mean Absolute Error (MAE)
                mae = np.mean(np.abs(df[f'{model}_Error']))
                print(f"Mean Absolute Error for {model}: {mae}")

                # Calculate Mean Squared Error (MSE)
                mse = np.mean(np.square(df[f'{model}_Error']))
                print(f"Mean Squared Error for {model}: {mse}")

                # Calculate Mean Absolute Percentage Error (MAPE)
                with np.errstate(divide='ignore', invalid='ignore'):
                    mape = np.mean(np.abs(df[f'{model}_Error'] / df['Sales'])) * 100
                print(f"Mean Absolute Percentage Error for {model}: {mape}%")

                # Store errors for comparison
                errors[model] = mae

            except Exception as e:
                print(f"Error calculating metrics for {model}: {e}")

    # Find the model with the lowest MAE
    best_model = min(errors, key=errors.get)
    print(f"The best model based on MAE is: {best_model}")
