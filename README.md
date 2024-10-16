### DEVELOPED BY : KULASEKARAPANDIAN K
### REGISTER NO : 212222240052
### Date: 

# Ex.No: 6               HOLT WINTERS METHOD

### AIM:
To analyze Ethereum prices and forecast future prices using the Holt-Winters exponential smoothing method. The goal is to predict the closing prices for Ethereum over the next 30 business days.

### ALGORITHM:
1. Import necessary libraries: pandas, numpy, matplotlib, and ExponentialSmoothing from statsmodels.
2.Parse the datetime column and set it as the index of the DataFrame.
3. Convert the closing price column to numeric and remove rows with missing values.
4. Extract the closing price column for time series analysis.
5. Apply the Holt-Winters exponential smoothing model with additive trend and seasonal components.
6. Fit the model to the cleaned data.
7. Forecast the closing prices for the next 30 business days.
8. Plot both the historical Ethereum prices and the forecasted prices.

### PROGRAM:
```python
# Import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load the dataset
data = pd.read_csv('eth10m.csv')  # Replace with the actual path

# Convert the 'dateTime' column to datetime format and set it as the index
data['dateTime'] = pd.to_datetime(data['dateTime'])
data.set_index('dateTime', inplace=True)

# Convert 'close' column to numeric (removing invalid values)
data['close'] = pd.to_numeric(data['close'], errors='coerce')

# Drop rows with missing values in 'close' column
clean_data = data.dropna(subset=['close'])

# Extract 'close' column for time series forecasting
close_data_clean = clean_data['close']

# Perform Holt-Winters exponential smoothing
model = ExponentialSmoothing(close_data_clean, trend="add", seasonal="add", seasonal_periods=12)
fit = model.fit()

# Forecast for the next 30 business days
n_steps = 30
forecast = fit.forecast(steps=n_steps)

# Plot the original data and the forecast
plt.figure(figsize=(10, 6))
plt.plot(close_data_clean.index, close_data_clean, label='Original Data')
plt.plot(pd.date_range(start=close_data_clean.index[-1], periods=n_steps+1, freq='B')[1:], forecast, label='Forecast', color='orange')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Holt-Winters Forecast for Ethereum Prices')
plt.legend()
plt.show()

```
### OUTPUT:
#### FINAL_PREDICTION
![image](https://github.com/user-attachments/assets/0ab7310c-5a37-4dbd-8597-ca19bee11543)

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
