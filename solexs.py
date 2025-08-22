import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_solexs = pd.read_parquet("/kaggle/input/isro-aditya-l1-solexs-lightcurve-fits/SoLEXS_dataset.parquet")

print(df_solexs.head())
print(df_solexs.info())
print(df_solexs.describe())

# Check for timestamp column
# Assume 'time' column exists, convert to datetime
df_solexs['time'] = pd.to_datetime(df_solexs['time'])

# Sort by time
df_solexs = df_solexs.sort_values('time').reset_index(drop=True)

# Example: basic feature extraction
# 1. Rolling statistics (window = 10 minutes, adjust to your data frequency)
df_solexs['flux_mean'] = df_solexs['flux'].rolling(window=10).mean()
df_solexs['flux_std'] = df_solexs['flux'].rolling(window=10).std()
df_solexs['flux_delta'] = df_solexs['flux'].diff()

# 2. Max/min flux in rolling window
df_solexs['flux_max'] = df_solexs['flux'].rolling(window=10).max()
df_solexs['flux_min'] = df_solexs['flux'].rolling(window=10).min()

# 3. Visualization
plt.figure(figsize=(12,5))
plt.plot(df_solexs['time'], df_solexs['flux'], label='Raw Flux')
plt.plot(df_solexs['time'], df_solexs['flux_mean'], label='Rolling Mean', color='orange')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Flux')
plt.title('Solexs Flux over Time')
plt.show()
